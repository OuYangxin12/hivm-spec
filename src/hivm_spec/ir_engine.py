"""IR 接口引擎（T1.1）：MLIR → VIR。

**架构地位**：这是**唯一**接触 MLIR 与 bindings 的组件（D6）。其余引擎一律消费
VIR；任何引擎绕过 VIR 直接遍历 MLIR 即为架构违规。

**职责边界**（D12）：一次调用处理一份 MLIR。不接受 pass 序列、不解析
`--print-ir-*` 转储、不跨调用保持状态。

**不认识就报缺口**（FR4 / VIR 不变量 4）：描述库里没有的 op 落入 `Coverage`，
不静默丢弃、也不猜测其语义。这是本项目区别于"跑通就算过"的关键。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from hivm_spec.vir import (
    Access,
    AllocOrigin,
    Coverage,
    Effect,
    Gap,
    GapKind,
    Loc,
    SizeOrigin,
    SyncKind,
    VAlloc,
    VIRError,
    VLoop,
    VModule,
    VNode,
    VRegion,
    VSync,
)

__all__ = ["IREngine", "LowerResult", "lower_module_text"]

ENGINE_VERSION = "0.1.0"

#: MLIR location 形如 loc("file":line:col)
LOC_RE = re.compile(r'loc\("(?P<file>[^"]*)":(?P<line>\d+):(?P<col>\d+)\)')

#: memref/tensor 类型里的地址空间：#hivm.address_space<ub>
SPACE_RE = re.compile(r"#hivm\.address_space<(?P<space>[a-z0-9_]+)>")

#: shape 文本，如 memref<4x16xf32, ...> / memref<?x16xf16>
SHAPE_RE = re.compile(r"(?:memref|tensor)<(?P<shape>[^,>]+)")

#: 元素类型 → 字节数。仅列本项目当前建模所需；未知类型走 UNKNOWN_SIZE 缺口。
ELEM_BYTES = {
    "i1": 1,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "u32": 4,
    "f32": 4,
    "i64": 8,
    "u64": 8,
    "f64": 8,
}

#: op 名 → 同步语义类别。跨核与核内的区分对死锁判定至关重要。
SYNC_KINDS = {
    "hivm.hir.set_flag": SyncKind.SET_FLAG,
    "hivm.hir.wait_flag": SyncKind.WAIT_FLAG,
    "hivm.hir.pipe_barrier": SyncKind.PIPE_BARRIER,
    "hivm.hir.sync_block_set": SyncKind.SYNC_BLOCK_SET,
    "hivm.hir.sync_block_wait": SyncKind.SYNC_BLOCK_WAIT,
}

#: 控制流 op → VRegion.kind
#:
#: `builtin.module` 必须在列：主仓语料常见"带属性的内层 module"（如
#: `module attributes {dlti.target_system_spec = ...}`），若不下钻，整个内层
#: module 的全部函数会被**静默跳过**——实测 hivm-insert-nz2nd-for-debug.mlir
#: 的 3 个 func 因此全部不可见，工具却仍给出结论。
REGION_KINDS = {
    "builtin.module": "module",
    #: scope.scope 是 HIVM 的作用域容器（preload kernel 的主结构）。它不以
    #: "hivm." 开头，若按社区方言处理会**连同函数体一起静默跳过**——与
    #: builtin.module 同一类缺陷，实测会丢掉整个 kernel 的主体。
    "scope.scope": "scope",
    "scf.for": "for",
    "scf.while": "while",
    "scf.if": "if",
    "scf.forall": "for",
    "func.func": "func",
}


@dataclass
class LowerResult:
    """降级结果：VIR + 过程中的诊断。"""

    module: VModule
    #: 引擎自身遇到的问题（非 IR 问题），如属性解析失败
    engine_notes: list[str] = field(default_factory=list)


def _parse_loc(loc_str: str, fallback_file: str) -> Loc:
    m = LOC_RE.search(loc_str)
    if m:
        return Loc(
            file=m.group("file") if m.group("file") != "-" else fallback_file,
            line=int(m.group("line")),
            col=int(m.group("col")),
            raw=loc_str,
        )
    # 复杂形态（fused/callsite/unknown）保真存档：不变量 3 只要求可回溯，
    # 不要求一定是 file:line。
    return Loc(file=fallback_file, raw=loc_str or "loc(unknown)")


def _space_of(type_str: str) -> str:
    m = SPACE_RE.search(type_str)
    return m.group("space") if m else ""


def _nbytes_of(type_str: str) -> tuple[int | None, SizeOrigin, str]:
    """从类型文本推算字节数。

    动态维度（`?`）一律返回 None + UNKNOWN——实测 123/189 语料含动态 shape，
    "尺寸未知"是常态，绝不能拿 1 或 0 顶替（那会让占用结论假精确）。
    """
    m = SHAPE_RE.search(type_str)
    if not m:
        return None, SizeOrigin.UNKNOWN, ""
    shape_text = m.group("shape").strip()
    parts = shape_text.split("x")
    if len(parts) < 1:
        return None, SizeOrigin.UNKNOWN, shape_text

    elem = parts[-1]
    dims = parts[:-1]
    if elem not in ELEM_BYTES:
        return None, SizeOrigin.UNKNOWN, shape_text
    if any(d == "?" or not d.isdigit() for d in dims):
        return None, SizeOrigin.UNKNOWN, shape_text

    n = ELEM_BYTES[elem]
    for d in dims:
        n *= int(d)
    return n, SizeOrigin.STATIC_SHAPE, shape_text


class IREngine:
    """MLIR → VIR 降级器。

    需要描述库信息来判断"哪些 op 已建模"：未建模的 op 必须落 coverage。
    """

    def __init__(
        self,
        modeled_ops: set[str] | frozenset[str],
        *,
        op_effects: dict[str, tuple[Effect, ...]] | None = None,
        op_pipes: dict[str, str] | None = None,
        arch: str = "a3",
    ) -> None:
        self.modeled = set(modeled_ops)
        self.op_effects = op_effects or {}
        self.op_pipes = op_pipes or {}
        self.arch = arch
        self._counter = 0
        self._gaps: list[Gap] = []
        self._allocs: list[VAlloc] = []
        self._syncs: list[VSync] = []
        self._seen_ops: set[str] = set()
        self._notes: list[str] = []

    # -- id 分配：确定性且唯一（不变量 2） -------------------------------

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    # -- 入口 ------------------------------------------------------------

    def lower(self, mlir_module: Any, source: str) -> LowerResult:
        self._counter = 0
        self._gaps = []
        self._allocs = []
        self._syncs = []
        self._seen_ops = set()
        self._notes = []

        top_regions: list[VRegion] = []
        for region in mlir_module.operation.regions:
            for block in region.blocks:
                for op in block.operations:
                    r = self._lower_op_as_region(op, source)
                    if r is not None:
                        top_regions.append(r)

        modeled_present = tuple(sorted(self._seen_ops & self.modeled))
        coverage = Coverage(gaps=tuple(self._gaps), modeled_ops=modeled_present)

        vmodule = VModule(
            source=source,
            items=tuple(top_regions),
            allocs=tuple(self._allocs),
            syncs=tuple(self._syncs),
            coverage=coverage,
            arch=self.arch,
            engine_version=ENGINE_VERSION,
        )
        return LowerResult(module=vmodule, engine_notes=list(self._notes))

    # -- 区域 ------------------------------------------------------------

    def _lower_op_as_region(self, op: Any, source: str) -> VRegion | None:
        """把带区域的 op（func/scf.*）降为 VRegion。非区域 op 返回 None。"""
        name = op.operation.name
        kind = REGION_KINDS.get(name)
        if kind is None:
            return None
        return self._build_region(op, kind, source)

    def _build_region(self, op: Any, kind: str, source: str) -> VRegion:
        loc = _parse_loc(str(op.location), source)
        rid = self._next_id("r")

        # 函数参数形态的片上 buffer 必须登记：它们同样占用空间，只是由调用方
        # 分配。忽略它们会造成假阴性（实测 annotate-vf-alias.mlir 的 3 个 UB
        # buffer 全是函数参数，漏掉后该 kernel 占用被算作 0 并给出 OK）。
        if kind == "func":
            self._record_func_args(op, loc)

        # 单一 items 序列保持**程序序**：节点与子区域按出现顺序交错。
        # 分成两个列表会丢失相对位置，M2 的顺序判定就会失真。
        items: list[VNode | VRegion] = []

        for region in op.operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    sub_kind = REGION_KINDS.get(inner.operation.name)
                    if sub_kind is not None:
                        items.append(self._build_region(inner, sub_kind, source))
                    else:
                        node = self._lower_node(inner, source)
                        if node is not None:
                            items.append(node)

        loop = self._extract_loop(op, kind) if kind in ("for", "while") else None
        attrs = self._extract_attrs(op)

        return VRegion(
            id=rid,
            kind=kind,
            loc=loc,
            items=tuple(items),
            loop=loop,
            attrs=attrs,
        )

    def _extract_loop(self, op: Any, kind: str) -> VLoop:
        iv = "iv"
        lower = upper = step = ""
        trip: int | None = None
        try:
            if kind == "for" and op.operation.regions:
                blocks = list(op.operation.regions[0].blocks)
                if blocks and list(blocks[0].arguments):
                    iv = str(blocks[0].arguments[0]).split("=")[0].strip() or "iv"
            operands = list(op.operation.operands)
            if len(operands) >= 3:
                lower, upper, step = (str(operands[i]) for i in range(3))
                trip = self._static_trip_count(lower, upper, step)
        except (IndexError, AttributeError) as exc:
            self._notes.append(f"循环信息提取不完整（{type(exc).__name__}）：{op.operation.name}")
        return VLoop(iv=iv, trip_count=trip, lower=lower, upper=upper, step=step, kind=kind)

    @staticmethod
    def _static_trip_count(lower: str, upper: str, step: str) -> int | None:
        """仅当三者都是字面常量时给出静态迭代次数；否则 None（动态）。"""
        nums = []
        for s in (lower, upper, step):
            m = re.search(r"arith\.constant\s+(-?\d+)", s)
            if not m:
                return None
            nums.append(int(m.group(1)))
        lo, hi, st = nums
        if st <= 0:
            return None
        return max(0, -(-(hi - lo) // st))

    def _extract_attrs(self, op: Any) -> dict[str, str]:
        attrs: dict[str, str] = {}
        try:
            for named in op.operation.attributes:
                attrs[named.name] = str(named.attr)
        except (AttributeError, TypeError):
            pass
        return attrs

    # -- 节点 ------------------------------------------------------------

    def _lower_node(self, op: Any, source: str) -> VNode | None:
        name = op.operation.name
        loc = _parse_loc(str(op.location), source)

        # memref.alloc → VAlloc（M1 占用分析的基本单位）
        if name in ("memref.alloc", "memref.alloca"):
            self._record_alloc(op, loc)
            return None

        # memref_ext.alloc_workspace：从 workspace 参数（GM）划出一块 buffer。
        # 它是**真实的分配点**，忽略会低估占用；但 workspace 在 GM，不参与
        # 片上容量判定，故单独登记（space=gm）而非与 memref.alloc 混同。
        if name == "memref_ext.alloc_workspace":
            self._record_workspace_alloc(op, loc)
            return None

        # annotation.mark：编译器标注（T1.3 尺寸策略的"标注优先"来源）。
        # 消费 hivm.multi_buffer；其余标注记录后忽略（不静默）。
        if name == "annotation.mark":
            self._apply_annotation(op, loc)
            return None

        # 只对 hivm op 建节点；社区方言（arith/memref/scf 等）作为支撑结构，
        # 不进 VIR 节点流——它们不承载 HIVM 语义，纳入只会放大遍历噪声。
        if not name.startswith("hivm."):
            return None

        self._seen_ops.add(name)
        nid = self._next_id("n")

        if name not in self.modeled:
            self._gaps.append(
                Gap(
                    kind=GapKind.UNMODELED_OP,
                    op=name,
                    detail=f"描述库未建模 {name}，其效应未纳入分析",
                    loc=loc,
                )
            )

        effects = self._effects_for(op, name)
        node = VNode(
            id=nid,
            op=name,
            loc=loc,
            operands=tuple(str(o) for o in op.operation.operands),
            results=tuple(str(r) for r in op.operation.results),
            effects=effects,
            pipe=self.op_pipes.get(name, ""),
            attrs=self._extract_attrs(op),
        )

        if name in SYNC_KINDS:
            self._record_sync(node, name, op)

        return node

    def _effects_for(self, op: Any, name: str) -> tuple[Effect, ...]:
        """按描述声明生成效应实例。

        描述声明的是"读第 0 个操作数、写第 0 个结果"这类**结构**；此处把它
        绑定到具体的操作数/类型上，得到带 space 与字节数的实例。
        """
        declared = self.op_effects.get(name)
        if not declared:
            return ()

        out: list[Effect] = []
        operand_types = [str(o.type) for o in op.operation.operands]
        for eff in declared:
            space = eff.space
            nbytes = eff.nbytes
            if space == "@from_type" or not space:
                # 从操作数类型提取地址空间
                spaces = [_space_of(t) for t in operand_types]
                space = next((s for s in spaces if s), "")
                if not space:
                    self._gaps.append(
                        Gap(
                            kind=GapKind.UNRECOGNIZED_STRUCTURE,
                            detail=f"{name} 的操作数类型未标注 #hivm.address_space，"
                            "无法确定效应所属空间",
                            op=name,
                        )
                    )
                    continue
            if nbytes is None and operand_types:
                nbytes, origin, _ = _nbytes_of(operand_types[0])
                if origin is SizeOrigin.UNKNOWN:
                    self._gaps.append(
                        Gap(
                            kind=GapKind.UNKNOWN_SIZE,
                            detail=f"{name} 的访问尺寸无法静态确定（动态 shape 或未知元素类型）",
                            op=name,
                        )
                    )
            out.append(Effect(access=eff.access, space=space, target=eff.target, nbytes=nbytes))
        return tuple(out)

    def _record_func_args(self, op: Any, loc: Loc) -> None:
        """把 memref 类型的函数参数登记为 FUNC_ARG 来源的 VAlloc。

        只登记带 `#hivm.address_space` 标注的参数：无标注的 memref 通常是
        host 侧内存或未指定空间，不参与片上占用判定。
        """
        try:
            regions = list(op.operation.regions)
            if not regions:
                return
            blocks = list(regions[0].blocks)
            if not blocks:
                return
            args = list(blocks[0].arguments)
        except (AttributeError, IndexError):
            return

        for arg in args:
            type_str = str(arg.type)
            space = _space_of(type_str)
            if not space:
                continue
            nbytes, origin, shape_text = _nbytes_of(type_str)
            name = self._next_id("arg")
            if origin is SizeOrigin.UNKNOWN:
                self._gaps.append(
                    Gap(
                        kind=GapKind.UNKNOWN_SIZE,
                        detail=f"函数参数 buffer {name}（{shape_text or type_str}）"
                        "尺寸无法静态确定",
                        loc=loc,
                    )
                )
            self._allocs.append(
                VAlloc(
                    name=name,
                    space=space,
                    loc=loc,
                    nbytes=nbytes,
                    size_origin=origin,
                    shape_text=shape_text,
                    value=str(arg),
                    origin=AllocOrigin.FUNC_ARG,
                )
            )

    def _record_workspace_alloc(self, op: Any, loc: Loc) -> None:
        """登记 memref_ext.alloc_workspace 划出的 workspace buffer。

        workspace 从 GM 参数划出，故 space=gm（provisional 判断，依据
        hacc.arg_type<workspace> 的主仓惯例）。它不参与 ub/cbuf 容量判定，
        但必须可见——否则 fixpipe 的落点 buffer 在结果里查无来历。
        """
        try:
            result_type = str(op.operation.results[0].type)
            result_value = str(op.operation.results[0])
        except (IndexError, AttributeError):
            self._gaps.append(
                Gap(
                    kind=GapKind.UNRECOGNIZED_STRUCTURE,
                    detail="alloc_workspace 无结果类型，无法确定分配尺寸",
                    loc=loc,
                )
            )
            return
        nbytes, origin, shape_text = _nbytes_of(result_type)
        name = self._next_id("ws")
        if origin is SizeOrigin.UNKNOWN:
            self._gaps.append(
                Gap(
                    kind=GapKind.UNKNOWN_SIZE,
                    detail=f"workspace buffer {name}（{shape_text or result_type}）"
                    "尺寸无法静态确定",
                    loc=loc,
                )
            )
        self._allocs.append(
            VAlloc(
                name=name,
                space="gm",
                loc=loc,
                nbytes=nbytes,
                size_origin=origin,
                shape_text=shape_text,
                value=result_value,
            )
        )

    def _apply_annotation(self, op: Any, loc: Loc) -> None:
        """消费 annotation.mark 标注。

        hivm.multi_buffer = N：buffer 复制 N 份供流水线交替，总占用 ×N。
        标注作用于 alloc 的**结果 value**，按 value 名精确匹配到 VAlloc；
        匹配不到的标注登记缺口（可能是引擎没建模的分配形态）。
        """
        attrs = self._extract_attrs(op)
        try:
            target = str(op.operation.operands[0])
        except (IndexError, AttributeError):
            return

        mb = attrs.get("hivm.multi_buffer")
        if mb is None:
            # 其余标注（如 cv_pipeline_lazy_load）与占用无关，记录即可
            self._notes.append(
                f"annotation.mark（{loc.describe()}）：标注 {sorted(attrs)} 与占用无关，忽略"
            )
            return
        try:
            # _extract_attrs 把属性字符串化为 "2 : i32" 形态，须剥掉类型后缀
            n = int(str(mb).split(":")[0].strip())
        except (TypeError, ValueError):
            self._gaps.append(
                Gap(
                    kind=GapKind.UNRECOGNIZED_STRUCTURE,
                    detail=f"annotation.mark 的 multi_buffer 值无法解析：{mb!r}",
                    loc=loc,
                )
            )
            return

        for i, alloc in enumerate(self._allocs):
            if alloc.value == target:
                updated = VAlloc(
                    name=alloc.name,
                    space=alloc.space,
                    loc=alloc.loc,
                    nbytes=alloc.nbytes,
                    size_origin=alloc.size_origin,
                    shape_text=alloc.shape_text,
                    value=alloc.value,
                    origin=alloc.origin,
                    multi_buffer=n,
                )
                self._allocs[i] = updated
                self._notes.append(
                    f"buffer {alloc.name} 命中 hivm.multi_buffer={n}，占用按 {alloc.nbytes}×{n} 计"
                )
                return

        self._gaps.append(
            Gap(
                kind=GapKind.UNRECOGNIZED_STRUCTURE,
                detail=f"annotation.mark 引用的 value {target} 未对应已登记的 buffer，"
                "multi_buffer 标注未生效",
                loc=loc,
            )
        )

    def _record_alloc(self, op: Any, loc: Loc) -> None:
        try:
            result_type = str(op.operation.results[0].type)
            result_value = str(op.operation.results[0])
        except (IndexError, AttributeError):
            self._gaps.append(
                Gap(
                    kind=GapKind.UNRECOGNIZED_STRUCTURE,
                    detail="alloc 无结果类型，无法确定分配尺寸",
                )
            )
            return

        space = _space_of(result_type)
        nbytes, origin, shape_text = _nbytes_of(result_type)
        name = self._next_id("buf")

        if not space:
            # 无 #hivm.address_space 标注：通常是 host/未指定空间的 memref，
            # 不参与片上占用。但**不能静默丢弃**——若它其实是片上 buffer 而
            # 只是标注缺失，占用就会被低估。登记缺口让人来判断（FR4）。
            space = "unannotated"
            self._gaps.append(
                Gap(
                    kind=GapKind.UNRECOGNIZED_STRUCTURE,
                    detail=f"alloc {name}（{result_type}）无 #hivm.address_space 标注，"
                    "未计入任何片上空间的占用；若它实为片上 buffer，占用将被低估",
                    loc=loc,
                )
            )

        if origin is SizeOrigin.UNKNOWN:
            self._gaps.append(
                Gap(
                    kind=GapKind.UNKNOWN_SIZE,
                    detail=f"alloc {name}（{shape_text or result_type}）尺寸无法静态确定；"
                    "占用判定须降级或依赖测试参数化",
                    loc=loc,
                )
            )

        self._allocs.append(
            VAlloc(
                name=name,
                space=space,
                loc=loc,
                nbytes=nbytes,
                size_origin=origin,
                shape_text=shape_text,
                value=result_value,
            )
        )

    def _record_sync(self, node: VNode, name: str, op: Any) -> None:
        kind = SYNC_KINDS[name]
        event_id: int | None = None
        core = ""
        pipe = ""

        # 事件 id 来自属性（flag = 15）或 assembly 里的 <EVENT_IDn>
        for key in ("flag", "event_id", "eventId"):
            if key in node.attrs:
                m = re.search(r"-?\d+", node.attrs[key])
                if m:
                    event_id = int(m.group())
                break
        if event_id is None:
            m = re.search(r"EVENT_ID(\d+)", str(op))
            if m:
                event_id = int(m.group(1))

        m_core = re.search(r"<(CUBE|VECTOR|AIC|AIV)>", str(op))
        if m_core:
            core = m_core.group(1)
        m_pipe = re.search(r"<(PIPE_[A-Z0-9_]+)>", str(op))
        if m_pipe:
            pipe = m_pipe.group(1)

        self._syncs.append(
            VSync(
                node_id=node.id,
                kind=kind,
                loc=node.loc,
                event_id=event_id,
                pipe=pipe or node.pipe,
                core=core,
            )
        )


def lower_module_text(
    text: str,
    modeled_ops: set[str] | frozenset[str],
    *,
    source: str = "<memory>",
    op_effects: dict[str, tuple[Effect, ...]] | None = None,
    op_pipes: dict[str, str] | None = None,
    arch: str = "a3",
) -> LowerResult:
    """便捷入口：MLIR 文本 → VIR（需要 bindings）。"""
    from hivm_spec.bindings import load_bindings

    handle = load_bindings()
    try:
        module = handle.parse_module(text)
    except VIRError:
        raise
    except Exception as exc:
        # 解析失败是**被验证 IR 的问题**（FR7），必须与环境错误（BindingsError）
        # 分源。否则 CLI 的 except BindingsError 会漏接，agent 看到的是裸 traceback。
        raise VIRError(f"MLIR 解析失败：{exc}") from exc
    engine = IREngine(modeled_ops, op_effects=op_effects, op_pipes=op_pipes, arch=arch)
    return engine.lower(module, source)


def effects_from_spec(spec: Any) -> tuple[dict[str, tuple[Effect, ...]], dict[str, str]]:
    """把描述里的效应声明转成引擎可用的形式。

    描述层的 `EffectDecl` 是**结构声明**（读哪个参数），引擎需要的是可绑定到
    具体操作数的形式。同步效应不产生内存 Effect，故此处跳过——它们通过
    `SYNC_KINDS` 走 `VSync` 路径。
    """
    from hivm_spec.spec import EffectKind

    access_map = {
        EffectKind.READ: Access.READ,
        EffectKind.WRITE: Access.WRITE,
        EffectKind.COND_WRITE: Access.WRITE,
    }
    effects: dict[str, tuple[Effect, ...]] = {}
    pipes: dict[str, str] = {}
    for op in spec.ops:
        items: list[Effect] = []
        for decl in op.effects:
            access = access_map.get(decl.kind)
            if access is None:
                continue  # 同步效应走 VSync
            items.append(
                Effect(access=access, space=decl.space or "@from_type", target=decl.target)
            )
        if items:
            effects[op.op] = tuple(items)
        if op.pipe:
            pipes[op.op] = op.pipe
    return effects, pipes
