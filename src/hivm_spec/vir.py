"""VIR 核心数据契约（T0.0 / D6）。

VIR 是引擎间**唯一**数据契约：IR 接口引擎是唯一接触 MLIR 与 bindings 的组件，
产出 VIR；占用引擎、状态机调度引擎、值执行引擎、视图渲染器一律只消费 VIR。

本模块是**纯 Python**，不得 import `bishengir`（D7）——这保证核心层测试不被
稀缺的 cp310 绑定环境阻塞，也使 VIR 可被手写微例直接构造（D13 的 L0 层）。

四条不变量（框架 §6.1），本模块以类型与运行时校验共同保障：

1. **构造后不可变** —— 全部为 `frozen=True` dataclass，容器一律 `tuple`；
   派生分析必须旁挂（`Derived`），不得原地改写。保障 FR8 确定性与多策略
   并行探索的隔离。
2. **节点顺序确定性** —— 顺序即 `tuple` 顺序，由构造者按拓扑序给定；
   `VModule.node_order()` 提供稳定遍历，同一输入逐字节稳定。
3. **可回溯源位置** —— 每个 `VNode` 必须携带非空 `loc`（FR5 可诊断性的
   物理基础），构造时校验。
4. **未识别必入 coverage** —— 不认识的 op/结构必须落入 `Coverage`，
   不得静默丢弃（FR4），直接驱动 `COVERAGE_GAP`。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

__all__ = [
    "Access",
    "AllocOrigin",
    "Coverage",
    "Effect",
    "GapKind",
    "Loc",
    "SizeOrigin",
    "SyncKind",
    "VAlloc",
    "VIRError",
    "VLoop",
    "VModule",
    "VNode",
    "VRegion",
    "VSync",
    "VTrace",
    "ValueSlot",
]


class VIRError(ValueError):
    """VIR 契约违规。

    独立异常类型：违反契约是**架构错误**，不应与被验证 IR 的语义错误混淆
    （FR7 要求区分"工具/契约问题"与"IR 问题"）。
    """


# --------------------------------------------------------------------------
# 基础标识
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Loc:
    """MLIR 源位置（不变量 3 的载体）。

    `file` 为空字符串是允许的（内存构造的语料），但 `describe()` 必须始终
    给出人类可定位的字符串——诊断信息不允许出现空位置。
    """

    file: str = ""
    line: int = 0
    col: int = 0
    #: MLIR 原始 location 字符串（fused/callsite 等复杂形态的保真存档）
    raw: str = ""

    def describe(self) -> str:
        if self.file and self.line:
            return f"{self.file}:{self.line}:{self.col}"
        if self.raw:
            return self.raw
        return "<unknown-loc>"

    def __bool__(self) -> bool:
        return bool(self.file or self.line or self.raw)


class Access(Enum):
    """效应的访问方向。"""

    READ = "read"
    WRITE = "write"
    #: 读改写（如累加类 op），生存期分析需视为同时读写
    READ_WRITE = "read_write"


class SyncKind(Enum):
    """同步节点语义类别（M2 状态机引擎消费）。"""

    SET_FLAG = "set_flag"
    WAIT_FLAG = "wait_flag"
    PIPE_BARRIER = "pipe_barrier"
    SYNC_BLOCK_SET = "sync_block_set"
    SYNC_BLOCK_WAIT = "sync_block_wait"


class SizeOrigin(Enum):
    """`VAlloc` 尺寸的来源（provenance）。

    实测主仓 123/189 个测试文件含动态 shape，故尺寸来源必须显式建模——
    "尺寸未知"是一等公民，不能假装静态（否则占用结论会假精确）。
    """

    STATIC_SHAPE = "static_shape"
    COMPILER_ANNOTATION = "compiler_annotation"
    TEST_PARAMETERIZED = "test_parameterized"
    UNKNOWN = "unknown"


class AllocOrigin(Enum):
    """buffer 的**来源**：本函数内分配，还是由外部传入。

    区分二者是必要的：函数参数形态的片上 buffer 同样占用空间，但其生存期
    覆盖整个函数（调用方持有），且不由本函数决定。把它们与 `memref.alloc`
    混为一谈会算错生存期；而完全忽略它们会造成**假阴性**——实测主仓
    `annotate-vf-alias.mlir` 的 3 个 UB buffer 全是函数参数，若只看 alloc
    则该 kernel 的占用被算作 0 并给出 OK。
    """

    #: 函数体内 memref.alloc / alloca
    LOCAL_ALLOC = "local_alloc"
    #: 函数参数（调用方分配，生存期覆盖全函数）
    FUNC_ARG = "func_arg"


class GapKind(Enum):
    """覆盖缺口类别（FR4）。"""

    UNMODELED_OP = "unmodeled_op"
    UNRECOGNIZED_STRUCTURE = "unrecognized_structure"
    UNKNOWN_SIZE = "unknown_size"
    UNTRUSTED_DESCRIPTION = "untrusted_description"


# --------------------------------------------------------------------------
# 效应与值槽位
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Effect:
    """描述解析出的单条效应：对某地址空间的读/写。

    M1 的占用与生存期分析只需效应级信息（D3），不需要值语义——这是
    "80% 收益极廉价"的技术基础。
    """

    access: Access
    space: str
    #: 被访问的 buffer 标识（`VAlloc.name`），跨 op 关联生存期
    target: str = ""
    #: 访问字节数；None 表示尺寸未知（须同时登记 UNKNOWN_SIZE 缺口）
    nbytes: int | None = None

    def __post_init__(self) -> None:
        if not self.space:
            raise VIRError("Effect.space 不得为空")
        if self.nbytes is not None and self.nbytes < 0:
            raise VIRError(f"Effect.nbytes 不得为负：{self.nbytes}")


@dataclass(frozen=True, slots=True)
class ValueSlot:
    """值槽位：具体/符号执行的载体（D8 为 M3/M4 预留）。

    M0/M1/M2 一律留空——但**契约里必须先有位置**，否则 M3 的值执行引擎
    只能另起一套遍历结构，重演 D6 要消灭的"双源真理"。
    """

    #: "concrete" | "symbolic" | "unbound"
    mode: str = "unbound"
    #: 具体模式下的值哈希（逐 op trace 与首发散点定位用）
    value_hash: str = ""
    #: 符号模式下的表达式标识（M4）
    symbol: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("concrete", "symbolic", "unbound"):
            raise VIRError(f"ValueSlot.mode 非法：{self.mode!r}")


# --------------------------------------------------------------------------
# 节点
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VNode:
    """一个操作节点。

    不变量 3：`loc` 必须非空——诊断信息若无源位置，FR5 的可诊断性就是空谈。
    """

    #: 稳定节点 id（同一输入下确定性；构造者负责唯一性）
    id: str
    #: MLIR op 全名，如 "hivm.hir.vadd"
    op: str
    loc: Loc
    operands: tuple[str, ...] = ()
    results: tuple[str, ...] = ()
    effects: tuple[Effect, ...] = ()
    #: pipe 归属（如 "PIPE_V"/"PIPE_MTE2"）；空表示描述未标注
    pipe: str = ""
    #: 值槽位（M3+ 使用）
    value: ValueSlot = field(default_factory=ValueSlot)
    #: 该 op 在描述库中的 trust 级别，用于结论降级标注
    trust: str = ""
    #: 透传属性（保真存档，引擎按需读取；不参与语义判定）
    attrs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise VIRError("VNode.id 不得为空")
        if not self.op:
            raise VIRError(f"VNode.op 不得为空（id={self.id}）")
        if not self.loc:
            raise VIRError(
                f"VNode.loc 不得为空（id={self.id}, op={self.op}）："
                "不变量 3 要求每个节点可回溯 MLIR 源位置"
            )


@dataclass(frozen=True, slots=True)
class VAlloc:
    """片上/全局内存分配点（M1 占用与生存期分析的基本单位）。"""

    name: str
    space: str
    loc: Loc
    #: 字节数；None 表示未知（须配 SizeOrigin.UNKNOWN + coverage 条目）
    nbytes: int | None = None
    size_origin: SizeOrigin = SizeOrigin.UNKNOWN
    #: 原始 shape 文本（如 "4x16x16xf32" 或 "?x16xf16"），保真存档
    shape_text: str = ""
    #: buffer 来源：本地分配还是函数参数（影响生存期口径）
    origin: AllocOrigin = AllocOrigin.LOCAL_ALLOC
    #: 编译器多缓冲标注（`annotation.mark {hivm.multi_buffer = N}`）。
    #: 多缓冲把同一 buffer 复制 N 份供流水线交替使用，**总占用 = nbytes × N**，
    #: 且整个环跨循环存活。默认 1 = 无多缓冲。来源必须是真实 IR 标注，
    #: 不得由分析器自行推断（T1.3：标注优先于推断，来源随结论输出）。
    multi_buffer: int = 1
    #: 该 alloc 结果的 SSA value 文本。生存期分析靠它把 buffer 与节点操作数
    #: 精确关联——若改用"按出现顺序对齐"的近似，关联一旦错位，生存期就会算错，
    #: 而错误方向不可控（可能低估峰值把溢出判成 OK）。
    value: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise VIRError("VAlloc.name 不得为空")
        if not self.space:
            raise VIRError(f"VAlloc.space 不得为空（name={self.name}）")
        if self.nbytes is None and self.size_origin is SizeOrigin.STATIC_SHAPE:
            raise VIRError(
                f"VAlloc({self.name}) 声明 STATIC_SHAPE 但 nbytes 为 None——尺寸来源与实际信息矛盾"
            )
        if self.nbytes is not None and self.nbytes < 0:
            raise VIRError(f"VAlloc({self.name}).nbytes 不得为负")
        if self.multi_buffer < 1:
            raise VIRError(
                f"VAlloc({self.name}).multi_buffer 必须 ≥1（1=无多缓冲），实为 {self.multi_buffer}"
            )

    @property
    def effective_nbytes(self) -> int | None:
        """计入多缓冲后的实际占用字节数。"""
        if self.nbytes is None:
            return None
        return self.nbytes * self.multi_buffer


@dataclass(frozen=True, slots=True)
class VSync:
    """同步事件节点（M2 状态机引擎消费）。

    真实死锁案例（CreatePreload stage-major）的教训：**不能只看 set/wait
    总数配平**，顺序不可行时计数照样平衡。故本结构保留节点在序列中的
    位置（通过 `node_id` 关联 `VModule.node_order()`）而非仅计数。
    """

    node_id: str
    kind: SyncKind
    loc: Loc
    #: 事件/flag id；None 表示未标注或不适用（如部分 pipe_barrier）
    event_id: int | None = None
    #: 所属 pipe（intra-core 同步）
    pipe: str = ""
    #: 核归属（如 "AIC"/"AIV"），跨核同步判定必需
    core: str = ""
    #: 配对候选（同 flag 的反向节点 id）；由 M2 分析填充旁挂结构，
    #: 此处仅在 IR 已显式标注时记录
    pair_candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            raise VIRError("VSync.node_id 不得为空")


# --------------------------------------------------------------------------
# 控制流
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VLoop:
    """循环的归一化表示（scf.for / scf.while）。

    迭代域可为静态或参数化符号——实测 123/189 语料含动态 shape，
    故 `trip_count is None` 是常态而非异常。
    """

    #: 归一化后的循环变量名
    iv: str
    #: 静态迭代次数；None 表示动态/未知
    trip_count: int | None = None
    lower: str = ""
    upper: str = ""
    step: str = ""
    #: 迭代参数与 yield 的绑定（iter_arg -> yielded value）
    iter_args: tuple[tuple[str, str], ...] = ()
    #: "for" | "while"
    kind: str = "for"

    def __post_init__(self) -> None:
        if not self.iv:
            raise VIRError("VLoop.iv 不得为空")
        if self.kind not in ("for", "while"):
            raise VIRError(f"VLoop.kind 非法：{self.kind!r}")
        if self.trip_count is not None and self.trip_count < 0:
            raise VIRError(f"VLoop.trip_count 不得为负：{self.trip_count}")


@dataclass(frozen=True, slots=True)
class VRegion:
    """控制流区域：函数体、scf.for/if/while 的 body、scope.scope 等。

    区域嵌套构成 VIR 的骨架；`nodes` 与 `regions` 的 tuple 顺序即
    确定性遍历顺序（不变量 2）。
    """

    id: str
    #: "module" | "func" | "for" | "if" | "while" | "scope" | "block"
    kind: str
    loc: Loc
    #: **程序序**的子项序列（节点与子区域交错），是顺序的唯一真源。
    #: 分离的 nodes/regions 元组会丢失两者的相对位置——若 wait 在循环体内、
    #: set 在循环之后，"先节点再子区域"的遍历会把 set 排到 wait 之前，
    #: 直接导致 M2 得出相反的死锁结论。
    items: tuple[VNode | VRegion, ...] = ()
    #: kind == "for"/"while" 时的循环信息
    loop: VLoop | None = None
    #: 区域级属性（如 hivm.preload_num、loop_core_type）
    attrs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise VIRError("VRegion.id 不得为空")
        if not self.kind:
            raise VIRError(f"VRegion.kind 不得为空（id={self.id}）")
        if self.kind in ("for", "while") and self.loop is None:
            raise VIRError(f"VRegion({self.id}) kind={self.kind} 但缺 loop 信息")
        if self.loop is not None and self.kind not in ("for", "while"):
            raise VIRError(f"VRegion({self.id}) kind={self.kind} 不应携带 loop")

    @property
    def nodes(self) -> tuple[VNode, ...]:
        """本区域的直接节点（派生视图，不含子区域）。"""
        return tuple(i for i in self.items if isinstance(i, VNode))

    @property
    def regions(self) -> tuple[VRegion, ...]:
        """本区域的直接子区域（派生视图）。"""
        return tuple(i for i in self.items if isinstance(i, VRegion))

    def walk_nodes(self) -> Iterator[VNode]:
        """按**程序序**深度优先遍历本区域及子区域的节点（不变量 2）。

        顺序即 `items` 的顺序：遇到子区域就进入，出来后继续。这保证
        "循环体内的 op 排在循环之后的 op 之前"，M2 的同步顺序判定才成立。
        """
        for item in self.items:
            if isinstance(item, VNode):
                yield item
            else:
                yield from item.walk_nodes()

    def walk_regions(self) -> Iterator[VRegion]:
        yield self
        for r in self.regions:
            yield from r.walk_regions()


# --------------------------------------------------------------------------
# 覆盖缺口
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gap:
    """单条覆盖缺口。"""

    kind: GapKind
    detail: str
    loc: Loc = field(default_factory=Loc)
    #: 相关 op 名（unmodeled_op 时必填）
    op: str = ""

    def __post_init__(self) -> None:
        if not self.detail:
            raise VIRError("Gap.detail 不得为空——缺口必须可读，否则等于静默")
        if self.kind is GapKind.UNMODELED_OP and not self.op:
            raise VIRError("GapKind.UNMODELED_OP 必须标明 op 名")


@dataclass(frozen=True, slots=True)
class Coverage:
    """覆盖报告（不变量 4 的载体，直接驱动 `COVERAGE_GAP`）。"""

    gaps: tuple[Gap, ...] = ()
    #: 已建模并实际出现的 op（去重后按名排序，供覆盖率统计）
    modeled_ops: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        """无缺口才算完整覆盖——L1 语料的准入门槛（D13）。"""
        return not self.gaps

    def unmodeled_ops(self) -> tuple[str, ...]:
        seen = {g.op for g in self.gaps if g.kind is GapKind.UNMODELED_OP and g.op}
        return tuple(sorted(seen))


# --------------------------------------------------------------------------
# 执行轨迹钩子
# --------------------------------------------------------------------------


@runtime_checkable
class VTrace(Protocol):
    """执行轨迹回调接口（D8 为 M3 解释执行预留）。

    定义在 VIR 契约中而非 M3 私有：值哈希 trace 与首发散点定位是 D8 的
    不可裁剪能力，其接口必须与 VIR 同期定稿，否则 M3 会另建遍历结构。
    """

    def on_node(self, node: VNode, values: Mapping[str, ValueSlot]) -> None:
        """每个节点执行后回调（逐 op 值哈希的落点）。"""
        ...

    def on_region_enter(self, region: VRegion) -> None: ...

    def on_region_exit(self, region: VRegion) -> None: ...


# --------------------------------------------------------------------------
# 根结构
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VModule:
    """VIR 根：一份 MLIR 输入的完整归一化表示。

    D12：一次工具调用对应一份（等价验证为两份）`VModule`；VIR 不表示
    pass 序列。
    """

    #: 来源标识（文件路径或语料 id）
    source: str
    #: 顶层区域（通常每个 func.func 一个）。命名与 VRegion.items 一致，
    #: 使"程序序容器"在两层是同一个概念。
    items: tuple[VRegion, ...] = ()
    allocs: tuple[VAlloc, ...] = ()
    syncs: tuple[VSync, ...] = ()
    coverage: Coverage = field(default_factory=Coverage)
    #: 目标架构（"a3"/"a5"），影响容量常量（OD7）
    arch: str = ""
    #: 产出该 VIR 的引擎版本，供结论审计（与 spec_hash 分离，见框架 §7.3）
    engine_version: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise VIRError("VModule.source 不得为空——结论必须可追溯输入")
        self._check_unique_node_ids()
        self._check_sync_refs()
        self._check_alloc_refs()

    # -- 构造期校验 --------------------------------------------------------

    def _check_unique_node_ids(self) -> None:
        seen: set[str] = set()
        for n in self.walk_nodes():
            if n.id in seen:
                raise VIRError(
                    f"VNode.id 重复：{n.id!r}——节点 id 必须唯一，"
                    "否则 VSync/诊断的节点引用会指向歧义目标"
                )
            seen.add(n.id)

    def _check_sync_refs(self) -> None:
        ids = {n.id for n in self.walk_nodes()}
        for s in self.syncs:
            if s.node_id not in ids:
                raise VIRError(
                    f"VSync.node_id={s.node_id!r} 未出现在任何区域中——"
                    "同步节点必须对应真实 op（否则时序判定建立在幻影节点上）"
                )

    def _check_alloc_refs(self) -> None:
        names = {a.name for a in self.allocs}
        if len(names) != len(self.allocs):
            raise VIRError("VAlloc.name 重复——buffer 标识必须唯一")

    # -- 遍历（不变量 2） --------------------------------------------------

    @property
    def regions(self) -> tuple[VRegion, ...]:
        """顶层区域（派生视图）。"""
        return self.items

    def walk_nodes(self) -> Iterator[VNode]:
        """全模块**程序序**遍历。"""
        for r in self.items:
            yield from r.walk_nodes()

    def walk_regions(self) -> Iterator[VRegion]:
        for r in self.items:
            yield from r.walk_regions()

    def node_order(self) -> tuple[str, ...]:
        """节点 id 的确定性序列——M2 顺序判定的基准。"""
        return tuple(n.id for n in self.walk_nodes())

    def node_by_id(self, node_id: str) -> VNode:
        for n in self.walk_nodes():
            if n.id == node_id:
                return n
        raise VIRError(f"节点不存在：{node_id!r}")

    def sync_order(self) -> tuple[VSync, ...]:
        """同步节点按其在 `node_order()` 中的位置排序。

        这是"顺序而非计数"的技术落点：真实死锁案例中 set/wait 计数配平
        但顺序不可行，只有序列化视图能暴露"3 个 wait 先于首个 set"。
        """
        pos = {nid: i for i, nid in enumerate(self.node_order())}
        return tuple(sorted(self.syncs, key=lambda s: pos[s.node_id]))

    # -- 确定性指纹 --------------------------------------------------------

    def fingerprint(self) -> str:
        """VIR 结构指纹（FR8）：同一输入必得同一值。

        仅覆盖影响判定的结构（节点顺序/op/效应/alloc/sync/coverage），
        不含 `source` 与 `engine_version`——前者是路径，后者的变化应通过
        独立字段体现而非混入语义指纹（同框架 §7.3 对 spec_hash 的处理）。
        """
        h = hashlib.sha256()
        for n in self.walk_nodes():
            h.update(f"N|{n.id}|{n.op}|{n.pipe}|".encode())
            for e in n.effects:
                h.update(f"E|{e.access.value}|{e.space}|{e.target}|{e.nbytes}|".encode())
        for a in self.allocs:
            h.update(f"A|{a.name}|{a.space}|{a.nbytes}|{a.size_origin.value}|".encode())
        for s in self.sync_order():
            h.update(f"S|{s.node_id}|{s.kind.value}|{s.event_id}|{s.core}|".encode())
        for g in self.coverage.gaps:
            h.update(f"G|{g.kind.value}|{g.op}|{g.detail}|".encode())
        return h.hexdigest()


# --------------------------------------------------------------------------
# 旁挂派生结构（不变量 1：不得原地改写 VIR）
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Derived:
    """派生分析结果的旁挂容器。

    不变量 1 禁止引擎改写 VIR，故一切分析产物（生存期区间、wait-for 图、
    值 trace）都以 `node_id -> 结果` 的旁挂映射表达。这使多策略并行探索
    天然隔离——每个策略持有自己的 `Derived`，共享同一份只读 VIR。
    """

    #: 产出者标识（引擎名 + 版本）
    producer: str
    payload: Mapping[str, object] = field(default_factory=dict)
    #: 所依据 VIR 的指纹，防止结果与 VIR 错配
    vir_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.producer:
            raise VIRError("Derived.producer 不得为空——派生结果须可归因")


def check_invariants(module: VModule) -> Sequence[str]:
    """显式复核四条不变量，返回违规描述列表（空表示全部满足）。

    构造期已强制不变量 3/4 的大部分，本函数用于**测试与 CI 的独立复核**：
    契约的自检不能只依赖构造函数，否则契约演进时容易悄悄放宽。
    """
    problems: list[str] = []

    # 不变量 1：不可变性 —— 抽查关键类型均为 frozen
    for cls in (VModule, VRegion, VNode, VAlloc, VSync, VLoop, Effect, Coverage):
        params = getattr(cls, "__dataclass_params__", None)
        if params is None or not params.frozen:
            problems.append(f"不变量 1 违规：{cls.__name__} 非 frozen dataclass")

    # 不变量 2：顺序确定性 —— 两次遍历必须一致，且容器为 tuple
    if module.node_order() != module.node_order():
        problems.append("不变量 2 违规：node_order() 两次调用结果不一致")
    for r in module.walk_regions():
        if not isinstance(r.items, tuple):
            problems.append(f"不变量 2 违规：VRegion({r.id}) 的 items 非 tuple")

    # 不变量 3：可回溯源位置
    for n in module.walk_nodes():
        if not n.loc:
            problems.append(f"不变量 3 违规：VNode({n.id}) 缺源位置")

    # 不变量 4：未识别必入 coverage
    modeled = set(module.coverage.modeled_ops)
    gapped = set(module.coverage.unmodeled_ops())
    for n in module.walk_nodes():
        if n.op not in modeled and n.op not in gapped:
            problems.append(
                f"不变量 4 违规：VNode({n.id}) 的 op={n.op!r} 既未登记为已建模，"
                "也未落入 coverage 缺口——这正是被禁止的静默丢弃"
            )
    return problems
