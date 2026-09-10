"""未初始化读检查（T4.3）。

## 为什么不用 poison 魔数

朴素做法是给未初始化缓冲填一个特殊值（如 `0xDEADBEEF` 或 NaN），读到它就报警。
这个做法在数值 kernel 上**不可靠**：

- 魔数可能是合法计算结果（`0xDEADBEEF` 作 f32 是个正常的负数）；
- NaN 更不安全——真实计算本来就会产生 NaN（0/0、inf-inf），届时无法区分
  "读了未初始化内存"与"算出了 NaN"。

误报会淹没真问题（NFR2），漏报则是自欺（FR6）。故本模块用**独立的 tainted
标记位**：污染状态与数值分离，两者互不干扰（M4 卡 §4.5）。

## 分析方式

这是一个**数据流分析**，不是值执行——不需要具体输入，也不需要 z3：

1. 函数参数（`VFuncArg`）入口即视为**已初始化**（调用方的责任）；
2. 本地 `memref.alloc` 分配出的缓冲初始为 **tainted**；
3. 每步按描述声明的参数 kind 判定读/写：`out`/`inout` 目标写入后转为 clean；
4. 读到仍为 tainted 的缓冲 → 报告。

顺序权威复用 `timeline.expand_steps`（D6：不新建遍历核）。

## 保守方向

歧义一律偏向**不报**（压制假阳性，NFR2）：

- 未建模 op 的输出目标视为已初始化——我们不知道它写没写，但假定它写了，
  这样不会因为"描述没建模"就把合法代码报成缺陷；
- 条件写（`cond_write`）视为已初始化——分支可能写也可能不写，报了会误伤；
- 但**这些让步都要在报告里留痕**，否则"没报问题"会被读成"没有问题"（FR6）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hivm_spec.vir import AllocOrigin, Loc, VModule

__all__ = [
    "UNINIT_ENGINE_VERSION",
    "UninitRead",
    "UninitReport",
    "analyze_uninit_reads",
]

#: 引擎版本——结论里回显，便于把历史结论对应到当时的判定口径。
UNINIT_ENGINE_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class UninitRead:
    """一次未初始化读。"""

    seq: int
    op: str
    buffer: str
    param: str
    loc: Loc
    loop_path: tuple[int, ...] = ()

    @property
    def label(self) -> str:
        if not self.loop_path:
            return self.op
        return f"{self.op}@i{'.'.join(str(i) for i in self.loop_path)}"

    def describe(self, *, with_loc: bool = True) -> str:
        """`with_loc=False` 供 Finding 使用——渲染层已附加位置，避免重复。"""
        where = f" @ {self.loc.describe()}" if with_loc else ""
        return f"{self.label} 读取未初始化缓冲 {self.buffer}（参数 {self.param}）{where}"


@dataclass(frozen=True, slots=True)
class UninitReport:
    reads: tuple[UninitRead, ...] = ()
    #: 参与分析的本地缓冲数（分母：为 0 时"没发现问题"毫无意义）
    tracked_buffers: int = 0
    #: 因未建模而被保守放行的 op 名
    unmodeled_ops: tuple[str, ...] = ()
    truncated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        return not self.reads


def _param_kinds(config: dict[str, Any]) -> dict[str, tuple[tuple[str, str], ...]]:
    """op → ((参数名, kind), …)。kind 取 in/out/inout。"""
    table: dict[str, tuple[tuple[str, str], ...]] = {}
    for entry in config.get("ops", []):
        params = tuple(
            (p.get("name", ""), p.get("kind", "")) for p in entry.get("params", []) or ()
        )
        if params:
            table[entry["op"]] = params
    return table


def _cond_write_targets(config: dict[str, Any]) -> dict[str, frozenset[str]]:
    """op → 条件写的目标参数名集合。

    条件写不能算"已初始化"的证据，但也不该算"读了未初始化"——保守放行。
    """
    out: dict[str, frozenset[str]] = {}
    for entry in config.get("ops", []):
        targets = {
            eff.get("target", "")
            for eff in entry.get("effects", []) or ()
            if eff.get("kind") == "cond_write"
        }
        if targets:
            out[entry["op"]] = frozenset(targets)
    return out


def analyze_uninit_reads(
    module: VModule, config: dict[str, Any], *, bound: int | None = None
) -> UninitReport:
    """未初始化读的数据流分析。

    复用 `timeline.expand_steps` 作唯一顺序权威（D6）。
    """
    from hivm_spec import timeline as tl

    kinds = _param_kinds(config)
    cond_writes = _cond_write_targets(config)
    modeled = set(kinds)

    # 本地 alloc 初始 tainted；函数参数视为已初始化（调用方责任）
    tainted: set[str] = set()
    #: SSA value 文本 → VAlloc.name。原始 value 文本是整行 IR dump，直接报给
    #: 用户不可读；用描述里的缓冲名才定位得了（FR5）。
    friendly: dict[str, str] = {}
    for alloc in module.allocs:
        if alloc.origin is AllocOrigin.LOCAL_ALLOC and alloc.value:
            tainted.add(alloc.value)
            friendly[alloc.value] = alloc.name
    tracked = len(tainted)

    exp = tl.expand_steps(module, bound if bound is not None else tl.DEFAULT_BOUND)

    reads: list[UninitRead] = []
    unmodeled: set[str] = set()
    reported: set[tuple[str, str]] = set()  # (buffer, op) 去重

    for step in exp.steps:
        node = step.node
        params = kinds.get(node.op)
        if params is None:
            # 未建模 op：不知道它读什么写什么。保守假定它**写**了所有操作数
            # ——宁可漏报也不误报（NFR2），但要留痕。
            if node.op not in modeled:
                unmodeled.add(node.op)
            for operand in node.operands:
                tainted.discard(operand)
            continue

        conds = cond_writes.get(node.op, frozenset())

        # 先判读：读发生在写之前（同一 op 内 in 参数先于 out 生效）
        for (pname, kind), operand in zip(params, node.operands, strict=False):
            if kind not in ("in", "inout"):
                continue
            if operand in tainted and (operand, node.op) not in reported:
                reported.add((operand, node.op))
                reads.append(
                    UninitRead(
                        seq=step.seq,
                        op=node.op,
                        buffer=friendly.get(operand, operand),
                        param=pname,
                        loc=node.loc,
                        loop_path=step.loop_path,
                    )
                )

        # 再判写：out/inout 目标转为 clean
        for (pname, kind), operand in zip(params, node.operands, strict=False):
            if kind in ("out", "inout") and pname not in conds:
                tainted.discard(operand)

    notes: list[str] = []
    if tracked == 0:
        notes.append("模块内无本地分配的缓冲——本检查未实际发生，不得据此判定无未初始化读")
    if unmodeled:
        notes.append(
            f"{len(unmodeled)} 个未建模 op（{', '.join(sorted(unmodeled)[:5])}）"
            "的读写未知，已保守视为已写入——可能漏报"
        )
    if exp.truncated:
        notes.append("循环展开被截断，结论只在展开界内成立")
    notes.append("函数参数视为已初始化（调用方责任），本检查只覆盖本地分配的缓冲")

    return UninitReport(
        reads=tuple(reads),
        tracked_buffers=tracked,
        unmodeled_ops=tuple(sorted(unmodeled)),
        truncated=exp.truncated,
        notes=tuple(notes),
    )
