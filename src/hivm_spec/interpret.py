"""控制流解释执行（T3.2，D8 等价内核的执行侧）。

## 遍历核唯一（D6 硬约束）

本模块**不自己走 VIR**，而是复用 `timeline.expand_steps`——那个函数已经把
"程序序 + 循环迭代 + 截断保护"实现了一遍，M2 的同步判定就建立在它之上。
再写一个解释器专用的遍历，等于让同一份 IR 有两种执行顺序：一旦二者漂移，
时间线结论与等价结论就会互相矛盾，而没有任何测试能指出谁错了。

所以这里的分工是：

- `expand_steps`：IR → 线性步骤序列（**唯一**的顺序权威，含 `loop_path`）；
- 本模块：步骤序列 → 值。

代价是解释执行也继承展开界与截断语义。这是**要的**：截断了就不能声称
"已完整验证"，而截断口径必须和 M2 一致，否则同一份 IR 会有两套"验证到哪"的说法。

## 未建模 op 的处理

遇到没有值语义的 op **不跳过**——跳过等于"少算一步却照常给结论"（M3 卡 §4
要点 7）。做法是把结果标记为 `unmodeled`，其下游值全部失去可比性，并登记覆盖
缺口让 verdict 走 `COVERAGE_GAP`。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from hivm_spec.timeline import expand_steps
from hivm_spec.values import (
    Env,
    Value,
    ValueError_,
    ValueKernel,
    parse_value_kernel,
    slot_of,
    value_hash,
)
from hivm_spec.vir import Gap, GapKind, Loc, ValueSlot, VModule, VTrace

__all__ = [
    "ExecResult",
    "OpTrace",
    "interpret",
]


@dataclass(frozen=True, slots=True)
class OpTrace:
    """一个执行步的值哈希记录（T3.6 首发散点定位的原料）。

    `loop_path` 必须在场：循环体内同一个 op 会执行多次，只报 op 名的话
    "第几次迭代开始发散"就丢了——而那恰恰是定位数值问题最关键的信息。
    """

    seq: int
    op: str
    loc: Loc
    #: 迭代下标路径（空 = 循环外）
    loop_path: tuple[int, ...]
    #: 输出值哈希；未建模或无值语义时为 ""
    out_hash: str = ""
    #: 该步是否因未建模而无法求值
    unmodeled: bool = False

    @property
    def label(self) -> str:
        """报告用标签：op@迭代路径。"""
        if not self.loop_path:
            return self.op
        return f"{self.op}@i{'.'.join(str(i) for i in self.loop_path)}"


@dataclass(frozen=True, slots=True)
class ExecResult:
    """解释执行产物。"""

    #: 逐步值哈希轨迹（程序序）
    traces: tuple[OpTrace, ...]
    #: 最终环境（SSA 名 → 值）
    env: Env
    #: 新增覆盖缺口（未建模 op 等）
    gaps: tuple[Gap, ...]
    #: 展开是否被截断（继承 expand_steps 的口径）
    truncated: bool = False
    truncation_notes: tuple[str, ...] = ()
    #: 每一步的输出值（展开序号 → 值）。
    #
    # 差分对拍要按**步**比较中间值，而 env 只留每个 SSA/缓冲区的**最终**内容
    # ——循环里同一缓冲区被写 N 次，env 里只剩最后一次。若只靠 env，首发散点
    # 就只能定位到"某个缓冲区最终不对"，而不是"第 2 次迭代开始不对"（T3.6）。
    values_by_seq: dict[int, Value] = field(default_factory=dict)

    @property
    def has_unmodeled(self) -> bool:
        return any(t.unmodeled for t in self.traces)


def _kernels_of(config: dict[str, Any]) -> tuple[dict[str, ValueKernel], dict[str, str]]:
    """从配置文档取每个 op 的值语义内核。

    配置文档里 `value` 的形态（见 generate.py）：
      {"kind": "declarative", "expr": "elementwise(add, a, b, into=out)"}
      {"kind": "host_fn", "name": ..., "reason": ...}   ← 逃生舱（OD8）

    返回 `(可执行内核, 不可执行原因)`。**不可执行的必须留下原因**——否则
    "这个 op 为什么没参与对拍"在报告里就没有答案（FR5 可诊断性）。
    """
    kernels: dict[str, ValueKernel] = {}
    reasons: dict[str, str] = {}
    for entry in config.get("ops", ()):
        name = entry.get("op", "")
        if not name:
            continue
        val = entry.get("value") or {}
        kind = val.get("kind", "") if isinstance(val, dict) else ""
        if kind == "host_fn":
            why = val.get("reason", "") if isinstance(val, dict) else ""
            reasons[name] = f"值语义由 host_fn 逃生舱提供（{why}）——配置文档中无可执行结构"
            continue
        expr = val.get("expr", "") if isinstance(val, dict) else str(val)
        if not expr:
            reasons[name] = "描述未声明值语义"
            continue
        try:
            kernels[name] = parse_value_kernel(str(expr))
        except ValueError_ as exc:
            reasons[name] = f"值语义无法解析：{exc}"
    return kernels, reasons


def _param_names(config: dict[str, Any], op: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """取 op 的 (输入参数名, 输出参数名)——用于把 SSA 操作数绑到语义参数名。"""
    ins: list[str] = []
    outs: list[str] = []
    for entry in config.get("ops", ()):
        if entry.get("op") != op:
            continue
        for p in entry.get("params", ()):
            kind = p.get("kind", "")
            nm = p.get("name", "")
            if not nm:
                continue
            if kind in ("in", "attr"):
                ins.append(nm)
            elif kind == "out":
                outs.append(nm)
    return tuple(ins), tuple(outs)


def interpret(
    module: VModule,
    config: dict[str, Any],
    inputs: Mapping[str, Value],
    *,
    bound: int | None = None,
    trace: VTrace | None = None,
) -> ExecResult:
    """按程序序解释执行，产出逐 op 值哈希轨迹。

    `inputs` 是**入口实参**（函数参数 SSA 名 → 值）。缺失的输入不臆造默认值：
    那会让"输入没给全"伪装成"算出了结果"（FR6）。

    循环携带值（`VLoop.iter_args`）沿用 `expand_steps` 的展开语义——每次迭代是
    一个独立步骤序列，iter_arg 的更新通过 SSA 重绑定体现。故此处对同一 SSA 名
    在不同迭代的写入采用**迭代限定名**，避免触发 `Env` 的单赋值保护。
    """
    from hivm_spec.timeline import DEFAULT_BOUND

    exp = expand_steps(module, bound if bound is not None else DEFAULT_BOUND)
    kernels, reasons = _kernels_of(config)

    env = Env()
    for name, val in inputs.items():
        env.bind(name, val)

    traces: list[OpTrace] = []
    gaps: list[Gap] = []
    reported: set[str] = set()
    #: 逐步输出值——差分对拍按步比较中间值时需要（env 只留最终内容）
    by_seq: dict[int, Value] = {}

    def _key(ssa: str, path: tuple[int, ...]) -> str:
        """迭代限定名：循环体内同一 SSA 在每次迭代是不同的值实例。

        不做限定就会撞上 Env 的单赋值保护——而放松那个保护是错的方向：
        它正是保证值哈希 trace 不错位的东西（T3.6 依赖）。
        """
        return ssa if not path else f"{ssa}#i{'.'.join(str(i) for i in path)}"

    #: memref 缓冲区的**最新内容**：SSA 名 → 值。
    #
    # 为什么不能只用迭代限定名：HIVM 的 op 通过 `outs(%buf)` 写的是**内存缓冲区**
    # （memref），不是 SSA 结果值。缓冲区的最新内容跨迭代、跨作用域可见——
    # 循环里 vadd 写 %ub_b，循环**之后**的 store 必须能读到它。
    # 若只按迭代限定名查找，store 就会报"操作数未绑定"，把一个语义正确的程序
    # 判成覆盖缺口（假缺口，NFR2）。
    #
    # 迭代限定名仍然保留：它服务于 Env 的单赋值保护与 trace 不错位；
    # 此表则表达"内存最后被写成什么"。二者职责不同，不可合并。
    buffers: dict[str, Value] = dict(inputs)

    def _lookup(ssa: str, path: tuple[int, ...]) -> Value | None:
        """先找本迭代的 SSA 值，再逐层回退，最后回退到缓冲区最新内容。"""
        for cut in range(len(path), -1, -1):
            try:
                return env.get(_key(ssa, path[:cut]))
            except ValueError_:
                continue
        return buffers.get(ssa)

    # VTrace 是 VIR 契约里为"逐 op 值哈希与首发散点定位"预留的**唯一落点**
    # （M3 卡 §1）。解释器通过它对外发布事件，外部工具挂自己的消费者即可，
    # 不必另建一套遍历——那正是 D6 要消灭的双源真理。
    open_regions: tuple[Any, ...] = ()

    def _sync_regions(target: tuple[Any, ...]) -> None:
        """把已进入的区域链对齐到 target，按需触发 exit/enter 回调。"""
        nonlocal open_regions
        if trace is None:
            open_regions = target
            return
        common = 0
        # strict=False 是刻意的：两条区域链长度本就不同（这正是要找的公共前缀）
        for a, b in zip(open_regions, target, strict=False):
            if a is not b:
                break
            common += 1
        for region in reversed(open_regions[common:]):
            trace.on_region_exit(region)
        for region in target[common:]:
            trace.on_region_enter(region)
        open_regions = target

    for step in exp.steps:
        _sync_regions(step.region_path)
        node = step.node
        kernel = kernels.get(node.op)
        if kernel is None:
            why = reasons.get(node.op, "描述未建模该 op")
            if node.op not in reported:
                reported.add(node.op)
                gaps.append(
                    Gap(
                        kind=GapKind.UNMODELED_OP,
                        detail=(
                            f"{node.op} 无可执行值语义（{why}）——其结果不参与数值"
                            f"对拍，下游值失去可比性"
                        ),
                        loc=node.loc,
                        op=node.op,
                    )
                )
            traces.append(
                OpTrace(
                    seq=step.seq,
                    op=node.op,
                    loc=node.loc,
                    loop_path=step.loop_path,
                    unmodeled=True,
                )
            )
            if trace is not None:
                # 未建模的步也要发事件：静默跳过会让 trace 消费者误以为那一步
                # 不存在，从而把"没算"读成"算过且一致"。mode="unbound" 正是
                # ValueSlot 用来表达"没有值"的取值。
                trace.on_node(node, {"": ValueSlot()})
            continue

        ins, _outs = _param_names(config, node.op)
        # 按声明顺序把 SSA 操作数绑到语义参数名。操作数不足即为覆盖缺口，
        # 不补零、不截断——臆造实参会让结论建立在虚构输入上。
        args: dict[str, Value] = {}
        missing = ""
        for idx, pname in enumerate(ins):
            if idx >= len(node.operands):
                missing = f"操作数不足：需要 {pname}（第 {idx + 1} 个）"
                break
            got = _lookup(node.operands[idx], step.loop_path)
            if got is None:
                missing = f"操作数 {node.operands[idx]} 未绑定值"
                break
            args[pname] = got
        if missing:
            if node.op not in reported:
                reported.add(node.op)
                gaps.append(
                    Gap(
                        kind=GapKind.UNMODELED_OP,
                        detail=f"{node.op} 无法求值（{missing}）",
                        loc=node.loc,
                        op=node.op,
                    )
                )
            traces.append(
                OpTrace(
                    seq=step.seq,
                    op=node.op,
                    loc=node.loc,
                    loop_path=step.loop_path,
                    unmodeled=True,
                )
            )
            continue

        out = kernel.apply(args)
        # 写入目标：有 SSA 结果就用它；HIVM 的 memref 语义 op 没有结果，
        # 写的是 `outs(...)` 那个缓冲区——即最后一个操作数。
        targets = list(node.results) or ([node.operands[-1]] if node.operands else [])
        for t in targets:
            if node.results:
                # 真 SSA 结果：受单赋值保护。同名重复出现是展开错位或 IR 非 SSA，
                # 必须响——放过它会让值哈希 trace 错位，T3.6 的首发散点指错 op。
                env.bind(_key(t, step.loop_path), out)
            else:
                # memref 缓冲区写入：**覆写是正常语义**，不是 SSA 违例。
                # store 写回入口的 %gm、循环里反复写同一 %ub_b，都属此类。
                # 早先版本在这里套用了单赋值检查，把合法程序判成"IR 非 SSA"——
                # 混淆了"SSA 值"与"内存缓冲区"两种载体。
                env.values[_key(t, step.loop_path)] = out
            # 同步缓冲区最新内容：后续迭代与循环之后的读取都依赖它
            buffers[t] = out
        traces.append(
            OpTrace(
                seq=step.seq,
                op=node.op,
                loc=node.loc,
                loop_path=step.loop_path,
                out_hash=value_hash(out),
            )
        )
        by_seq[step.seq] = out
        if trace is not None:
            trace.on_node(node, {t_: slot_of(out) for t_ in targets})

    _sync_regions(())

    return ExecResult(
        traces=tuple(traces),
        env=env,
        gaps=tuple(gaps),
        values_by_seq=by_seq,
        truncated=exp.truncated,
        truncation_notes=exp.truncation_notes,
    )
