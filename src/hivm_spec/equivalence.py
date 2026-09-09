"""差分对拍（T3.5，D8 等价内核的判定侧）。

## 锚点缺失绝不自比

没有锚点时 verdict 必须是 `COVERAGE_GAP`/`PENDING`，**绝不能**拿同一份 IR 自比
后报 OK——那是最典型的自欺形态（M3 卡 §4 要点 2）。自比恒等于"通过"，却什么都
没验证；而报告上的 OK 与真验证过的 OK 长得一模一样，读者无从分辨。
故本模块的 `compare` **要求**两个 `ExecResult`，没有"单参数模式"可退化。

## 判等走容差，不走哈希

值哈希是精确的，浮点相等不是。哈希只用于**快速筛选**与**定位第一个不同**；
最终判等一律走 `within_tolerance`（M3 卡 §4 要点 4）。所以流程是：

1. 逐步比哈希 → 找出候选发散点（快）；
2. 对候选点做容差比较 → 确认是否真的超差（准）。

只做第 1 步会把"容差内的正常浮点抖动"报成 MISMATCH（假阳性）；
只做第 2 步则无法定位——容差比较能说"不等"，但说不清"从哪一步开始不等"。

## 首发散点

同一个发散会沿数据流传播成一大片。只报**首现**（拓扑序第一个超容差的节点）
+ 影响计数，否则摘要会被淹没，而真正要看的那一行埋在几十条后面（FR5）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hivm_spec.interpret import ExecResult
from hivm_spec.numeric import Tolerance, within_tolerance
from hivm_spec.values import ConcreteValue, Value
from hivm_spec.verdict import Verdict

__all__ = [
    "EQUIVALENCE_ENGINE_VERSION",
    "DiffResult",
    "Divergence",
    "compare",
    "tolerance_from_config",
]


@dataclass(frozen=True, slots=True)
class Divergence:
    """一处发散。"""

    #: 展开序号（对拍两侧同序号对应同一步）
    seq: int
    op: str
    #: 报告用标签（含迭代路径，如 vadd@i2）
    label: str
    #: 源位置（待验侧）
    file: str = ""
    line: int = 0
    #: 两侧值摘要（人可读，非全量转储）
    left_summary: str = ""
    right_summary: str = ""
    #: 超差的最大绝对/相对偏差
    max_abs: float = 0.0
    max_rel: float = 0.0

    def describe(self) -> str:
        where = f"{self.file}:{self.line}" if self.file else "?"
        return (
            f"{self.label} @ {where}：待验={self.left_summary} 锚点={self.right_summary}"
            f"（最大绝对偏差 {self.max_abs:.3g}，相对 {self.max_rel:.3g}）"
        )


@dataclass(frozen=True, slots=True)
class DiffResult:
    """对拍产物。"""

    verdict: Verdict
    #: 首个发散（拓扑序）；None = 未发现发散
    first: Divergence | None = None
    #: 受影响的后续步数（首发散点之后仍超差的步数）
    impacted: int = 0
    #: 实际生效的容差（结论必须回显，M3 卡 §4 要点 3）
    tolerance: Tolerance = field(default_factory=Tolerance)
    #: 诊断信息（截断、覆盖缺口、步数不齐等）
    notes: tuple[str, ...] = ()
    #: 参与比较的步数
    compared: int = 0

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.OK


def tolerance_from_config(config: dict[str, Any]) -> Tolerance:
    """从配置文档取容差。

    容差只能来自**描述**（进 spec_hash）——不接受 CLI 覆盖，那等于允许用命令行
    参数消灭一个真实的 MISMATCH（M3 卡 §4 要点 3）。
    """
    for check in config.get("checks", ()):
        if check.get("name") == "equivalence":
            opts = check.get("options", {}) or {}
            return Tolerance(
                rtol=float(opts.get("rtol", Tolerance().rtol)),
                atol=float(opts.get("atol", Tolerance().atol)),
            )
    return Tolerance()


def _summary(v: Value | None) -> str:
    """值摘要：够定位，不刷屏。"""
    if v is None:
        return "(缺失)"
    if isinstance(v, ConcreteValue):
        import numpy as np

        arr = np.asarray(v.array).ravel()
        if arr.size == 0:
            return f"{v.dtype}[] (空)"
        head = ", ".join(f"{float(x):.6g}" for x in arr[:4])
        more = "…" if arr.size > 4 else ""
        return f"{v.dtype}{list(v.shape)}[{head}{more}]"
    return f"symbolic:{getattr(v, 'text', lambda: '?')()}"


def _deviation(a: Value, b: Value) -> tuple[float, float]:
    """最大绝对偏差与最大相对偏差。"""
    import numpy as np

    if not isinstance(a, ConcreteValue) or not isinstance(b, ConcreteValue):
        return (0.0, 0.0)
    x = np.asarray(a.array, dtype=np.float64)
    y = np.asarray(b.array, dtype=np.float64)
    if x.shape != y.shape:
        return (float("inf"), float("inf"))
    diff = np.abs(x - y)
    if diff.size == 0:
        return (0.0, 0.0)
    denom = np.maximum(np.abs(y), np.finfo(np.float64).tiny)
    return (float(np.max(diff)), float(np.max(diff / denom)))


def compare(
    left: ExecResult,
    right: ExecResult,
    tolerance: Tolerance,
    *,
    left_label: str = "待验",
    right_label: str = "锚点",
) -> DiffResult:
    """差分对拍：`left`（待验）对 `right`（锚点）。

    **没有单参数模式**：自比恒等于"通过"却什么都没验证，而报告上的 OK 与真验证
    过的 OK 无法分辨（M3 卡 §4 要点 2）。锚点缺失应在调用方就降级。
    """
    notes: list[str] = []

    # 覆盖缺口先行：有未建模 op 时，"没发现发散"可能只是因为没算那一步。
    # 这种情况下报 OK 是把"没检查"说成"检查通过"（FR4/FR6）。
    if left.has_unmodeled or right.has_unmodeled:
        gaps = {g.op for g in (*left.gaps, *right.gaps) if g.op}
        notes.append(
            f"存在未建模 op（{', '.join(sorted(gaps)) or '未具名'}），"
            "其结果未参与对拍——不得据此判定等价"
        )
        return DiffResult(
            verdict=Verdict.COVERAGE_GAP,
            tolerance=tolerance,
            notes=tuple(notes),
            compared=0,
        )

    if left.truncated or right.truncated:
        notes.extend(left.truncation_notes)
        notes.extend(right.truncation_notes)
        notes.append("展开被截断：结论仅在展开界内成立，不可读作『已完整验证』")

    # 步数不齐：两侧结构不同是等价验证的常态（这正是要验的），但**逐步对齐**
    # 就失去依据了。此时只能报覆盖缺口，不能挑"能对上的部分"比——那等于自选
    # 有利证据（FR6）。
    if len(left.traces) != len(right.traces):
        notes.append(
            f"两侧展开步数不同（{left_label} {len(left.traces)} 步、"
            f"{right_label} {len(right.traces)} 步）：无法逐步对齐，"
            "需按输出值而非中间步对拍"
        )
        return DiffResult(
            verdict=Verdict.COVERAGE_GAP,
            tolerance=tolerance,
            notes=tuple(notes),
            compared=0,
        )

    first: Divergence | None = None
    impacted = 0
    compared = 0

    for lt, rt in zip(left.traces, right.traces, strict=True):
        lv = left.values_by_seq.get(lt.seq)
        rv = right.values_by_seq.get(rt.seq)
        if lv is None or rv is None:
            notes.append(f"seq={lt.seq} {lt.label} 缺少可比值，已跳过该步的数值比较")
            continue
        compared += 1

        # 第 1 步：哈希筛选（快）。哈希相同即值内容逐字节相同，无需再比容差。
        if lt.out_hash and lt.out_hash == rt.out_hash:
            continue

        # 第 2 步：容差裁决（准）。哈希不同不代表超差——浮点在容差内的抖动
        # 会改变字节但属正常。
        if within_tolerance(getattr(lv, "array", lv), getattr(rv, "array", rv), tolerance):
            continue

        max_abs, max_rel = _deviation(lv, rv)
        if first is None:
            first = Divergence(
                seq=lt.seq,
                op=lt.op,
                label=lt.label,
                file=lt.loc.file,
                line=lt.loc.line,
                left_summary=_summary(lv),
                right_summary=_summary(rv),
                max_abs=max_abs,
                max_rel=max_rel,
            )
        else:
            impacted += 1

    if first is not None:
        return DiffResult(
            verdict=Verdict.MISMATCH,
            first=first,
            impacted=impacted,
            tolerance=tolerance,
            notes=tuple(notes),
            compared=compared,
        )
    if compared == 0:
        notes.append("没有任何一步产生可比值——对拍未实际发生")
        return DiffResult(
            verdict=Verdict.COVERAGE_GAP,
            tolerance=tolerance,
            notes=tuple(notes),
            compared=0,
        )
    return DiffResult(
        verdict=Verdict.OK,
        tolerance=tolerance,
        notes=tuple(notes),
        compared=compared,
    )


#: 等价引擎版本——结论里回显，便于把历史结论对应到当时的判定口径。
EQUIVALENCE_ENGINE_VERSION = "0.1.0"
