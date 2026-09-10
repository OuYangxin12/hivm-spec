"""有界符号等价（T4.2）。

## 与具体档（M3）的关系：并列，不替代

| | 具体档（M3） | 符号档（本模块） |
|---|---|---|
| 结论 | "**这组输入**上没发现发散" | "**有界内所有输入**上无反例" |
| 判等 | rtol/atol 容差 | SMT 求解（Real 语义） |
| 浮点精度 | **覆盖**（这是它的强项） | **不覆盖**（Real 近似） |
| 逃生舱 op | 整链失去可比性 | 可作未解释函数继续推 |

二者结论强度不同，**必须分别标注**（M4 卡 §4.1）。符号档报 UNSAT 不等于
具体档报 OK，反之亦然。

## 三种结论，各自的含义

- `EQUIVALENT`（UNSAT）：在**声明的界内**、**Real 语义下**找不到反例；
- `COUNTEREXAMPLE`（SAT）：找到了具体的反例赋值，直接可读；
- `UNKNOWN`：求解器超时或放弃。

**`UNKNOWN` 按缺口处理，绝不按通过处理**（M4 卡 §4.4）。超时的含义是"没证
出来"，不是"证明了没问题"。这是本模块最容易被写错、也最致命的一处——一个
把超时当通过的验证器，比没有验证器更危险。

## 界是结论的一部分

"有界内无反例"里的"界"（循环展开次数）缺了它，读者会误读为无条件成立。
故 `SymbolicDiff.bound` 恒随结论输出，与 M2 的截断语义同理（M4 卡 §4.7）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hivm_spec.symbolic import SymbolicSupportError, require_z3
from hivm_spec.values import SymbolicValue
from hivm_spec.z3_backend import translate

__all__ = [
    "SYMBOLIC_ENGINE_VERSION",
    "SymbolicDiff",
    "SymbolicOutcome",
    "compare_symbolic",
]

SYMBOLIC_ENGINE_VERSION = "0.1.0"

#: 单次求解的墙钟上限（毫秒）。超时按缺口处理，不按通过处理。
DEFAULT_TIMEOUT_MS = 5000


class SymbolicOutcome(Enum):
    """符号等价的三种结论。"""

    EQUIVALENT = "EQUIVALENT"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    #: 超时/放弃——**是缺口，不是通过**
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class SymbolicDiff:
    """一次符号对拍的结论。"""

    outcome: SymbolicOutcome
    #: 循环展开界——结论的一部分，缺了它"无反例"会被误读为无条件成立
    bound: int
    #: 比较过的输出步数
    compared: int = 0
    #: SAT 时的反例赋值（符号名 → 值的文本）
    counterexample: dict[str, str] = field(default_factory=dict)
    #: 首个不等价的步（seq, op）
    first_divergence: tuple[int, str] | None = None
    #: 用到的未解释函数——非空则结论只能声称"两侧做了相同的事"
    uninterpreted: tuple[str, ...] = ()
    #: 求解中出现除法（Real 下除零未定义）
    has_division: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def proven(self) -> bool:
        """是否**证明**了等价。UNKNOWN 不算——这是本模块的核心纪律。"""
        return self.outcome is SymbolicOutcome.EQUIVALENT

    def describe(self) -> str:
        if self.outcome is SymbolicOutcome.EQUIVALENT:
            base = f"在展开界 {self.bound} 内、Real 语义下未找到反例"
            if self.uninterpreted:
                base += (
                    f"（含未解释函数 {', '.join(self.uninterpreted)}："
                    "只能证明两侧做了相同的事，不能证明那件事是对的）"
                )
            return base
        if self.outcome is SymbolicOutcome.COUNTEREXAMPLE:
            where = ""
            if self.first_divergence:
                seq, op = self.first_divergence
                where = f"，首个发散在 seq{seq} {op}"
            args = ", ".join(f"{k}={v}" for k, v in sorted(self.counterexample.items()))
            return f"找到反例{where}：{args}"
        return "求解未完成（超时或放弃）——**没能证明**等价，不得据此认为无问题"


def compare_symbolic(
    left: dict[int, Any],
    right: dict[int, Any],
    *,
    bound: int,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    op_names: dict[int, str] | None = None,
) -> SymbolicDiff:
    """逐步比较两侧的符号输出。

    `left`/`right` 是 `seq → SymbolicValue`（来自 `interpret` 的 `values_by_seq`）。
    `op_names` 把 seq 映射到 op 名，仅用于让报告可读（FR5）——缺了它结论仍
    正确，只是定位信息少一截。
    """
    names = op_names or {}
    try:
        z3 = require_z3()
    except SymbolicSupportError as exc:
        return SymbolicDiff(
            outcome=SymbolicOutcome.UNKNOWN,
            bound=bound,
            notes=(str(exc),),
        )

    shared_syms: dict[str, Any] = {}
    shared_fns: dict[str, Any] = {}
    notes: list[str] = []
    uninterpreted: set[str] = set()
    has_division = False

    common = sorted(set(left) & set(right))
    if not common:
        return SymbolicDiff(
            outcome=SymbolicOutcome.UNKNOWN,
            bound=bound,
            notes=("两侧没有可比较的输出步——未实际比较，不得视为等价",),
        )

    if set(left) != set(right):
        # 步数不同不挑能对上的部分比——那是自选有利证据（沿用 M3 口径）
        return SymbolicDiff(
            outcome=SymbolicOutcome.UNKNOWN,
            bound=bound,
            compared=0,
            notes=(
                f"两侧步数不同（{len(left)} vs {len(right)}）——"
                "不挑能对上的部分比较，那是自选有利证据",
            ),
        )

    compared = 0
    for seq in common:
        lv, rv = left[seq], right[seq]
        if not isinstance(lv, SymbolicValue) or not isinstance(rv, SymbolicValue):
            notes.append(f"seq{seq} 非符号值，跳过——已计入未比较")
            continue

        lt = translate(lv, symbols=shared_syms, functions=shared_fns)
        rt = translate(rv, symbols=shared_syms, functions=shared_fns)
        uninterpreted |= set(lt.uninterpreted) | set(rt.uninterpreted)
        has_division = has_division or lt.has_division or rt.has_division

        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        solver.add(lt.expr != rt.expr)
        result = solver.check()
        compared += 1

        if result == z3.unsat:
            continue

        if result == z3.sat:
            model = solver.model()
            return SymbolicDiff(
                outcome=SymbolicOutcome.COUNTEREXAMPLE,
                bound=bound,
                compared=compared,
                counterexample={str(d): str(model[d]) for d in model.decls()},
                first_divergence=(seq, names.get(seq, "")),
                uninterpreted=tuple(sorted(uninterpreted)),
                has_division=has_division,
                notes=tuple(notes),
            )

        # z3.unknown：超时或放弃。**绝不**当作等价。
        return SymbolicDiff(
            outcome=SymbolicOutcome.UNKNOWN,
            bound=bound,
            compared=compared,
            uninterpreted=tuple(sorted(uninterpreted)),
            has_division=has_division,
            notes=(
                *notes,
                f"seq{seq} 求解超时（{timeout_ms}ms）或放弃——"
                "这表示「没能证明」，不是「证明了没问题」",
            ),
        )

    if compared == 0:
        return SymbolicDiff(
            outcome=SymbolicOutcome.UNKNOWN,
            bound=bound,
            compared=0,
            notes=(*notes, "没有任何一步被实际比较——不得视为等价"),
        )

    if has_division:
        notes.append("表达式含除法：Real 语义下除零未定义，本结论未排除该情形")
    notes.append(f"结论仅在展开界 {bound} 内成立")

    return SymbolicDiff(
        outcome=SymbolicOutcome.EQUIVALENT,
        bound=bound,
        compared=compared,
        uninterpreted=tuple(sorted(uninterpreted)),
        has_division=has_division,
        notes=tuple(notes),
    )
