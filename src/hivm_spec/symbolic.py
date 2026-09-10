"""符号后端的可用性探测与降级（T4.0）。

## 为什么单独一个模块

z3 是**可选依赖**（`[symbolic]` extra，决策见
`docs/decisions/T4.0-z3-dependency.md`）。核心层不得因为它缺失而崩溃，但
**更不得**因为它缺失就悄悄跳过检查——那会把"没装依赖"伪装成"验证通过"（FR6）。

本模块提供唯一的探测点，杜绝多处 `try: import z3` 各自漂移。

## 降级口径（照搬 `numeric.py` 的既有模式，不自创第四种）

1. 本模块抛 `SymbolicSupportError`（对应 `NumericSupportError`）；
2. 装配层捕获，降级为 `COVERAGE_GAP` 并写明环境原因；
3. CLI 层对"环境不可用"返回 `EXIT_PENDING`（3），与"验证失败"（1）区分。

**注意**：`Verdict` 是封闭枚举，**没有** `PENDING` 成员。PENDING 是退出码层的
概念，不是 verdict 层的——两者不要混淆。
"""

from __future__ import annotations

import importlib.util
from typing import Any

__all__ = [
    "Z3_HINT",
    "SymbolicSupportError",
    "require_z3",
    "z3_available",
]

Z3_HINT = (
    "符号档需要 z3，但它不可用。安装：pip install -e '.[symbolic]'。"
    "决策与理由见 docs/decisions/T4.0-z3-dependency.md。"
)


class SymbolicSupportError(RuntimeError):
    """符号后端不可用。

    刻意**不**继承 ImportError：调用方要能把它与"代码写错了导致的导入失败"
    区分开——前者是环境问题（报 PENDING），后者是缺陷（应当崩溃）。
    """


def z3_available() -> bool:
    """z3 是否可导入。

    用 `find_spec` 而非真的 `import z3`：导入耗时 0.17s，而本函数可能在
    每次工具装配时被调用。探测不该有这个开销。
    """
    return importlib.util.find_spec("z3") is not None


def require_z3() -> Any:
    """返回 z3 模块；不可用时抛 `SymbolicSupportError`。

    绝不返回 None 或一个假的 stub——那会让调用方在无声的降级路径上继续走，
    最终产出一个看起来正常、实际什么都没验证的结论。
    """
    if not z3_available():
        raise SymbolicSupportError(Z3_HINT)
    import z3

    return z3
