"""数值基础设施（T3.3，D4/OD3）。

## dtype

f16 走 numpy 原生；**bf16 需要 `ml_dtypes`**（可选依赖 `numeric` extra）。
`ml_dtypes` 是 JAX/TF 生态的既有实现，自己手写 bf16 舍入是在造一个没人对拍过的
数值语义——那正是 D9 要防的漂移源。

缺 `ml_dtypes` 时**不静默退化为 f32**：f32 算 bf16 会给出比真实硬件更精确的
结果，对拍就会"通过"，而那是假阴性（FR6）。做法是抛 `NumericSupportError`，
由调用方降级成 `COVERAGE_GAP`/`PENDING(env)`——"没装依赖"必须表现为"没验证"，
不能表现为"验证通过"。

## round_mode

枚举与语义**照搬主仓** `HIVMAttrs.td:440-460`（`HIVM_RoundModeEnum`），不自创：

    RINT   round to nearest, tie to even      (C rint)
    ROUND  round to nearest, tie away from 0  (C round)
    FLOOR  round to minus infinity            (C floor)
    CEIL   round to positive infinity         (C ceil)
    TRUNC  round to zero                      (C trunc)
    ODD    round to odd (Von Neumann rounding)
    TRUNCWITHOVERFLOW                          （主仓未给描述，见下）

`TRUNCWITHOVERFLOW` 在主仓的 description 里**没有说明**，故此处不实现它的语义
（`apply_round` 对其抛错），并登记为语义假设的候选——猜一个溢出行为写进去，等于
凭空发明一条数值规则。

## 容差

rtol + atol 双参数（OD3）。全局默认 + per-op 覆盖；两者都进 `spec_hash`，
所以"放宽容差"是一次可审计的描述变更，而不是命令行上悄悄调一个数
（M3 卡 §4 要点 3）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "DTYPES",
    "NumericSupportError",
    "RoundMode",
    "Tolerance",
    "apply_round",
    "cast_to",
    "dtype_of",
    "within_tolerance",
]


class NumericSupportError(Exception):
    """所需数值支持不可用（如缺 `ml_dtypes`）。

    调用方**必须**把它降级成覆盖缺口，不得当作"通过"——见模块 docstring。
    """


class RoundMode(Enum):
    """舍入模式。取值与语义照搬主仓 `HIVM_RoundModeEnum`。"""

    RINT = "rint"
    ROUND = "round"
    FLOOR = "floor"
    CEIL = "ceil"
    TRUNC = "trunc"
    ODD = "odd"
    TRUNCWITHOVERFLOW = "truncwithoverflow"


#: 规范 dtype 名 → 是否需要 ml_dtypes
#:
#: 用规范名（f32/f16/bf16）而非 numpy dtype 名，是为了在**没装 ml_dtypes 时也能
#: 表达 bf16**——描述可以声明 bf16，执行时才因缺依赖而降级。若直接用 numpy
#: dtype 对象做键，描述在导入期就会崩，连"声明了 bf16"这个事实都留不下来。
DTYPES: dict[str, bool] = {
    "f32": False,
    "f16": False,
    "bf16": True,
    "f64": False,
    "i8": False,
    "i16": False,
    "i32": False,
    "i64": False,
    "i1": False,
}


def dtype_of(name: str) -> Any:
    """规范 dtype 名 → numpy dtype。

    bf16 缺依赖时抛 `NumericSupportError`，**不退化为 f32**。
    """
    import numpy as np

    if name not in DTYPES:
        raise NumericSupportError(
            f"未知 dtype {name!r}；已知：{sorted(DTYPES)}。"
            "新增 dtype 需同时确认它在 numpy/ml_dtypes 里的确切表示"
        )
    if name == "bf16":
        try:
            import ml_dtypes
        except ImportError as exc:
            raise NumericSupportError(
                "bf16 需要 ml_dtypes（可选依赖 numeric extra）。"
                "缺依赖时**不退化为 f32**：f32 算 bf16 会比真实硬件更精确，"
                "对拍会假通过。请安装 `pip install -e '.[numeric]'`，"
                "或接受 COVERAGE_GAP 降级"
            ) from exc
        return np.dtype(ml_dtypes.bfloat16)
    return np.dtype(
        {
            "f32": "float32",
            "f16": "float16",
            "f64": "float64",
            "i8": "int8",
            "i16": "int16",
            "i32": "int32",
            "i64": "int64",
            "i1": "bool",
        }[name]
    )


def cast_to(array: Any, dtype_name: str) -> Any:
    """按规范名转换 dtype——低精度的截断由 numpy/ml_dtypes 负责，不自己实现。"""
    return array.astype(dtype_of(dtype_name))


def apply_round(array: Any, mode: RoundMode) -> Any:
    """按主仓语义做舍入。

    每个分支都对应 `HIVMAttrs.td` 的一行描述，不做"差不多"的近似。
    """
    import numpy as np

    if mode is RoundMode.RINT:
        # tie to even —— numpy 的 rint 正是此语义
        return np.rint(array)
    if mode is RoundMode.ROUND:
        # tie away from zero。**不能用 np.round**：np.round 是 tie-to-even，
        # 与主仓的 ROUND 不同。用 floor(|x|+0.5) 带符号还原。
        return np.copysign(np.floor(np.abs(array) + 0.5), array)
    if mode is RoundMode.FLOOR:
        return np.floor(array)
    if mode is RoundMode.CEIL:
        return np.ceil(array)
    if mode is RoundMode.TRUNC:
        return np.trunc(array)
    if mode is RoundMode.ODD:
        # Von Neumann rounding：结果为整数时取奇数侧。
        # 即：截断后若丢失了小数且结果为偶，则朝远离零方向调 1 使其为奇。
        t = np.trunc(array)
        frac_lost = array != t
        is_even = np.mod(t, 2) == 0
        adjust = frac_lost & is_even
        return np.where(adjust, t + np.copysign(1.0, array), t)
    raise NumericSupportError(
        f"round_mode={mode.value} 的语义未实现。"
        "主仓 HIVMAttrs.td 的 description 未给出 TRUNCWITHOVERFLOW 的行为，"
        "猜一个溢出规则等于凭空发明数值语义（D9）——须先与 C++ 链对拍"
    )


@dataclass(frozen=True, slots=True)
class Tolerance:
    """数值容差（OD3：rtol + atol 双参数）。

    两者都进 `spec_hash`（由描述侧负责），所以"放宽容差"是一次**可审计的描述
    变更**，而不是命令行上悄悄调一个数（M3 卡 §4 要点 3）。
    """

    rtol: float = 1e-5
    atol: float = 1e-8

    def __post_init__(self) -> None:
        if self.rtol < 0 or self.atol < 0:
            raise NumericSupportError("容差不得为负")

    def as_dict(self) -> dict[str, float]:
        return {"rtol": self.rtol, "atol": self.atol}


def within_tolerance(a: Any, b: Any, tol: Tolerance) -> bool:
    """容差比较：`|a-b| <= atol + rtol*|b|`（numpy allclose 口径）。

    **NaN 视为不相等**（`equal_nan=False`）。理由：NaN 出现在结果里通常本身就是
    缺陷（未初始化、除零、溢出），把 NaN==NaN 当通过会让这类问题静默溜过。
    """
    import numpy as np

    return bool(np.allclose(a, b, rtol=tol.rtol, atol=tol.atol, equal_nan=False))
