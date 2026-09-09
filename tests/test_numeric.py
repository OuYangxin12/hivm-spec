"""T3.3 数值基础设施测试（OD3）。

`round_mode` 的验收基准是 **C 语言语义**——主仓 `HIVMAttrs.td` 的 description
就是这么写的（"c language rint" 等）。本测试的期望值经 `gcc` 编译的 C 程序
逐值交叉验证过（rint/round/floor/ceil/trunc × 7 个含 .5 的样点全部一致）。
"""

from __future__ import annotations

import numpy as np
import pytest

from hivm_spec.numeric import (
    DTYPES,
    NumericSupportError,
    RoundMode,
    Tolerance,
    apply_round,
    cast_to,
    dtype_of,
    within_tolerance,
)

#: 覆盖两侧符号与 tie（.5）的样点——tie 才能区分各模式
TIES = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=np.float64)


def _has_ml_dtypes() -> bool:
    try:
        import ml_dtypes  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# round_mode：期望值由 C 语言参照实现交叉验证
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        # C rint：tie to even
        (RoundMode.RINT, [-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0]),
        # C round：tie away from zero —— 与 RINT 在 ±2.5 上分道扬镳
        (RoundMode.ROUND, [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]),
        (RoundMode.FLOOR, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
        (RoundMode.CEIL, [-2.0, -1.0, -0.0, 1.0, 2.0, 3.0, 4.0]),
        (RoundMode.TRUNC, [-2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0]),
        # Von Neumann：结果恒取奇数侧
        (RoundMode.ODD, [-3.0, -1.0, -1.0, 1.0, 1.0, 3.0, 3.0]),
    ],
    ids=lambda v: v.value if isinstance(v, RoundMode) else "",
)
def test_round_modes_match_c_semantics(mode: RoundMode, expected: list[float]) -> None:
    """每个模式对齐主仓 description 指名的 C 函数语义。"""
    np.testing.assert_array_equal(apply_round(TIES, mode), np.array(expected))


def test_rint_and_round_differ_on_ties() -> None:
    """RINT 与 ROUND 必须在 tie 上不同——若相同，说明其一实现错了。

    这是最容易出错的地方：`np.round` 是 tie-to-even，直接拿它当 ROUND 会静默
    产生错误的数值语义（主仓 ROUND 是 tie away from zero）。
    """
    rint = apply_round(TIES, RoundMode.RINT)
    rnd = apply_round(TIES, RoundMode.ROUND)
    assert not np.array_equal(rint, rnd)
    # 具体分歧点：±2.5 与 ±0.5
    assert rint[5] == 2.0 and rnd[5] == 3.0  # 2.5
    assert rint[3] == 0.0 and rnd[3] == 1.0  # 0.5


def test_odd_rounding_never_yields_even() -> None:
    """Von Neumann 舍入：有小数被丢弃时结果必为奇数。"""
    xs = np.array([0.25, 1.75, 2.5, 4.5, -0.25, -4.5], dtype=np.float64)
    got = apply_round(xs, RoundMode.ODD)
    assert np.all(np.mod(np.abs(got), 2) == 1), got
    # 本来就是整数的值不动（没有小数被丢弃）
    ints = np.array([2.0, 4.0, -6.0], dtype=np.float64)
    np.testing.assert_array_equal(apply_round(ints, RoundMode.ODD), ints)


def test_unspecified_round_mode_is_refused() -> None:
    """主仓未描述语义的模式必须拒绝执行，不得猜一个行为。

    TRUNCWITHOVERFLOW 在 HIVMAttrs.td 的 description 里没有说明；猜一个溢出
    规则写进去等于凭空发明数值语义（D9）。
    """
    with pytest.raises(NumericSupportError, match="TRUNCWITHOVERFLOW"):
        apply_round(TIES, RoundMode.TRUNCWITHOVERFLOW)


def test_round_mode_enum_matches_upstream() -> None:
    """枚举成员与主仓 HIVM_RoundModeEnum 一一对应（防漏防臆造）。"""
    assert {m.name for m in RoundMode} == {
        "RINT",
        "ROUND",
        "FLOOR",
        "CEIL",
        "TRUNC",
        "ODD",
        "TRUNCWITHOVERFLOW",
    }


# ---------------------------------------------------------------------------
# dtype：bf16 缺依赖必须降级而非静默变 f32
# ---------------------------------------------------------------------------


def test_known_dtypes_resolve() -> None:
    for name, needs_ml in DTYPES.items():
        if needs_ml and not _has_ml_dtypes():
            continue
        assert dtype_of(name) is not None


def test_unknown_dtype_is_loud() -> None:
    with pytest.raises(NumericSupportError, match="未知 dtype"):
        dtype_of("f8_maybe")


@pytest.mark.skipif(not _has_ml_dtypes(), reason="PENDING(env) 缺 ml_dtypes")
def test_bf16_truncates_precision() -> None:
    """bf16 必须真的损失精度——若结果与 f32 相同，说明退化成了 f32。

    这是 T3.3 最关键的一条：f32 算 bf16 会比真实硬件更精确，对拍会**假通过**。
    """
    import ml_dtypes

    got = cast_to(np.array([0.1], dtype=np.float64), "bf16")
    assert got.dtype == np.dtype(ml_dtypes.bfloat16)
    # bf16 只有 8 位尾数：0.1 → 0.100098
    assert float(got[0]) != pytest.approx(0.1, abs=1e-9)
    assert float(got[0]) == pytest.approx(0.100098, abs=1e-6)
    # 与 f32 的结果必须不同（否则就是退化）
    assert float(got[0]) != float(np.array([0.1], dtype=np.float32)[0])


@pytest.mark.skipif(_has_ml_dtypes(), reason="仅在缺 ml_dtypes 时可测")
def test_bf16_without_ml_dtypes_raises_not_degrades() -> None:
    """缺 ml_dtypes 时必须抛错，让调用方降级成 COVERAGE_GAP。"""
    with pytest.raises(NumericSupportError, match="ml_dtypes"):
        dtype_of("bf16")


def test_f16_needs_no_optional_dependency() -> None:
    """f16 走 numpy 原生，不受可选依赖影响。"""
    assert DTYPES["f16"] is False
    got = cast_to(np.array([0.1], dtype=np.float64), "f16")
    assert got.dtype == np.dtype("float16")
    # f16 有 10 位尾数，精度损失小于 bf16 但仍存在
    assert float(got[0]) != 0.1


# ---------------------------------------------------------------------------
# 容差
# ---------------------------------------------------------------------------


def test_tolerance_defaults_and_dict() -> None:
    t = Tolerance()
    assert t.rtol > 0 and t.atol > 0
    assert t.as_dict() == {"rtol": t.rtol, "atol": t.atol}


def test_negative_tolerance_is_refused() -> None:
    with pytest.raises(NumericSupportError, match="不得为负"):
        Tolerance(rtol=-1e-3)


def test_within_tolerance_uses_both_rtol_and_atol() -> None:
    """rtol 管相对误差、atol 管接近零的绝对误差——二者缺一不可（OD3）。"""
    a = np.array([1.0, 1e-10], dtype=np.float64)
    b = np.array([1.0 + 1e-7, 0.0], dtype=np.float64)
    # 宽松：相对 1e-7 落在 rtol 内，1e-10 落在 atol 内
    assert within_tolerance(a, b, Tolerance(rtol=1e-5, atol=1e-8))
    # 收紧 rtol 后第一个元素超差
    assert not within_tolerance(a, b, Tolerance(rtol=1e-9, atol=1e-8))
    # 收紧 atol 后第二个元素超差
    assert not within_tolerance(a, b, Tolerance(rtol=1e-5, atol=1e-12))


def test_nan_is_never_within_tolerance() -> None:
    """NaN 视为不相等——NaN 出现在结果里通常本身就是缺陷，不能当通过。"""
    a = np.array([np.nan], dtype=np.float64)
    assert not within_tolerance(a, a, Tolerance())
    assert not within_tolerance(a, np.array([1.0]), Tolerance())


# ---------------------------------------------------------------------------
# 容差必须进 spec_hash（M3 卡 §4 要点 3）
# ---------------------------------------------------------------------------


def test_tolerance_change_changes_spec_hash() -> None:
    """改容差必须改 spec_hash——否则"放宽容差"就成了无痕操作。

    锁定的是设计约束而非实现细节：容差只能在描述里声明、经 spec_hash 固化，
    这样每个历史结论都能对应到它当时用的容差。若有人把容差挪到 CLI 参数上，
    此测试不会失败，但 test_tolerance_is_declared_in_description 会。
    """
    from hivm_spec.spec import Spec

    def _mk(rtol: float) -> str:
        s = Spec(name="t", arch="a3")
        s.op("hivm.hir.vadd", value="elementwise(add, a, b, into=out)")
        s.check("equivalence", rtol=rtol, atol=1e-8)
        return s.spec_hash()

    assert _mk(1e-5) == _mk(1e-5), "同输入必须同哈希（FR8）"
    assert _mk(1e-5) != _mk(1e-3), "容差变化必须产生新 spec_hash"


def test_tolerance_is_declared_in_description() -> None:
    """toy 描述必须声明等价容差，且 rtol/atol 齐备（OD3 双参数）。

    若将来容差被挪到命令行，这里就会失败——那正是要拦的方向：CLI 上调容差
    绕过了描述评审与 spec_hash 记录。
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    loader = importlib.util.spec_from_file_location("toy_tol", root / "specs" / "toy.py")
    assert loader and loader.loader
    mod = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod)

    eq = next(c for c in mod.spec.checks if c.name == "equivalence")
    assert "rtol" in eq.options and "atol" in eq.options, eq.options
    assert eq.options["rtol"] > 0 and eq.options["atol"] > 0
    # round_mode 须为主仓枚举里的名字（小写形式）
    assert eq.options["round_mode"] in {m.value for m in RoundMode}
