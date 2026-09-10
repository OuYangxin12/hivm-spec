"""T4.1 z3 后端接入测试。

验收（M4 卡 §3 T4.1）：**op 函数零改动**切换符号模式。
"""

from __future__ import annotations

import pytest

from hivm_spec.values import (
    SYMBOLIC_OPS,
    add,
    div,
    exp,
    mul,
    neg,
    select,
    sub,
    symbol,
)
from hivm_spec.z3_backend import UnsupportedSymbolicOp, translate

pytestmark = pytest.mark.requires_z3


def _solve(left, right):
    """返回 'unsat'（等价）或 'sat'（有反例）。"""
    import z3

    lt = translate(left)
    rt = translate(right, symbols=lt.symbols, functions={})
    solver = z3.Solver()
    solver.add(lt.expr != rt.expr)
    return str(solver.check())


# ---------------------------------------------------------------------------
# 核心验收：op 函数零改动
# ---------------------------------------------------------------------------


def test_op_functions_need_no_change_for_symbolic_mode() -> None:
    """**T4.1 验收**：同一个 op 函数（`values.add`）既产出具体值也产出符号值，
    后者可直接喂给 z3——中间没有任何"符号专用"的 op 实现。
    """
    x, y = symbol("x"), symbol("y")
    tree = add(x, y)  # 与具体模式调用的是同一个函数
    assert tree.mode == "symbolic"

    result = translate(tree)
    assert "x" in result.symbols and "y" in result.symbols
    assert str(result.expr) == "x + y"


def test_commutativity_is_provable() -> None:
    x, y = symbol("x"), symbol("y")
    assert _solve(add(x, y), add(y, x)) == "unsat"


def test_genuine_inequivalence_yields_a_counterexample() -> None:
    """不等价时必须给出反例——否则 UNSAT 与"没算"无法区分。"""
    import z3

    x, y = symbol("x"), symbol("y")
    lt = translate(add(x, y))
    rt = translate(mul(x, y), symbols=lt.symbols)
    solver = z3.Solver()
    solver.add(lt.expr != rt.expr)
    assert solver.check() == z3.sat
    model = solver.model()
    assert len(model) > 0, "SAT 必须伴随可读的反例"


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda x, y: (sub(x, y), sub(y, x)), "sat"),  # 减法不可交换
        (lambda x, y: (mul(x, y), mul(y, x)), "unsat"),  # 乘法可交换
        (lambda x, y: (add(x, neg(y)), sub(x, y)), "unsat"),  # x+(-y) == x-y
    ],
)
def test_algebraic_identities(build, expected) -> None:
    x, y = symbol("x"), symbol("y")
    left, right = build(x, y)
    assert _solve(left, right) == expected


def test_select_translates_to_ite() -> None:
    import z3

    from hivm_spec.values import gt

    c, a, b = symbol("c"), symbol("a"), symbol("b")
    # select 的条件在本子集里是一个比较表达式
    tree = select(gt(c, a), a, b)
    result = translate(tree)
    assert z3.is_app(result.expr)
    assert "If" in str(result.expr)


# ---------------------------------------------------------------------------
# exp：未解释函数（M4 卡 §4.2）
# ---------------------------------------------------------------------------


def test_exp_uses_an_uninterpreted_function() -> None:
    """SMT 无超越函数，exp 建模为未解释函数。"""
    x = symbol("x")
    result = translate(exp(x))
    assert result.uses_uninterpreted
    assert "exp" in result.uninterpreted


def test_uninterpreted_function_gives_congruence_only() -> None:
    """未解释函数只保证"同实参必同值"，不保证任何 exp 的真实性质。"""
    import z3

    x, y = symbol("x"), symbol("y")
    fns: dict = {}
    lt = translate(exp(x), functions=fns)
    rt = translate(exp(x), symbols=lt.symbols, functions=fns)
    solver = z3.Solver()
    solver.add(lt.expr != rt.expr)
    assert solver.check() == z3.unsat, "同实参必同值"

    # 但 exp(x) > 0 这类真实性质**不**成立——UF 对它一无所知
    lt2 = translate(exp(y), functions=fns)
    s2 = z3.Solver()
    s2.add(lt2.expr <= 0)
    assert s2.check() == z3.sat, (
        "UF 不蕴含 exp 的任何真实性质——这正是「只能证明两侧做了相同的事」的原因"
    )


def test_uninterpreted_usage_is_disclosed() -> None:
    """用了 UF 必须留痕，供报告区分结论强度（M4 卡 §4.2）。"""
    x = symbol("x")
    assert not translate(add(x, x)).uses_uninterpreted
    assert translate(exp(x)).uses_uninterpreted


# ---------------------------------------------------------------------------
# 披露与边界
# ---------------------------------------------------------------------------


def test_division_is_disclosed_not_constrained_away() -> None:
    """除零在 Real 下未定义。

    **不**自动加 `b != 0` 约束——那会把"这段代码可能除零"悄悄改成"假设它不会"，
    等于篡改被验证的语义。改为披露。
    """
    x, y = symbol("x"), symbol("y")
    result = translate(div(x, y))
    assert result.has_division

    # 只看可执行代码：注释里**解释**为何不加约束是正当的
    src = _backend_source()
    div_branch = src.split('if node.op == "div":')[1].split('if node.op == "select":')[0]
    code = "\n".join(ln.split("#", 1)[0] for ln in div_branch.splitlines())
    assert "!= 0" not in code, "不得自动加除零约束——那会篡改被验证的语义"
    assert "Distinct" not in code


def test_real_semantics_approximation_is_documented() -> None:
    """用 Real 而非 FP 是**有意的近似**，必须在模块里写明其代价。"""
    src = _backend_source()
    assert "Real" in src
    assert "近似" in src or "不等于" in src


def test_associativity_holds_under_real_but_not_f32() -> None:
    """记录 Real 近似的具体代价：浮点加法**不**满足结合律，但 Real 满足。

    这条测试的价值是把近似的边界钉死——将来若有人以为符号档能验浮点精度，
    这里会告诉他不能。
    """
    a, b, c = symbol("a"), symbol("b"), symbol("c")
    assert _solve(add(add(a, b), c), add(a, add(b, c))) == "unsat", (
        "Real 下结合律成立；f32 下不成立——符号档不覆盖浮点精度问题，那是 M3 具体档 + 容差的职责"
    )


def test_subset_and_translator_stay_in_sync() -> None:
    """`SYMBOLIC_OPS` 里的每一项都必须能翻译。

    子集是封闭集合，若有人扩了子集却没同步翻译器，子集就形同虚设——
    `values.py` 的注释明确警告过这一点。
    """
    x, y, z = symbol("x"), symbol("y"), symbol("z")
    from hivm_spec.values import _COMPARE

    builders = {
        "add": lambda: add(x, y),
        "sub": lambda: sub(x, y),
        "mul": lambda: mul(x, y),
        "div": lambda: div(x, y),
        "neg": lambda: neg(x),
        "exp": lambda: exp(x),
        "select": lambda: select(_cmp("gt", x, y), y, z),
    }
    for name in _COMPARE:
        builders[name] = lambda n=name: _cmp(n, x, y)

    missing = sorted(SYMBOLIC_OPS - set(builders))
    assert not missing, f"子集含 {missing} 但测试未覆盖——请同步"

    for name, build in builders.items():
        try:
            translate(build())
        except UnsupportedSymbolicOp as exc:  # pragma: no cover
            pytest.fail(f"子集含 {name} 但翻译器不支持：{exc}")


def _cmp(name: str, a, b):
    from hivm_spec import values

    return getattr(values, name)(a, b)


def _backend_source() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / "src" / "hivm_spec" / "z3_backend.py").read_text(
        encoding="utf-8"
    )
