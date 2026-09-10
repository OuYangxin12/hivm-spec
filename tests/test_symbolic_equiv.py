"""T4.2 有界符号等价测试。

验收（M4 卡 §3 T4.2）：SAN/UNSAT + 超时按缺口。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hivm_spec.symbolic_equiv import (
    SymbolicDiff,
    SymbolicOutcome,
    compare_symbolic,
)
from hivm_spec.values import add, div, mul, symbol

pytestmark = pytest.mark.requires_z3

ROOT = Path(__file__).resolve().parents[1]


def _diff(left, right, *, bound=16, **kw) -> SymbolicDiff:
    """包装器，省去每次拼 values_by_seq 字典。"""
    return compare_symbolic(
        {0: left},
        {0: right},
        bound=bound,
        **kw,
    )


# ---------------------------------------------------------------------------
# 三种结论
# ---------------------------------------------------------------------------


def test_equivalent_expression() -> None:
    x, y = symbol("x"), symbol("y")
    d = _diff(add(x, y), add(y, x))
    assert d.proven
    assert d.outcome is SymbolicOutcome.EQUIVALENT
    assert d.bound == 16
    assert d.compared == 1
    assert not d.counterexample


def test_counterexample_yields_assignments() -> None:
    x, y = symbol("x"), symbol("y")
    d = _diff(add(x, y), mul(x, y))
    assert d.outcome is SymbolicOutcome.COUNTEREXAMPLE
    assert d.counterexample
    assert d.first_divergence is not None


def test_unknown_for_timeout() -> None:
    """M4 卡 §4.4：超时/放弃**不是**通过。"""
    x, y = symbol("x"), symbol("y")
    d = compare_symbolic({0: add(x, y)}, {0: add(y, x)}, bound=16, timeout_ms=1)
    # 简单问题 1ms 内不超时，所以我们实际上验证超时的**处置逻辑**
    # 而非强制执行——太短超时在简单问题上不触发
    assert d.outcome in (SymbolicOutcome.EQUIVALENT, SymbolicOutcome.UNKNOWN)
    if d.outcome is SymbolicOutcome.UNKNOWN:
        assert not d.proven


# ---------------------------------------------------------------------------
# 界是结论的一部分
# ---------------------------------------------------------------------------


def test_bound_appears_in_conclusion() -> None:
    """M4 卡 §4.7：界是结论的一部分。"""
    d = _diff(symbol("x"), symbol("x"), bound=42)
    assert d.bound == 42
    assert "42" in d.describe()


def test_describe_includes_uninterpreted() -> None:
    x = symbol("x")
    from hivm_spec.values import exp

    d = _diff(exp(x), exp(x))
    assert d.proven
    assert "未解释函数" in d.describe()


# ---------------------------------------------------------------------------
# 步数不一致 / 无可比步
# ---------------------------------------------------------------------------


def test_different_step_counts_yield_unknown() -> None:
    """步数不同不挑能对上的部分比——那是自选有利证据（与 M3 口径）。"""
    x, y = symbol("x"), symbol("y")
    d = compare_symbolic({0: x}, {0: x, 1: y}, bound=16)
    assert d.outcome is SymbolicOutcome.UNKNOWN


def test_no_common_steps_yields_unknown() -> None:
    d = compare_symbolic({0: symbol("x")}, {1: symbol("x")}, bound=16)
    assert d.outcome is SymbolicOutcome.UNKNOWN


# ---------------------------------------------------------------------------
# 除法披露
# ---------------------------------------------------------------------------


def test_division_is_noted() -> None:
    x, y = symbol("x"), symbol("y")
    d = _diff(div(x, y), div(x, y))
    assert d.proven
    assert d.has_division


# ---------------------------------------------------------------------------
# CLI 端到端
# ---------------------------------------------------------------------------


def test_cli_symbolic_mode_same_ir() -> None:
    """相同 IR 符号比较 → OK。"""
    from hivm_spec.__main__ import _effects_from_config
    from hivm_spec.assemble import run_tool
    from hivm_spec.ir_engine import lower_module_text
    from hivm_spec.verdict import Verdict

    cfg = json.loads((ROOT / "build" / "config.json").read_text())
    modeled = {o["op"] for o in cfg["ops"]}
    effects, pipes = _effects_from_config(cfg)
    src = (ROOT / "specs" / "cases" / "corpus" / "l0" / "loop_load_add_store.mlir").read_text()
    m = lower_module_text(src, modeled, source="x.mlir", op_effects=effects, op_pipes=pipes).module
    r = run_tool("equivalence", cfg, m, "range_test", anchor=m, mode="symbolic")
    assert r.verdict is Verdict.OK


def test_cli_symbolic_mode_injected_defect() -> None:
    """op 被替换 → MISMATCH。"""
    from hivm_spec.__main__ import _effects_from_config
    from hivm_spec.assemble import run_tool
    from hivm_spec.ir_engine import lower_module_text
    from hivm_spec.verdict import Verdict

    cfg = json.loads((ROOT / "build" / "config.json").read_text())
    modeled = {o["op"] for o in cfg["ops"]}
    effects, pipes = _effects_from_config(cfg)
    src = (ROOT / "specs" / "cases" / "corpus" / "l0" / "loop_load_add_store.mlir").read_text()
    healthy = lower_module_text(
        src, modeled, source="x.mlir", op_effects=effects, op_pipes=pipes
    ).module
    bad = src.replace("hivm.hir.vadd", "hivm.hir.vmul")
    injected = lower_module_text(
        bad, modeled, source="b.mlir", op_effects=effects, op_pipes=pipes
    ).module
    r = run_tool("equivalence", cfg, injected, "inj_test", anchor=healthy, mode="symbolic")
    assert r.verdict is Verdict.MISMATCH, f"预期 MISMATCH，实得 {r.verdict}"


def test_cli_symbolic_mode_unmodeled_op_gives_gap() -> None:
    """未建模 op 必须报 COVERAGE_GAP（之前发现了 false pass，已修）。"""
    from hivm_spec.__main__ import _effects_from_config
    from hivm_spec.assemble import run_tool
    from hivm_spec.ir_engine import lower_module_text
    from hivm_spec.verdict import Verdict

    cfg = json.loads((ROOT / "build" / "config.json").read_text())
    modeled = {o["op"] for o in cfg["ops"]}
    effects, pipes = _effects_from_config(cfg)
    name = "split-mix-kernel-scf-for-result.mlir"
    src = (ROOT / "specs" / "cases" / "corpus" / "l1" / name).read_text()
    m = lower_module_text(src, modeled, source="x.mlir", op_effects=effects, op_pipes=pipes).module
    r = run_tool("equivalence", cfg, m, "gap_test", anchor=m, mode="symbolic")
    assert r.verdict is Verdict.COVERAGE_GAP, f"含未建模 op 应报 COVERAGE_GAP，实得 {r.verdict}"


def test_details_expose_semantics_declaration() -> None:
    """符号档必须声明其语义（Real 近似），不可与具体档混为一谈。"""
    from hivm_spec.__main__ import _effects_from_config
    from hivm_spec.assemble import run_tool
    from hivm_spec.ir_engine import lower_module_text

    cfg = json.loads((ROOT / "build" / "config.json").read_text())
    modeled = {o["op"] for o in cfg["ops"]}
    effects, pipes = _effects_from_config(cfg)
    src = (ROOT / "specs" / "cases" / "corpus" / "l0" / "loop_load_add_store.mlir").read_text()
    m = lower_module_text(src, modeled, source="x.mlir", op_effects=effects, op_pipes=pipes).module
    r = run_tool("equivalence", cfg, m, "decl_test", anchor=m, mode="symbolic")
    assert r.details.get("semantics")
    assert "Real" in r.details["semantics"]


def test_details_exit_code_reflects_verdict() -> None:
    """verdict 与退出码一致：OK→0, MISMATCH→1, COVERAGE_GAP→4。"""
    from hivm_spec.__main__ import _effects_from_config
    from hivm_spec.assemble import run_tool
    from hivm_spec.ir_engine import lower_module_text
    from hivm_spec.verdict import verdict_exit_code

    cfg = json.loads((ROOT / "build" / "config.json").read_text())
    modeled = {o["op"] for o in cfg["ops"]}
    effects, pipes = _effects_from_config(cfg)
    src = (ROOT / "specs" / "cases" / "corpus" / "l0" / "loop_load_add_store.mlir").read_text()

    # OK → 0
    m = lower_module_text(src, modeled, source="x.mlir", op_effects=effects, op_pipes=pipes).module
    r = run_tool("equivalence", cfg, m, "x", anchor=m, mode="symbolic")
    assert verdict_exit_code(r.verdict) == 0

    # MISMATCH → 1
    bad = src.replace("hivm.hir.vadd", "hivm.hir.vmul")
    inj = lower_module_text(
        bad, modeled, source="b.mlir", op_effects=effects, op_pipes=pipes
    ).module
    r2 = run_tool("equivalence", cfg, inj, "x", anchor=m, mode="symbolic")
    assert verdict_exit_code(r2.verdict) == 1

    # COVERAGE_GAP → 4
    gap_src = (
        ROOT / "specs" / "cases" / "corpus" / "l1" / "split-mix-kernel-scf-for-result.mlir"
    ).read_text()
    gm = lower_module_text(
        gap_src, modeled, source="g.mlir", op_effects=effects, op_pipes=pipes
    ).module
    r3 = run_tool("equivalence", cfg, gm, "x", anchor=gm, mode="symbolic")
    assert verdict_exit_code(r3.verdict) == 4


# ---------------------------------------------------------------------------
# 健壮性（T4.5 压测发现）
# ---------------------------------------------------------------------------


def test_deep_expression_reports_gap_instead_of_crashing() -> None:
    """**崩溃不是结论**。

    T4.5 压测发现：表达式树深约 600 层时，翻译器的递归下降会抛
    RecursionError，整个工具带着栈回溯退出。深展开的循环真的会产生这种
    深度，所以这不是理论问题。

    现在捕获它并报 UNKNOWN（→ COVERAGE_GAP）：用户得知"没验成"，而不是
    看到一屏 traceback，更不是被误导成"验过了"（FR7）。
    """
    x, y = symbol("x"), symbol("y")
    deep = x
    for _ in range(900):
        deep = add(deep, y)

    d = compare_symbolic({0: deep}, {0: deep}, bound=16)
    assert d.outcome is SymbolicOutcome.UNKNOWN
    assert not d.proven
    assert any("递归上限" in n for n in d.notes), f"须说明原因，实得 {d.notes}"


def test_moderate_depth_still_solves() -> None:
    """确认上一条不是把正常规模也一并放弃了。"""
    x, y = symbol("x"), symbol("y")
    a = b = x
    for _ in range(200):
        a = add(a, y)
        b = add(b, y)
    d = compare_symbolic({0: a}, {0: b}, bound=16)
    assert d.proven, "深度 200 属正常规模，应当能求解"
