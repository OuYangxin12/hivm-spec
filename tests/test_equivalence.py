"""T3.5 差分对拍测试。

验收核心（milestone-plan §5 T3.5）：两份结构不同的 IR 出 verdict（含 MISMATCH）。
最要紧的一条是 **M3 卡 §4 要点 2**：锚点缺失绝不静默降级为自比后报 OK。
"""

from __future__ import annotations

import numpy as np
import pytest

from hivm_spec.equivalence import compare, tolerance_from_config
from hivm_spec.interpret import ExecResult, interpret
from hivm_spec.numeric import Tolerance
from hivm_spec.values import concrete
from hivm_spec.verdict import Verdict, verdict_exit_code
from hivm_spec.vir import Coverage, Loc, VLoop, VModule, VNode, VRegion

LOC = Loc(file="t.mlir", line=7)


def _cfg(add_expr: str = "elementwise(add, a, b, into=out)") -> dict:
    return {
        "ops": [
            {
                "op": "hivm.hir.vadd",
                "pipe": "PIPE_V",
                "params": [
                    {"name": "a", "kind": "in"},
                    {"name": "b", "kind": "in"},
                    {"name": "out", "kind": "out"},
                ],
                "value": {"kind": "declarative", "expr": add_expr},
            },
            {
                "op": "hivm.hir.mystery",
                "pipe": "PIPE_V",
                "params": [{"name": "a", "kind": "in"}, {"name": "out", "kind": "out"}],
                "value": {"kind": "host_fn", "name": "f", "reason": "布局代数"},
            },
        ],
        "vm": {"pipes": ["PIPE_V"]},
        "checks": [
            {"name": "equivalence", "options": {"rtol": 1e-5, "atol": 1e-8}},
        ],
    }


def _module(items: tuple) -> VModule:
    return VModule(
        source="t.mlir",
        items=(VRegion(id="f0", kind="func", loc=LOC, items=items),),
        coverage=Coverage(modeled_ops=("hivm.hir.vadd",)),
        arch="a3",
    )


def _add_module() -> VModule:
    return _module((VNode(id="n1", op="hivm.hir.vadd", loc=LOC, operands=("%a", "%b", "%o")),))


def _f32(*vals: float) -> object:
    return concrete(np.array(vals, dtype=np.float32), "f32")


def _inputs() -> dict:
    return {"%a": _f32(1.0, 2.0, 3.0), "%b": _f32(0.5, 0.5, 0.5)}


# ---------------------------------------------------------------------------
# 容差来自描述（M3 卡 §4 要点 3）
# ---------------------------------------------------------------------------


def test_tolerance_comes_from_description() -> None:
    tol = tolerance_from_config(_cfg())
    assert tol.rtol == 1e-5
    assert tol.atol == 1e-8


def test_tolerance_falls_back_to_defaults_when_undeclared() -> None:
    """描述没声明时用默认值，而不是"无容差"（那会让任何浮点都发散）。"""
    tol = tolerance_from_config({"checks": []})
    assert tol.rtol > 0 and tol.atol > 0


# ---------------------------------------------------------------------------
# 相同 / 不同
# ---------------------------------------------------------------------------


def test_identical_execution_is_ok() -> None:
    cfg = _cfg()
    m = _add_module()
    left = interpret(m, cfg, _inputs())
    right = interpret(m, cfg, _inputs())
    res = compare(left, right, Tolerance())
    assert res.verdict is Verdict.OK
    assert res.first is None
    assert res.compared == 1
    assert res.ok


def test_injected_semantic_defect_is_caught_and_localized() -> None:
    """注入语义缺陷（add→sub）必须被检出，且定位到具体 op 与源位置。

    这是 M3 的成败判据（卡片开头）：不是"工具能跑"，而是"注入缺陷能被检出并
    定位到首个发散点"。
    """
    m = _add_module()
    good = interpret(m, _cfg(), _inputs())
    bad = interpret(m, _cfg("elementwise(sub, a, b, into=out)"), _inputs())

    res = compare(bad, good, Tolerance())
    assert res.verdict is Verdict.MISMATCH
    assert res.first is not None
    assert res.first.op == "hivm.hir.vadd"
    assert res.first.file == "t.mlir" and res.first.line == 7
    # 双侧值摘要都在场——只报"不相等"无法判断是谁错了（FR5）
    assert res.first.left_summary and res.first.right_summary
    assert res.first.max_abs > 0
    assert "t.mlir:7" in res.first.describe()


def test_verdict_exit_codes_distinguish_gap_from_problem() -> None:
    """缺口与问题必须用不同退出码——脚本要能区分"验证失败"与"没验成"。"""
    assert verdict_exit_code(Verdict.OK) == 0
    assert verdict_exit_code(Verdict.MISMATCH) == 1
    assert verdict_exit_code(Verdict.COVERAGE_GAP) == 4


# ---------------------------------------------------------------------------
# 容差：不假阳性、也不放过真发散
# ---------------------------------------------------------------------------


def test_within_tolerance_drift_is_not_a_mismatch() -> None:
    """容差内的浮点抖动不得报 MISMATCH（假阳性会淹没真问题，NFR2）。

    注意这里哈希**必然不同**（字节不同），所以这条同时证明了"判等走容差而非
    走哈希"——只比哈希的话这个用例会误报。
    """
    m = _add_module()
    base = _inputs()
    # 扰动量刻意选在 f32 分辨率之上、容差之下：1e-9 会被 f32 直接舍掉（字节相同），
    # 那样就测不到"哈希不同但容差内"这个真正要覆盖的情形。
    drifted = {k: concrete(np.asarray(v.array) * (1 + 1e-7), "f32") for k, v in base.items()}

    left = interpret(m, _cfg(), drifted)
    right = interpret(m, _cfg(), base)
    assert left.traces[0].out_hash != right.traces[0].out_hash, "前提：字节确实不同"
    assert compare(left, right, Tolerance(rtol=1e-5, atol=1e-8)).verdict is Verdict.OK


def test_beyond_tolerance_drift_is_a_mismatch() -> None:
    m = _add_module()
    base = _inputs()
    drifted = {k: concrete(np.asarray(v.array) * 1.5, "f32") for k, v in base.items()}
    res = compare(interpret(m, _cfg(), drifted), interpret(m, _cfg(), base), Tolerance(rtol=1e-5))
    assert res.verdict is Verdict.MISMATCH


def test_hash_equality_short_circuits_comparison() -> None:
    """哈希相同即内容逐字节相同——此时无需容差比较也应判等。"""
    m = _add_module()
    left = interpret(m, _cfg(), _inputs())
    right = interpret(m, _cfg(), _inputs())
    assert left.traces[0].out_hash == right.traces[0].out_hash
    # 即便给一个荒谬地严格的容差，哈希相同仍应 OK
    assert compare(left, right, Tolerance(rtol=0.0, atol=0.0)).verdict is Verdict.OK


# ---------------------------------------------------------------------------
# 反自欺：缺口不得报 OK
# ---------------------------------------------------------------------------


def test_unmodeled_op_forces_coverage_gap_not_ok() -> None:
    """有未建模 op 时必须 COVERAGE_GAP。

    "没发现发散"可能只是因为没算那一步。此时报 OK 是把"没检查"说成"检查通过"
    （FR4/FR6）。
    """
    m = _module((VNode(id="n1", op="hivm.hir.mystery", loc=LOC, operands=("%a", "%o")),))
    left = interpret(m, _cfg(), {"%a": _f32(1.0)})
    right = interpret(m, _cfg(), {"%a": _f32(1.0)})
    assert left.has_unmodeled

    res = compare(left, right, Tolerance())
    assert res.verdict is Verdict.COVERAGE_GAP
    assert res.verdict is not Verdict.OK
    assert any("未建模" in n for n in res.notes)
    # 缺口不算"发现了问题"——它说的是"我没能完成验证"
    assert not res.verdict.is_problem


def test_step_count_mismatch_is_a_gap_not_a_cherry_picked_compare() -> None:
    """两侧步数不同时不得挑"能对上的部分"比——那等于自选有利证据（FR6）。"""
    one = _add_module()
    loop = VRegion(
        id="L",
        kind="for",
        loc=LOC,
        items=(VNode(id="n1", op="hivm.hir.vadd", loc=LOC, operands=("%a", "%b", "%o")),),
        loop=VLoop(iv="i", trip_count=3),
    )
    three = _module((loop,))

    res = compare(
        interpret(three, _cfg(), _inputs()), interpret(one, _cfg(), _inputs()), Tolerance()
    )
    assert res.verdict is Verdict.COVERAGE_GAP
    assert res.compared == 0
    assert any("步数不同" in n for n in res.notes)


def test_no_comparable_values_is_a_gap_not_ok() -> None:
    """一步都没比成时不得报 OK——"对拍未实际发生"必须显式说出来。"""
    empty = ExecResult(traces=(), env=interpret(_add_module(), _cfg(), _inputs()).env, gaps=())
    res = compare(empty, empty, Tolerance())
    assert res.verdict is Verdict.COVERAGE_GAP
    assert any("未实际发生" in n for n in res.notes)


def test_truncation_is_disclosed_in_notes() -> None:
    """截断必须如实标注——结论只在展开界内成立。"""
    loop = VRegion(
        id="L",
        kind="for",
        loc=LOC,
        items=(VNode(id="n1", op="hivm.hir.vadd", loc=LOC, operands=("%a", "%b", "%o")),),
        loop=VLoop(iv="i", trip_count=None),
    )
    m = _module((loop,))
    left = interpret(m, _cfg(), _inputs(), bound=3)
    right = interpret(m, _cfg(), _inputs(), bound=3)
    assert left.truncated

    res = compare(left, right, Tolerance())
    assert res.verdict is Verdict.OK  # 界内确实一致
    assert any("截断" in n for n in res.notes), res.notes
    assert any("已完整验证" in n for n in res.notes), "必须点明不可读作已完整验证"


def test_first_divergence_reports_impact_count() -> None:
    """只报首现 + 影响计数——同一发散会沿数据流传播成一大片（FR5）。"""
    loop = VRegion(
        id="L",
        kind="for",
        loc=LOC,
        items=(VNode(id="n1", op="hivm.hir.vadd", loc=LOC, operands=("%a", "%b", "%o")),),
        loop=VLoop(iv="i", trip_count=4),
    )
    m = _module((loop,))
    good = interpret(m, _cfg(), _inputs())
    bad = interpret(m, _cfg("elementwise(mul, a, b, into=out)"), _inputs())

    res = compare(bad, good, Tolerance())
    assert res.verdict is Verdict.MISMATCH
    assert res.first is not None
    assert res.first.label.endswith("@i0"), "首发散应是第 0 次迭代"
    assert res.impacted == 3, f"另外 3 次迭代受影响，实际 {res.impacted}"


def test_tolerance_is_echoed_in_result() -> None:
    """结论必须回显实际生效的容差（M3 卡 §4 要点 3）。"""
    tol = Tolerance(rtol=1e-3, atol=1e-6)
    res = compare(
        interpret(_add_module(), _cfg(), _inputs()),
        interpret(_add_module(), _cfg(), _inputs()),
        tol,
    )
    assert res.tolerance.rtol == 1e-3
    assert res.tolerance.atol == 1e-6


def test_compare_has_no_self_compare_mode() -> None:
    """compare 必须要求两个结果——没有可退化的"单参数自比"模式。

    锁定的是 API 形状：自比恒等于"通过"，却什么都没验证（§4 要点 2）。
    """
    import inspect

    sig = inspect.signature(compare)
    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty and p.kind is not p.VAR_KEYWORD
    ]
    assert len(required) == 3, f"应为 (left, right, tolerance)，实际 {required}"


# ---------------------------------------------------------------------------
# CLI：--anchor 是选项，位置参数恒为一份 IR（M3 卡 §4 要点 1）
# ---------------------------------------------------------------------------


def test_cli_exposes_anchor_as_option_not_positional() -> None:
    """锚点必须是 --anchor 选项，位置参数恒一份 IR（D12 契约不破）。"""
    from hivm_spec.__main__ import _build_parser

    args = _build_parser().parse_args(
        ["tool", "equivalence", "a.mlir", "--anchor", "b.mlir", "-c", "cfg.json"]
    )
    assert args.inputs == ["a.mlir"], "位置参数恒为一份待验 IR"
    assert args.anchor == "b.mlir"


def test_cli_anchor_defaults_to_none_not_self() -> None:
    """不给 --anchor 时缺省为 None，**不是**待验 IR 自身。

    若缺省成自身，等价验证就会恒报 OK 而什么都没验（§4 要点 2）。
    """
    from hivm_spec.__main__ import _build_parser

    args = _build_parser().parse_args(["tool", "equivalence", "a.mlir", "-c", "cfg.json"])
    assert args.anchor is None
    assert args.anchor != args.inputs[0]


def test_run_equivalence_without_anchor_is_gap_with_exit_4() -> None:
    """无锚点 → COVERAGE_GAP + 退出码 4，而非 OK/0。"""
    from hivm_spec.assemble import run_equivalence

    result = run_equivalence(_cfg(), _add_module(), "sha256:x", anchor=None)
    assert result.verdict is Verdict.COVERAGE_GAP
    assert result.exit_code == 4
    assert any(f.rule == "equivalence/no-anchor" for f in result.diagnostics)
    # 容差仍须回显，便于读者知道"本来会用什么口径比"
    assert result.details["tolerance"]["rtol"] > 0


def test_run_equivalence_without_declared_check_is_untrusted() -> None:
    """描述没声明 equivalence check → UNTRUSTED_DESCRIPTION（容差口径不明）。"""
    from hivm_spec.assemble import run_equivalence

    cfg = _cfg()
    cfg["checks"] = []
    result = run_equivalence(cfg, _add_module(), "sha256:x", anchor=_add_module())
    assert result.verdict is Verdict.UNTRUSTED_DESCRIPTION


def test_run_equivalence_compares_two_different_irs() -> None:
    """结构不同的两份 IR 出 verdict——这是 T3.5 的验收句。

    注意 run_equivalence 两侧共用**同一份配置文档**（语义口径必须一致，否则
    比的是两套语义而不是两份 IR）。所以缺陷要注入在 **IR** 上：待验侧多算一轮
    vadd，等价性因此被破坏。
    """
    from hivm_spec.assemble import run_equivalence

    anchor = _add_module()
    # 待验侧：循环 3 次 —— 与锚点的单次执行结构不同
    loop = VRegion(
        id="L",
        kind="for",
        loc=LOC,
        items=(VNode(id="n1", op="hivm.hir.vadd", loc=LOC, operands=("%a", "%b", "%o")),),
        loop=VLoop(iv="i", trip_count=3),
    )
    candidate = _module((loop,))

    result = run_equivalence(_cfg(), candidate, "sha256:x", anchor=anchor)
    # 合成 VIR 没有 allocs，故 run_equivalence 推不出入口输入，vadd 无法求值。
    # 结果仍是缺口而非 OK——**这正是要的**：输入推不出来时报"没验成"，而不是
    # 拿空输入跑一遍然后报"通过"（FR6）。真实语料的 MISMATCH 路径由
    # test_real_corpus_mismatch_is_detected_end_to_end 覆盖。
    assert result.verdict is Verdict.COVERAGE_GAP
    assert result.exit_code == 4
    assert any("未建模" in f.message or "步数不同" in f.message for f in result.diagnostics)
    assert result.details["anchor_fingerprint"]


def test_run_equivalence_echoes_input_provenance() -> None:
    """结论须回显输入来源（种子等），否则读者无法重放（FR5/FR8）。"""
    from hivm_spec.assemble import run_equivalence

    m = _add_module()
    result = run_equivalence(_cfg(), m, "sha256:x", anchor=m)
    prov = result.details["inputs"]
    assert prov["seed"] > 0
    assert "shrunk" in prov


@pytest.mark.requires_bindings
def test_real_corpus_mismatch_is_detected_end_to_end() -> None:
    """**M3 成败判据**：真实语料上注入语义缺陷，必须检出并定位到首个发散点。

    注入方式是把 IR 里的 `hivm.hir.vadd` 换成 `hivm.hir.vmul`——这是真实的 pass
    缺陷形态（算子选错），且两个 op 都已在描述里建模，故不会退化成覆盖缺口。
    """
    import importlib.util
    from pathlib import Path

    from hivm_spec.assemble import run_equivalence
    from hivm_spec.generate import generate
    from hivm_spec.ir_engine import lower_module_text

    root = Path(__file__).resolve().parents[1]
    loader = importlib.util.spec_from_file_location("toy_eq", root / "specs" / "toy.py")
    assert loader and loader.loader
    mod = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod)
    import json as _json

    cfg = _json.loads(generate(mod.spec, timestamp="2026-01-01T00:00:00+00:00").config_bytes)
    modeled = {o["op"] for o in cfg["ops"]}
    pipes = {o["op"]: o.get("pipe", "") for o in cfg["ops"]}

    src = (root / "specs" / "cases" / "corpus" / "l0" / "loop_load_add_store.mlir").read_text()
    broken_src = src.replace("hivm.hir.vadd", "hivm.hir.vmul")
    assert broken_src != src, "注入前提：语料里确实有 vadd"

    def _lower(text: str, name: str) -> object:
        return lower_module_text(
            text, modeled, source=name, op_pipes=pipes, arch=cfg.get("arch", "a3")
        ).module

    anchor = _lower(src, "anchor.mlir")
    candidate = _lower(broken_src, "broken.mlir")

    # 先确认健康对拍是 OK——否则下面的 MISMATCH 可能只是噪音
    same = run_equivalence(cfg, _lower(src, "anchor.mlir"), "sha256:x", anchor=anchor, bound=4)
    assert same.verdict is Verdict.OK, [f.message for f in same.diagnostics]

    result = run_equivalence(cfg, candidate, "sha256:x", anchor=anchor, bound=4)
    assert result.verdict is Verdict.MISMATCH, [f.message for f in result.diagnostics]
    assert result.exit_code == 1

    first = result.details["first_divergence"]
    assert first["op"] == "hivm.hir.vmul"
    assert first["label"].endswith("@i0"), first["label"]
    assert first["line"] > 0, "必须定位到源码行"
    assert first["left"] and first["right"], "双侧值摘要都要在场"
    assert first["max_abs"] > 0
    # 发散沿数据流传播 → 后续步受影响，只报首现 + 计数
    assert result.details["impacted_steps"] >= 1
    assert result.details["tolerance"]["rtol"] > 0


# ---------------------------------------------------------------------------
# T3.6：逐 op 值哈希 trace 随结论输出
# ---------------------------------------------------------------------------


def test_details_carry_per_op_hash_trace() -> None:
    """结论里必须带逐 op 值哈希 trace，供人工比对与二次定位（T3.6）。"""
    from hivm_spec.assemble import run_equivalence

    m = _add_module()
    result = run_equivalence(_cfg(), m, "sha256:x", anchor=m)
    trace = result.details["trace"]
    assert trace, "trace 不得为空"
    row = trace[0]
    assert {"seq", "label", "op", "line", "left", "right", "same"} <= set(row)
    assert row["same"] is True


def test_trace_holds_hashes_not_full_tensors() -> None:
    """trace 只放哈希与标签，不放全量张量。

    全量张量动辄上百 MB，而真正要回答的是"从哪一步起两侧不同"——哈希足够。
    """
    from hivm_spec.assemble import run_equivalence

    m = _add_module()
    result = run_equivalence(_cfg(), m, "sha256:x", anchor=m)
    for row in result.details["trace"]:
        assert isinstance(row["left"], str)
        assert row["left"] == "" or row["left"].startswith("sha256:")


@pytest.mark.requires_bindings
def test_trace_shows_divergence_propagation_boundary() -> None:
    """trace 必须让"哪一步开始发散"一眼可读：之前 same=True，之后 same=False。

    这是 T3.6 定位能力的可检验形态——不只报一个结论，还留下可复核的证据链。
    """
    import importlib.util
    import json as _json
    from pathlib import Path

    from hivm_spec.assemble import run_equivalence
    from hivm_spec.generate import generate
    from hivm_spec.ir_engine import lower_module_text

    root = Path(__file__).resolve().parents[1]
    loader = importlib.util.spec_from_file_location("toy_tr", root / "specs" / "toy.py")
    assert loader and loader.loader
    mod = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod)
    cfg = _json.loads(generate(mod.spec, timestamp="2026-01-01T00:00:00+00:00").config_bytes)
    modeled = {o["op"] for o in cfg["ops"]}
    pipes = {o["op"]: o.get("pipe", "") for o in cfg["ops"]}

    src = (root / "specs" / "cases" / "corpus" / "l0" / "loop_load_add_store.mlir").read_text()
    broken = src.replace("hivm.hir.vadd", "hivm.hir.vmul")

    def _lower(text: str, name: str):  # type: ignore[no-untyped-def]
        return lower_module_text(text, modeled, source=name, op_pipes=pipes).module

    result = run_equivalence(
        cfg, _lower(broken, "b.mlir"), "sha256:x", anchor=_lower(src, "a.mlir"), bound=4
    )
    trace = result.details["trace"]
    # load 在缺陷之前 → 一致；其后全部发散（沿数据流传播）
    assert trace[0]["same"] is True, trace[0]
    assert all(r["same"] is False for r in trace[1:]), trace
    # 首发散点与 trace 里第一个 same=False 的行一致
    first_false = next(r for r in trace if not r["same"])
    assert result.details["first_divergence"]["seq"] == first_false["seq"]
