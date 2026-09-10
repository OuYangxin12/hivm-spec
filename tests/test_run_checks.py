"""`hivm-spec run` 编排的测试。

最关键的是聚合语义：**已确证的问题不能被别处的覆盖缺口掩盖**，而**没验全也
不能给干净的 OK**。这两条各有专门的测试。
"""

from __future__ import annotations

import pytest

from hivm_spec.run_checks import ORCHESTRATOR_VERSION, aggregate
from hivm_spec.verdict import Finding, ToolResult, Verdict, verdict_exit_code


def _r(tool: str, verdict: Verdict, *, rule: str = "t/x") -> ToolResult:
    diags = []
    if verdict is Verdict.MISMATCH:
        diags = [Finding(severity="error", message="m", rule=rule)]
    return ToolResult(
        tool=tool,
        verdict=verdict,
        spec_hash="h",
        engine_version="0.1.0",
        trust="provisional",
        diagnostics=diags,
    )


# ---------------------------------------------------------------------------
# 聚合：最关键的语义
# ---------------------------------------------------------------------------


def test_confirmed_bug_not_masked_by_unrelated_gap() -> None:
    """一个检查确证 MISMATCH，另一个检查因看不懂而 GAP → 整体必须是 MISMATCH。

    反例（本测试防的就是它）：若 GAP 永远优先，等价检查发现了真 bug，却被一个
    "模块无同步结构、未能分析"的 timeline 拉成 COVERAGE_GAP，退出码从 1 变 4，
    CI 就会漏放真缺陷。

    这与 `assemble._decide_verdict` 的口径一致：已确证的问题优先于别处的缺口。
    """
    results = [
        _r("ub_occupancy", Verdict.OK),
        _r("timeline", Verdict.COVERAGE_GAP),
        _r("sync_pairing", Verdict.COVERAGE_GAP),
        _r("uninit_read", Verdict.OK),
        _r("equivalence", Verdict.MISMATCH),
    ]
    v = aggregate(results)
    assert v is Verdict.MISMATCH
    assert verdict_exit_code(v) == 1, "真 bug 必须让进程失败"


def test_each_concrete_problem_beats_gap() -> None:
    for problem in (Verdict.OVERFLOW, Verdict.DEADLOCK, Verdict.MISMATCH):
        assert aggregate([_r("a", Verdict.COVERAGE_GAP), _r("b", problem)]) is problem


def test_gap_blocks_clean_ok() -> None:
    """没有真问题，但有一项没验成 → 不能给 OK（缺口挡在 OK 前面）。"""
    assert aggregate([_r("a", Verdict.OK), _r("b", Verdict.COVERAGE_GAP)]) is Verdict.COVERAGE_GAP


def test_all_ok_is_ok() -> None:
    assert aggregate([_r("a", Verdict.OK), _r("b", Verdict.OK)]) is Verdict.OK


def test_untrusted_description_dominates_everything() -> None:
    """描述不可信时所有结论失去根基，即使有具体问题也先报不可信。"""
    results = [
        _r("a", Verdict.UNTRUSTED_DESCRIPTION),
        _r("b", Verdict.MISMATCH),
        _r("c", Verdict.COVERAGE_GAP),
    ]
    assert aggregate(results) is Verdict.UNTRUSTED_DESCRIPTION


def test_empty_results_is_gap_not_ok() -> None:
    """一项检查都没跑就说"没问题"是自欺。"""
    assert aggregate([]) is Verdict.COVERAGE_GAP


# ---------------------------------------------------------------------------
# 跳过：不静默消失
# ---------------------------------------------------------------------------


def test_no_anchor_records_gap_instead_of_silently_dropping() -> None:
    """无 --anchor：等价验证必须以一条 GAP 出现在报告里，而不是被删掉。

    删掉会让"4 项全 OK"被读成"这份 IR 没问题"，而最重要的语义等价根本没验。
    """
    from hivm_spec.run_checks import RunReport, _skipped

    rec = _skipped("equivalence", "h", "no anchor")
    assert rec.verdict is Verdict.COVERAGE_GAP
    assert rec.diagnostics[0].rule == "run/skipped"
    report = RunReport([_r("a", Verdict.OK), rec])
    assert report.verdict is Verdict.COVERAGE_GAP
    assert "equivalence" in report.to_json()["checks"][1]["tool"]


# ---------------------------------------------------------------------------
# 端到端（需 bindings）：一条命令跑五检查
# ---------------------------------------------------------------------------


@pytest.mark.requires_bindings
def test_run_e2e_selfcheck() -> None:
    """真实 IR 端到端：恒跑四项 + 无锚点等价记缺口。"""
    import json
    from pathlib import Path

    from hivm_spec.__main__ import _lower_ir, _resolve_config
    from hivm_spec.assemble import load_config
    from hivm_spec.run_checks import run_checks

    root = Path(__file__).resolve().parents[1]
    ir = root / "specs/cases/corpus/l0/loop_load_add_store.mlir"
    resolved, rc = _resolve_config(None, None)
    assert resolved is not None and rc == 0
    config, spec_hash = load_config(Path(resolved))

    lowered, rc = _lower_ir(str(ir), config)
    assert lowered is not None and rc == 0

    # 无锚点：等价验证记缺口，整体不能是 OK
    report = run_checks(lowered.module, config, spec_hash)
    tools = {r.tool for r in report.results}
    assert {"ub_occupancy", "timeline", "sync_pairing", "uninit_read", "equivalence"} <= tools
    eq = next(r for r in report.results if r.tool == "equivalence")
    assert eq.verdict is Verdict.COVERAGE_GAP

    payload = report.to_json()
    assert payload["orchestrator_version"] == ORCHESTRATOR_VERSION
    # JSON 可确定性序列化（FR8）
    json.dumps(payload, sort_keys=True)


@pytest.mark.requires_bindings
def test_run_e2e_injected_bug_fails() -> None:
    """注入 vadd→vmul：等价检查 MISMATCH，整体退出码必须是 1，不被 timeline 的缺口掩盖。"""
    from pathlib import Path

    from hivm_spec.__main__ import _lower_ir, _resolve_config
    from hivm_spec.assemble import load_config
    from hivm_spec.run_checks import run_checks

    root = Path(__file__).resolve().parents[1]
    anchor_ir = root / "specs/cases/corpus/l0/loop_load_add_store.mlir"
    buggy = anchor_ir.read_text(encoding="utf-8").replace("hivm.hir.vadd", "hivm.hir.vmul")

    resolved, _ = _resolve_config(None, None)
    assert resolved is not None
    config, spec_hash = load_config(Path(resolved))

    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(buggy)
        bug_path = f.name

    bad_lowered, _ = _lower_ir(bug_path, config)
    anchor_lowered, _ = _lower_ir(str(anchor_ir), config)
    assert bad_lowered is not None and anchor_lowered is not None

    report = run_checks(bad_lowered.module, config, spec_hash, anchor=anchor_lowered.module)
    eq = next(r for r in report.results if r.tool == "equivalence")
    assert eq.verdict is Verdict.MISMATCH
    assert report.verdict is Verdict.MISMATCH, "真 bug 不被别处缺口掩盖"
    assert report.exit_code == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.mark.requires_bindings
def test_cli_run_smoke(capsys, tmp_path) -> None:
    from pathlib import Path

    from hivm_spec.__main__ import main

    ir = Path(__file__).resolve().parents[1] / "specs/cases/corpus/l0/loop_load_add_store.mlir"
    out_json = tmp_path / "r.json"
    rc = main(["run", str(ir), "--json", str(out_json)])
    captured = capsys.readouterr()
    assert "综合结论" in captured.out
    # 无锚点 → 含缺口 → 退出码 4
    assert rc == 4
    assert out_json.is_file()
