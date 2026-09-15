"""`hivm-spec verify` 的测试：能力自述不得粉饰，前置门不得越权。

`verify` 不新增判定逻辑，因此这里**不测判定**（判定在 `test_run_checks.py` 等
处已锁）。测的是它独有的两件事：

1. **能力自述的诚实性**——没跑的说成没跑，跑过的不漏记；
2. **前置门的码位分离**——"环境跑不起来"(2) 不得与"这份 IR 没验全"(4) 或
   "验出问题"(1) 混淆（FR7）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hivm_spec.__main__ import EXIT_ENV, main
from hivm_spec.doctor import CheckStatus, DoctorReport, ProbeResult
from hivm_spec.run_checks import RunReport
from hivm_spec.verdict import Finding, ToolResult, Verdict
from hivm_spec.verify import VerifyReport

REPO = Path(__file__).resolve().parent.parent


def _r(
    tool: str,
    verdict: Verdict,
    *,
    trust: str = "provisional",
    message: str = "",
) -> ToolResult:
    return ToolResult(
        tool=tool,
        verdict=verdict,
        spec_hash="sha256:x",
        engine_version="0.1.0",
        trust=trust,
        diagnostics=[Finding(severity="info", message=message, rule="run/skipped")]
        if message
        else [],
    )


def _doctor(
    *,
    degraded: tuple[str, str] | None = None,
    missing: tuple[str, str] | None = None,
) -> DoctorReport:
    probes = [ProbeResult(name="Python", status=CheckStatus.OK, detail="3.10.21")]
    for status, one in ((CheckStatus.DEGRADED, degraded), (CheckStatus.MISSING, missing)):
        if one is None:
            continue
        name, detail = one
        probes.append(
            ProbeResult(
                name=name,
                status=status,
                detail=detail,
                affects="符号档",
                remedy="pip install -e '.[symbolic]'",
            )
        )
    return DoctorReport(probes=tuple(probes))


def _report(results: list[ToolResult], doctor: DoctorReport | None = None) -> VerifyReport:
    return VerifyReport(
        run_report=RunReport(results),
        doctor=doctor or _doctor(),
        mode="concrete",
        input_ir="after.mlir",
        config_path="build/config.json",
        anchor=None,
    )


# ---------------------------------------------------------------------------
# 能力自述
# ---------------------------------------------------------------------------


def test_gap_count_matches_gap_verdicts() -> None:
    """自述里的"没验成"计数必须逐项对应，不能只报个总数。"""
    v = _report(
        [
            _r("ub_occupancy", Verdict.OK),
            _r("timeline", Verdict.COVERAGE_GAP),
            _r("sync_pairing", Verdict.COVERAGE_GAP),
            _r("uninit_read", Verdict.OK),
            _r("equivalence", Verdict.COVERAGE_GAP),
        ]
    )
    assert v.gaps == ["timeline", "sync_pairing", "equivalence"]
    assert v.problems == []
    assert "3" in v.summary()


def test_executed_lists_every_check_not_only_clean_ones() -> None:
    """跑过但记缺口的检查也必须在账上——否则报告显得全绿。"""
    v = _report([_r("ub_occupancy", Verdict.OK), _r("timeline", Verdict.COVERAGE_GAP)])
    assert v.executed == ["ub_occupancy", "timeline"]


def test_skipped_checks_do_not_pollute_trust_display() -> None:
    """编排层替缺锚点的等价验证造的记录 trust 是 n/a（根本没跑）。

    若把它计入，汇报会出现 `n/a, provisional` 这种读起来像"混合信任源"的字符串，
    让人以为有一份权威结论混在里面。
    """
    v = _report(
        [
            _r("ub_occupancy", Verdict.OK),
            _r("equivalence", Verdict.COVERAGE_GAP, trust="n/a", message="未提供 --anchor"),
        ]
    )
    assert v.trust == "provisional"


def test_problem_and_gap_are_not_conflated() -> None:
    """确证问题 + 别处缺口：两者都要出现，谁也不盖谁。"""
    v = _report(
        [
            _r("equivalence", Verdict.MISMATCH),
            _r("timeline", Verdict.COVERAGE_GAP),
        ]
    )
    assert v.problems == ["equivalence"] and v.gaps == ["timeline"]
    rendered = v.render()
    assert "没验成" in rendered and "MISMATCH" in rendered


def test_missing_anchor_is_stated_in_render() -> None:
    """不给锚点时，必须在渲染里明写"本轮不含等效性维度"。

    否则一份"5 项里有 4 项 OK"的报告会被读成完整验证。
    """
    v = _report([_r("ub_occupancy", Verdict.OK)])
    assert "--anchor" in v.render()

    v2 = _report([_r("ub_occupancy", Verdict.OK)])
    v2.anchor = "before.mlir"
    assert "--anchor：等价验证未执行" not in v2.render()


def test_limitations_come_from_doctor_not_from_optimism() -> None:
    """降级运行必须把限制写进结论（缺 z3 不得读成"符号档也过了"）。"""
    v = _report(
        [_r("ub_occupancy", Verdict.OK)],
        doctor=_doctor(degraded=("z3", "未安装")),
    )
    assert any("z3" in lim for lim in v.limitations)
    assert "能力限制" in v.render()


def test_json_carries_self_description_and_probes(tmp_path: Path) -> None:
    """agent 只拿 JSON 也必须能知道"这轮验了什么、没验什么"。"""
    v = _report([_r("ub_occupancy", Verdict.OK), _r("timeline", Verdict.COVERAGE_GAP)])
    out = tmp_path / "v.json"
    v.write_json(str(out))
    doc: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    caps = doc["capabilities"]
    assert caps["executed"] == ["ub_occupancy", "timeline"]
    assert caps["gaps"] == ["timeline"]
    assert "1 项未能完成验证" in caps["self_description"]
    assert "provisional" in caps["self_description"]
    assert {p["name"] for p in caps["probes"]} == {"Python"}
    assert doc["verify_version"]


def test_json_is_deterministic(tmp_path: Path) -> None:
    """与 `run --json` 同款确定性（FR8）：同输入两次逐字节一致。"""
    v = _report([_r("ub_occupancy", Verdict.OVERFLOW)])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    v.write_json(str(a))
    v.write_json(str(b))
    assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------------------
# 前置门：码位分离
# ---------------------------------------------------------------------------


def test_verify_blocks_without_bindings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """无 bindings 时 `verify` 必须挡住并退出 2，而不是产出 COVERAGE_GAP(4)。

    这是 FR7 要求的三态分离：跑不起来(2) / 没验全(4) / 验出问题(1)。
    若退 4，脚本层就会把"环境缺 bindings"当成"这份 IR 有覆盖缺口"。
    """
    monkeypatch.delenv("HIVM_SPEC_BINDINGS", raising=False)
    monkeypatch.delenv("HIVM_SPEC_BINDINGS_ALT", raising=False)
    rc = main(["verify", str(REPO / "specs/cases/corpus/l0/loop_load_add_store.mlir")])
    err = capsys.readouterr().err
    assert rc == EXIT_ENV, f"期望前置门 2，实得 {rc}"
    assert "未产出任何验证结论" in err
    assert "修复" in err, "挡住必须给可执行的下一步（FR5）"


def test_verify_gate_does_not_emit_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """前置门挡住时 stdout 上不得出现任何 verdict 字样——没有结论就是没有。"""
    monkeypatch.delenv("HIVM_SPEC_BINDINGS", raising=False)
    main(["verify", str(REPO / "specs/cases/corpus/l0/loop_load_add_store.mlir")])
    out = capsys.readouterr().out
    for word in ("OK", "MISMATCH", "OVERFLOW", "DEADLOCK", "COVERAGE_GAP"):
        assert word not in out


def test_env_const_is_2_not_reusing_gap_code() -> None:
    """退出码 2 是新增位，必须与既有 1/3/4/5 互不重叠。"""
    from hivm_spec.verdict import verdict_exit_code

    used = {verdict_exit_code(v) for v in Verdict} | {3}
    assert EXIT_ENV == 2
    assert EXIT_ENV not in used


def test_cwd_default_outranks_env_config_when_both_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """配置定位优先级：`-c` > CWD 的 `build/config.json` > `HIVM_SPEC_CONFIG`。

    在仓内工作时以仓内既有配置为准（否则改了描述却还在用 env 指向的旧配置，
    spec_hash 会对不上）；env 是给**离开仓目录**的安装态用法兜底的。
    """
    from hivm_spec.__main__ import _resolve_config

    monkeypatch.chdir(tmp_path)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HIVM_SPEC_CONFIG", str(tmp_path / "elsewhere.json"))

    resolved, rc = _resolve_config(None, None)
    assert rc == 0 and resolved == "build/config.json"


def test_env_config_pointing_nowhere_is_an_error_not_a_silent_gen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`HIVM_SPEC_CONFIG` 指向不存在的文件 → 报错并给 gen 指引，不悄悄造一份。

    用户明确说了用哪份配置，就不该被静默替换（同 `-c` 的既有口径）。
    """
    from hivm_spec.__main__ import EXIT_FAIL, _resolve_config

    monkeypatch.chdir(tmp_path)  # CWD 无 build/config.json，才会走到 env 这一级
    monkeypatch.setenv("HIVM_SPEC_CONFIG", str(tmp_path / "nope.json"))

    resolved, rc = _resolve_config(None, None)
    assert resolved is None and rc == EXIT_FAIL
    err = capsys.readouterr().err
    assert "不存在" in err and "gen" in err


def test_env_config_fallback_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`HIVM_SPEC_CONFIG` 让安装后的调用可以脱离仓目录（CWD 相对默认会失效）。"""
    cfg = tmp_path / "cfg.json"
    monkeypatch.chdir(tmp_path)  # 关键：离开仓目录，CWD 相对的 build/config.json 找不到
    monkeypatch.setenv("HIVM_SPEC_CONFIG", str(cfg))
    rc = main(
        [
            "gen",
            str(REPO / "specs/toy.py"),
            str(REPO / "specs/cv.py"),
            "-o",
            str(cfg),
        ]
    )
    assert rc == 0 and cfg.is_file(), "gen 应能把配置写到指定位置"

    # 配置已就位：即便 CWD 里没有 build/config.json，也应通过 HIVM_SPEC_CONFIG 找到
    from hivm_spec.__main__ import _resolve_config

    resolved, rc2 = _resolve_config(None, None)
    assert rc2 == 0 and resolved == str(cfg)


@pytest.mark.requires_bindings
def test_verify_agrees_with_run_on_the_same_ir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """同一份 IR，`verify` 与 `run` 的逐项 verdict 必须一致。

    两者共用 `_run_all`，这条测试锁的是"别有一天分叉出两套编排"。
    """
    ir = REPO / "specs/cases/corpus/l0/ub_overflow_injected.mlir"
    j = tmp_path / "v.json"
    rc = main(["verify", str(ir), "--json", str(j)])
    assert rc == 1, "该语料是注入的溢出用例，应为 OVERFLOW"
    doc: dict[str, Any] = json.loads(j.read_text(encoding="utf-8"))
    per_tool = {c["tool"]: c["verdict"] for c in doc["checks"]}
    assert per_tool["ub_occupancy"] == Verdict.OVERFLOW.value
    assert doc["capabilities"]["input_ir"] == str(ir)
