"""生成链路与 CLI 的端到端测试（T0.3 / T0.4 / T0.5）。

核心断言是 FR8：同一描述两次生成**逐字节一致**。这不是锦上添花——判定依据若
不可复现，一切结论都不可审计。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hivm_spec.__main__ import EXIT_FAIL, EXIT_OK, EXIT_PENDING, main
from hivm_spec.generate import (
    ENGINE_VERSION,
    DriftEntry,
    TrustLedger,
    generate,
    validate_config,
    write_outputs,
)
from hivm_spec.spec import Attr as A
from hivm_spec.spec import In, Out, Spec, Trust, host_fn, sync_set, wr

REPO_ROOT = Path(__file__).resolve().parent.parent
TOY = REPO_ROOT / "specs" / "toy.py"
FIXED_TS = "2026-09-09T00:00:00+00:00"


def _spec() -> Spec:
    s = Spec(name="t", arch="a3")
    s.space("gm")
    s.space("ub", capacity=192 * 1024, align=32)
    s.pipe("PIPE_V", "PIPE_MTE2")
    s.event("EVENT_ID0")
    s.op(
        "hivm.hir.vadd",
        params={"a": In(), "b": In(), "out": Out()},
        effects=(wr("out"),),
        pipe="PIPE_V",
        value="elementwise(add, a, b, into=out)",
    )
    s.op(
        "hivm.hir.set_flag",
        params={"sp": A("pipe"), "wp": A("pipe"), "ev": A("event")},
        effects=(sync_set(event="ev", from_pipe="sp", to_pipe="wp"),),
    )
    s.check("ub_occupancy", spaces=["ub"])
    return s


# ---------------------------------------------------------------------------
# FR8：确定性
# ---------------------------------------------------------------------------


def test_two_generations_are_byte_identical() -> None:
    a = generate(_spec(), timestamp=FIXED_TS)
    b = generate(_spec(), timestamp=FIXED_TS)
    assert a.config_bytes == b.config_bytes, "FR8：配置文档必须逐字节一致"
    assert a.spec_hash == b.spec_hash


def test_written_config_is_the_canonical_bytes_not_a_reserialization() -> None:
    """配置文档必须原样写出 canonical_bytes。

    若写盘时重新序列化，FR8 的逐字节一致性就依赖两处独立的序列化配置保持
    同步——那是必然漂移的双源真理。
    """
    result = generate(_spec(), timestamp=FIXED_TS)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "config.json"
        write_outputs(result, out)
        assert out.read_bytes() == result.config_bytes


def test_ledger_timestamp_is_injectable_for_reproducibility() -> None:
    r = generate(_spec(), timestamp=FIXED_TS)
    assert r.ledger.generated_at == FIXED_TS


# ---------------------------------------------------------------------------
# schema 校验（T0.3）
# ---------------------------------------------------------------------------


def test_generated_config_satisfies_schema() -> None:
    result = generate(_spec(), timestamp=FIXED_TS)
    errors = validate_config(result.config)
    assert errors == [] or errors == ["jsonschema 不可用（环境问题，非配置错误）"]


def test_schema_rejects_unknown_top_level_key() -> None:
    result = generate(_spec(), timestamp=FIXED_TS)
    bad = dict(result.config)
    bad["surprise"] = 1
    errors = validate_config(bad)
    if errors == ["jsonschema 不可用（环境问题，非配置错误）"]:
        pytest.skip("jsonschema 不可用")
    assert errors, "schema 必须拒绝未知顶层字段（additionalProperties: false）"


def test_schema_rejects_bad_trust_value() -> None:
    result = generate(_spec(), timestamp=FIXED_TS)
    bad = json.loads(json.dumps(result.config))
    bad["ops"][0]["trust"] = "totally-trusted"
    errors = validate_config(bad)
    if errors == ["jsonschema 不可用（环境问题，非配置错误）"]:
        pytest.skip("jsonschema 不可用")
    assert errors


def test_schema_rejects_zero_capacity() -> None:
    result = generate(_spec(), timestamp=FIXED_TS)
    bad = json.loads(json.dumps(result.config))
    for sp in bad["vm"]["spaces"]:
        sp["capacity"] = 0
    errors = validate_config(bad)
    if errors == ["jsonschema 不可用（环境问题，非配置错误）"]:
        pytest.skip("jsonschema 不可用")
    assert errors


# ---------------------------------------------------------------------------
# 信任账本与漂移（T0.4 / D9）
# ---------------------------------------------------------------------------


def test_ledger_records_trust_per_op() -> None:
    r = generate(_spec(), timestamp=FIXED_TS)
    assert r.ledger.entries["hivm.hir.vadd"] == "provisional"
    assert r.ledger.engine_version == ENGINE_VERSION


def test_escape_hatch_is_counted_in_coverage() -> None:
    s = _spec()
    s.op(
        "hivm.hir.mmadL1",
        params={"c": Out()},
        effects=(wr("c"),),
        value=host_fn("mmad_ref", reason="布局代数"),
    )
    r = generate(s, timestamp=FIXED_TS)
    assert "hivm.hir.mmadL1" in r.ledger.escape_hatches
    assert r.ledger.to_json()["coverage"]["escape_hatch_count"] == 1


def test_open_drift_freezes_trust_upgrade() -> None:
    """D9：漂移登记即冻结该 op 的 trust 升级。"""
    ledger = TrustLedger(spec_name="t", spec_hash="sha256:x")
    ledger.entries["hivm.hir.vadd"] = "provisional"
    ledger.drift.append(
        DriftEntry(
            op="hivm.hir.vadd",
            observed="C++ 链显示额外的 pipe 依赖",
            described="仅 PIPE_V",
            status="open",
        )
    )
    assert ledger.frozen_ops() == ("hivm.hir.vadd",)
    assert ledger.to_json()["frozen_by_drift"] == ["hivm.hir.vadd"]


def test_resolved_drift_does_not_freeze() -> None:
    ledger = TrustLedger(spec_name="t", spec_hash="sha256:x")
    ledger.drift.append(
        DriftEntry(op="hivm.hir.vadd", observed="x", described="y", status="description-fixed")
    )
    assert ledger.frozen_ops() == ()


def test_ledger_max_trust_reflects_highest_entry() -> None:
    ledger = TrustLedger(spec_name="t", spec_hash="sha256:x")
    assert ledger.max_trust() == "provisional"
    ledger.entries["a"] = "provisional"
    ledger.entries["b"] = "cross-validated"
    assert ledger.max_trust() == "cross-validated"


def test_ledger_json_is_sorted_for_stable_diffs() -> None:
    ledger = TrustLedger(spec_name="t", spec_hash="sha256:x")
    ledger.entries.update({"z": "provisional", "a": "provisional"})
    assert list(ledger.to_json()["trust"]) == ["a", "z"]


# ---------------------------------------------------------------------------
# CLI（T0.5）
# ---------------------------------------------------------------------------


def test_cli_check_passes_on_toy(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(TOY)]) == EXIT_OK
    assert "静态检查通过" in capsys.readouterr().out


def test_cli_gen_produces_config_and_ledger(tmp_path: Path) -> None:
    out = tmp_path / "config.json"
    rc = main(["gen", str(TOY), "-o", str(out), "--timestamp", FIXED_TS])
    assert rc == EXIT_OK
    assert out.is_file()
    ledger = tmp_path / "config.ledger.json"
    assert ledger.is_file()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["spec_name"] == "hivm_toy"
    # equivalence 于 M3/T3.3 加入（带 rtol/atol/round_mode，进 spec_hash）；
    # sync_pairing 于 M4/T4.4 加入（账本级配对检查，与 timeline 互补）
    assert {c["name"] for c in doc["checks"]} == {
        "ub_occupancy",
        "timeline",
        "equivalence",
        "sync_pairing",
        "uninit_read",
    }


def test_cli_gen_is_byte_reproducible(tmp_path: Path) -> None:
    """M0 退出标准之一。"""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    main(["gen", str(TOY), "-o", str(a), "--timestamp", FIXED_TS])
    main(["gen", str(TOY), "-o", str(b), "--timestamp", FIXED_TS])
    assert a.read_bytes() == b.read_bytes()


def test_cli_gen_refuses_to_generate_when_static_check_fails(tmp_path: Path) -> None:
    """宁可不产出工具，也不产出语义有洞的工具（FR6）。"""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from hivm_spec.spec import In, Out, Spec, wr\n"
        "spec = Spec(name='bad', arch='a3')\n"
        "spec.space('ub', capacity=1024)\n"
        "spec.op('hivm.hir.vadd', params={'a': In(), 'out': Out()}, effects=(wr('ghost'),))\n",
        encoding="utf-8",
    )
    out = tmp_path / "nope.json"
    assert main(["gen", str(bad), "-o", str(out)]) == EXIT_FAIL
    assert not out.exists(), "静态检查失败时不得留下配置文档"


def test_cli_reports_missing_description_file(tmp_path: Path) -> None:
    assert main(["check", str(tmp_path / "nope.py")]) == EXIT_FAIL


def test_cli_reports_description_without_spec_object(tmp_path: Path) -> None:
    p = tmp_path / "empty.py"
    p.write_text("x = 1\n", encoding="utf-8")
    assert main(["check", str(p)]) == EXIT_FAIL


def test_cli_reports_broken_description_readably(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "broken.py"
    p.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    assert main(["check", str(p)]) == EXIT_FAIL
    assert "RuntimeError" in capsys.readouterr().err


def test_cli_unknown_tool_reports_pending_not_fake_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """契约已定义但实现未落地的工具必须显式 PENDING，不得假成功（FR7）。

    T3.5 起 equivalence 已实现，_IMPLEMENTED_TOOLS 三个工具齐备，故这里用一个
    虚构工具名来守住 PENDING 通路本身——它服务于将来新增的工具。
    """
    assert main(["tool", "not_a_tool_yet", "x.mlir"]) == EXIT_PENDING
    assert "PENDING" in capsys.readouterr().err


@pytest.mark.parametrize("tool", ["timeline", "ub_occupancy", "equivalence"])
def test_cli_implemented_tools_do_not_report_pending(
    tool: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """已实现的工具不得停留在 PENDING。

    equivalence 于 T3.5 加入本清单——这是防它退回 PENDING 的回归守卫。
    """
    # x.mlir 不存在 → 走"IR 文件不存在"路径（exit 1），但绝不能是 PENDING(3)
    assert main(["tool", tool, "x.mlir"]) == EXIT_FAIL
    assert "PENDING" not in capsys.readouterr().err


def test_cli_tool_reports_missing_config_actionably(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """缺配置文档时要给出可执行的下一步，而非只说"失败"。"""
    assert main(["tool", "ub_occupancy", "-c", "/nonexistent.json", "x.mlir"]) == EXIT_FAIL
    assert "hivm-spec gen" in capsys.readouterr().err


def test_cli_tool_enforces_single_ir_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D12：占用工具恒吃一份 IR。"""
    cfg = tmp_path / "c.json"
    main(["gen", str(TOY), "-o", str(cfg), "--timestamp", FIXED_TS])
    rc = main(["tool", "ub_occupancy", "-c", str(cfg), "a.mlir", "b.mlir"])
    assert rc == EXIT_FAIL
    assert "恒吃一份 IR" in capsys.readouterr().err


def test_toy_spec_is_importable_and_clean() -> None:
    """toy 描述是 M1 的输入，必须始终无静态错误。"""
    from hivm_spec.static_check import check_spec, has_errors

    sys.path.insert(0, str(REPO_ROOT / "specs"))
    try:
        import importlib.util

        spec_obj = importlib.util.spec_from_file_location("toy", TOY)
        assert spec_obj and spec_obj.loader
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)
        assert not has_errors(check_spec(mod.spec))
        assert mod.spec.arch == "a3"
    finally:
        sys.path.remove(str(REPO_ROOT / "specs"))


def test_toy_declares_sync_ops_so_timeline_check_is_meaningful() -> None:
    """toy 含 timeline check，故必须有同步 op，否则是假阴性温床。"""
    import importlib.util

    spec_obj = importlib.util.spec_from_file_location("toy2", TOY)
    assert spec_obj and spec_obj.loader
    mod = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)
    sync_ops = [o for o in mod.spec.ops if any(e.is_sync for e in o.effects)]
    assert len(sync_ops) >= 2, "至少需 set/wait 一对"


def test_trust_starts_at_provisional_everywhere() -> None:
    """未与 C++ 链对拍前，一律 provisional（D4/OD4）。"""
    import importlib.util

    spec_obj = importlib.util.spec_from_file_location("toy3", TOY)
    assert spec_obj and spec_obj.loader
    mod = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)
    assert all(o.effective_trust is Trust.PROVISIONAL for o in mod.spec.ops)
