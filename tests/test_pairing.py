"""T4.4 同步静态配对检查测试。

验收核心（M4 卡 §3 T4.4）：注入用例检出。
最要紧的纪律是 §4.6：本检查与 timeline **互补**，且**不得声称能判死锁**——
计数平衡不蕴含无死锁（真实案例 CreatePreload stage-major 即计数平衡却死锁）。
"""

from __future__ import annotations

import pytest

from hivm_spec.assemble import run_sync_pairing
from hivm_spec.pairing import analyze_pairing
from hivm_spec.verdict import Verdict
from hivm_spec.vir import Coverage, Loc, SyncKind, VModule, VNode, VRegion, VSync

LOC = Loc(file="t.mlir", line=7)


def _cfg(with_check: bool = True) -> dict:
    cfg: dict = {"ops": [], "vm": {"pipes": ["PIPE_MTE2", "PIPE_V"]}, "checks": []}
    if with_check:
        cfg["checks"].append({"name": "sync_pairing", "options": {}})
    return cfg


def _module(syncs: tuple[VSync, ...]) -> VModule:
    """按 syncs 反推出对应的 VNode。

    VIR 不变量要求每个 VSync.node_id 对应真实区域内的 op——否则时序判定会建立
    在幻影节点上。测试夹具必须尊重这条，不能绕过。
    """
    kind_to_op = {
        SyncKind.SET_FLAG: "hivm.hir.set_flag",
        SyncKind.WAIT_FLAG: "hivm.hir.wait_flag",
        SyncKind.SYNC_BLOCK_SET: "hivm.hir.sync_block_set",
        SyncKind.SYNC_BLOCK_WAIT: "hivm.hir.sync_block_wait",
        SyncKind.PIPE_BARRIER: "hivm.hir.pipe_barrier",
    }
    nodes = tuple(VNode(id=s.node_id, op=kind_to_op[s.kind], loc=LOC, operands=()) for s in syncs)
    return VModule(
        source="t.mlir",
        items=(VRegion(id="f0", kind="func", loc=LOC, items=nodes),),
        syncs=syncs,
        coverage=Coverage(),
        arch="a3",
    )


def _sync(
    node_id: str,
    kind: SyncKind,
    event_id: int | None = 0,
    pipe: str = "PIPE_V",
    implicit: bool = False,
) -> VSync:
    return VSync(
        node_id=node_id, kind=kind, loc=LOC, event_id=event_id, pipe=pipe, implicit=implicit
    )


# ---------------------------------------------------------------------------
# 基本配对
# ---------------------------------------------------------------------------


def test_balanced_pair_is_clean() -> None:
    report = analyze_pairing(
        _module((_sync("s", SyncKind.SET_FLAG), _sync("w", SyncKind.WAIT_FLAG)))
    )
    assert not report.has_error
    assert [f.rule for f in report.findings] == []
    assert len(report.ledgers) == 1
    assert report.ledgers[0].balanced


def test_wait_without_any_set_is_an_error() -> None:
    """wait 全无 set → 必然等不到，这是确定的问题而非"可疑"。"""
    report = analyze_pairing(_module((_sync("w", SyncKind.WAIT_FLAG),)))
    assert report.has_error
    assert [f.rule for f in report.findings] == ["pairing/unpaired-wait"]
    assert report.findings[0].severity == "error"
    assert report.findings[0].loc is not None, "必须可定位（FR5）"


def test_set_without_any_wait_is_a_warning_not_an_error() -> None:
    """set 无人 wait 值得报，但**不是** error。

    它可能是"配对的 wait 被误删"（真缺陷），也可能是冗余同步（性能问题）。
    报成 error 会让大量合法 IR 变红，淹没真问题（NFR2）。
    """
    report = analyze_pairing(_module((_sync("s", SyncKind.SET_FLAG),)))
    assert not report.has_error
    assert [f.rule for f in report.findings] == ["pairing/orphan-set"]
    assert report.findings[0].severity == "warning"


def test_count_imbalance_is_reported_with_caveat() -> None:
    """供需不平要报，但必须附上"平衡不蕴含无死锁"的限定语。"""
    report = analyze_pairing(
        _module(
            (
                _sync("s", SyncKind.SET_FLAG),
                _sync("w1", SyncKind.WAIT_FLAG),
                _sync("w2", SyncKind.WAIT_FLAG),
            )
        )
    )
    assert [f.rule for f in report.findings] == ["pairing/count-imbalance"]
    assert "timeline" in report.findings[0].message, "须指引读者去看真正能判死锁的工具"


def test_multiple_events_tracked_independently() -> None:
    report = analyze_pairing(
        _module(
            (
                _sync("s0", SyncKind.SET_FLAG, event_id=0),
                _sync("w0", SyncKind.WAIT_FLAG, event_id=0),
                _sync("w1", SyncKind.WAIT_FLAG, event_id=1),
            )
        )
    )
    assert len(report.ledgers) == 2
    assert [f.rule for f in report.findings] == ["pairing/unpaired-wait"]
    assert report.findings[0].event_id == 1


def test_sync_block_kinds_participate_in_pairing() -> None:
    """跨核同步（sync_block_set/wait）与 flag 同样参与配对。"""
    report = analyze_pairing(
        _module(
            (
                _sync("s", SyncKind.SYNC_BLOCK_SET),
                _sync("w", SyncKind.SYNC_BLOCK_WAIT),
            )
        )
    )
    assert report.findings == ()


def test_pipe_barrier_is_excluded_from_event_pairing() -> None:
    """barrier 不按 event id 配对，不应被算成孤儿。"""
    report = analyze_pairing(_module((_sync("b", SyncKind.PIPE_BARRIER, event_id=None),)))
    assert report.findings == ()
    assert report.ledgers == ()
    assert report.unattributable == 0, "barrier 是有意排除，不算分析缺口"


# ---------------------------------------------------------------------------
# 反自欺：缺口与不完整账目
# ---------------------------------------------------------------------------


def test_sync_without_event_id_counts_as_a_gap() -> None:
    """无 event id 的 sync 无法配对——必须计入缺口，不得静默丢弃。"""
    report = analyze_pairing(_module((_sync("s", SyncKind.SET_FLAG, event_id=None),)))
    assert report.unattributable == 1
    assert any("无 event id" in n for n in report.notes)


def test_implicit_sync_suppresses_judgement() -> None:
    """隐式（macro 内部）同步使账目不完整 → 不判异常，但要说明原因。

    macro 内部的 set/wait 在 IR 里不可见。据不完整的账目下判断，等于把
    "没看全"说成"有问题"或"没问题"——两个方向都是错的。
    """
    report = analyze_pairing(_module((_sync("w", SyncKind.WAIT_FLAG, implicit=True),)))
    assert report.findings == (), "账目不完整时不下判断"
    assert report.implicit_events == 1
    assert any("隐式" in n for n in report.notes)


def test_report_always_carries_the_non_deadlock_caveat() -> None:
    """每份报告都必须声明"账目级检查不蕴含无死锁"。

    这是 M4 卡 §4.6 的硬性要求：本工具与 timeline 结论不一致时以 timeline
    为准，故它绝不能让读者以为"配对 OK = 同步没问题"。
    """
    report = analyze_pairing(
        _module((_sync("s", SyncKind.SET_FLAG), _sync("w", SyncKind.WAIT_FLAG)))
    )
    assert any("不蕴含无死锁" in n for n in report.notes)


# ---------------------------------------------------------------------------
# 工具装配层
# ---------------------------------------------------------------------------


def test_tool_reports_mismatch_for_unpaired_wait() -> None:
    result = run_sync_pairing(_cfg(), _module((_sync("w", SyncKind.WAIT_FLAG),)), "sha256:x")
    assert result.verdict is Verdict.MISMATCH
    assert result.exit_code == 1
    assert any(f.rule == "pairing/unpaired-wait" for f in result.diagnostics)


def test_tool_reports_ok_with_warning_for_orphan_set() -> None:
    """orphan-set 是 warning，不应让 verdict 变成失败。"""
    result = run_sync_pairing(_cfg(), _module((_sync("s", SyncKind.SET_FLAG),)), "sha256:x")
    assert result.verdict is Verdict.OK
    assert result.exit_code == 0
    assert any(f.rule == "pairing/orphan-set" for f in result.diagnostics)


def test_tool_reports_gap_when_no_events_exist() -> None:
    """一个事件都没有 → 本检查未实际发生，报缺口而非 OK。"""
    result = run_sync_pairing(_cfg(), _module(()), "sha256:x")
    assert result.verdict is Verdict.COVERAGE_GAP
    assert result.exit_code == 4
    assert any(f.rule == "pairing/vacuous" for f in result.diagnostics)


def test_tool_requires_declared_check() -> None:
    """描述未声明 sync_pairing → UNTRUSTED_DESCRIPTION。"""
    result = run_sync_pairing(
        _cfg(with_check=False), _module((_sync("w", SyncKind.WAIT_FLAG),)), "sha256:x"
    )
    assert result.verdict is Verdict.UNTRUSTED_DESCRIPTION


def test_tool_never_emits_deadlock_verdict() -> None:
    """本工具**不得**产出 DEADLOCK——死锁判定归 timeline（M4 卡 §4.6）。

    账目异常在语义上是"可疑"，不是"必然死锁"：真实案例里顺序不可行时计数
    照样平衡。若本工具报 DEADLOCK，读者会以为它做了交错分析。
    """
    cases = (
        (_sync("w", SyncKind.WAIT_FLAG),),
        (_sync("s", SyncKind.SET_FLAG),),
        (_sync("s", SyncKind.SET_FLAG), _sync("w", SyncKind.WAIT_FLAG)),
    )
    for syncs in cases:
        result = run_sync_pairing(_cfg(), _module(syncs), "sha256:x")
        assert result.verdict is not Verdict.DEADLOCK


def test_details_expose_the_ledger() -> None:
    """账目须结构化输出，供 agent 直接消费（FR5）。"""
    result = run_sync_pairing(
        _cfg(),
        _module((_sync("s", SyncKind.SET_FLAG), _sync("w", SyncKind.WAIT_FLAG))),
        "sha256:x",
    )
    events = result.details["events"]
    assert len(events) == 1
    assert events[0]["sets"] == 1 and events[0]["waits"] == 1
    assert events[0]["balanced"] is True


# ---------------------------------------------------------------------------
# 与 timeline 的互补性（M4 卡 §4.6 的核心主张）
# ---------------------------------------------------------------------------


@pytest.mark.requires_bindings
def test_catches_what_timeline_structurally_cannot() -> None:
    """**互补性实证**：set 无人 wait 时 timeline 报 OK，本工具报出 warning。

    timeline 判的是"能否调度通"——一个没人等的 set 不影响可调度性，所以它
    结构上就看不见这个缺陷。若二者结论总是一致，本工具就没有存在价值。
    """
    import importlib.util
    import json as _json
    from pathlib import Path

    from hivm_spec.assemble import run_tool
    from hivm_spec.generate import generate
    from hivm_spec.ir_engine import lower_module_text

    root = Path(__file__).resolve().parents[1]
    loader = importlib.util.spec_from_file_location("toy_p", root / "specs" / "toy.py")
    assert loader and loader.loader
    mod = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod)
    cfg = _json.loads(generate(mod.spec, timestamp="2026-01-01T00:00:00+00:00").config_bytes)
    modeled = {o["op"] for o in cfg["ops"]}
    pipes = {o["op"]: o.get("pipe", "") for o in cfg["ops"]}

    src = (root / "specs" / "cases" / "corpus" / "l0" / "cross_iter_event_pair.mlir").read_text()
    # 删掉 wait：留下一个没人等的 set
    broken = "\n".join(ln for ln in src.splitlines() if "wait_flag" not in ln)
    assert broken != src, "注入前提：语料里确实有 wait_flag"

    module = lower_module_text(broken, modeled, source="b.mlir", op_pipes=pipes).module

    tl = run_tool("timeline", cfg, module, "sha256:x")
    pairing = run_tool("sync_pairing", cfg, module, "sha256:x")

    assert tl.verdict is Verdict.OK, "前提：timeline 看不见这个缺陷"
    assert any(f.rule == "pairing/orphan-set" for f in pairing.diagnostics), (
        "本工具必须报出 timeline 看不见的 orphan-set"
    )
