"""占用引擎与结论契约测试（T1.2–T1.6）。

全部用**手写 VIR**，不需 bindings——D6/D7 要求核心层可独立测试。
"""

from __future__ import annotations

import json

import pytest

from hivm_spec.assemble import config_spec_hash, run_tool, run_ub_occupancy
from hivm_spec.occupancy import Interval, analyze_occupancy
from hivm_spec.verdict import Finding, ToolResult, Verdict, verdict_exit_code
from hivm_spec.vir import (
    Access,
    Coverage,
    Effect,
    Gap,
    GapKind,
    Loc,
    SizeOrigin,
    VAlloc,
    VLoop,
    VModule,
    VNode,
    VRegion,
)

LOC = Loc(file="t.mlir", line=1)
UB = 192 * 1024


def _alloc(name: str, nbytes: int | None, value: str, space: str = "ub") -> VAlloc:
    return VAlloc(
        name=name,
        space=space,
        loc=LOC,
        nbytes=nbytes,
        size_origin=SizeOrigin.STATIC_SHAPE if nbytes else SizeOrigin.UNKNOWN,
        shape_text="x" if nbytes else "?xf32",
        value=value,
    )


def _node(nid: str, *values: str, op: str = "hivm.hir.vadd") -> VNode:
    return VNode(
        id=nid,
        op=op,
        loc=LOC,
        operands=values,
        effects=(Effect(access=Access.WRITE, space="ub"),),
    )


def _module(
    nodes: tuple[VNode, ...],
    allocs: tuple[VAlloc, ...],
    gaps: tuple[Gap, ...] = (),
) -> VModule:
    return VModule(
        source="t.mlir",
        items=(VRegion(id="r0", kind="func", loc=LOC, items=nodes),),
        allocs=allocs,
        coverage=Coverage(gaps=gaps, modeled_ops=("hivm.hir.vadd",)),
        arch="a3",
        engine_version="0.1.0",
    )


# ---------------------------------------------------------------------------
# 生存期：区间语义（本引擎最容易写错的地方）
# ---------------------------------------------------------------------------


def test_disjoint_lifetimes_do_not_sum() -> None:
    """生存期不重叠的两个 buffer 不得被累加。

    若按 alloc 时刻简单累加，两个各 128KB 的 buffer 会得出 256KB 的
    **虚假溢出**。这是占用分析最典型的错误。
    """
    a, b = _alloc("A", 128 * 1024, "%a"), _alloc("B", 128 * 1024, "%b")
    nodes = (_node("n1", "%a"), _node("n2", "%a"), _node("n3", "%b"), _node("n4", "%b"))
    occ = analyze_occupancy(_module(nodes, (a, b)), {"ub": UB})
    so = occ.spaces["ub"]
    assert so.peak_bytes == 128 * 1024, f"峰值应为 128KB，实为 {so.peak_bytes}"
    assert not so.overflows


def test_overlapping_lifetimes_do_sum() -> None:
    """生存期重叠必须累加，否则会**漏报**真实溢出。"""
    a, b = _alloc("A", 128 * 1024, "%a"), _alloc("B", 128 * 1024, "%b")
    nodes = (_node("n1", "%a", "%b"), _node("n2", "%a", "%b"))
    occ = analyze_occupancy(_module(nodes, (a, b)), {"ub": UB})
    so = occ.spaces["ub"]
    assert so.peak_bytes == 256 * 1024
    assert so.overflows


def test_buffer_alive_across_loop_is_counted() -> None:
    """跨循环存活的 buffer 必须计入——漏掉它会低估峰值。"""
    a, b = _alloc("A", 64 * 1024, "%a"), _alloc("B", 64 * 1024, "%b")
    loop_body = VRegion(
        id="r1",
        kind="for",
        loc=LOC,
        items=(_node("in_loop", "%b"),),
        loop=VLoop(iv="i", trip_count=4),
    )
    func = VRegion(
        id="r0",
        kind="func",
        loc=LOC,
        items=(_node("before", "%a"), loop_body, _node("after", "%a")),
    )
    m = VModule(
        source="t.mlir",
        items=(func,),
        allocs=(a, b),
        coverage=Coverage(modeled_ops=("hivm.hir.vadd",)),
    )
    occ = analyze_occupancy(m, {"ub": UB})
    so = occ.spaces["ub"]
    # A 活到 after（跨越循环），B 只在循环体内 → 峰值 = A + B
    assert so.peak_bytes == 128 * 1024


def test_unsized_buffer_is_excluded_but_reported() -> None:
    """尺寸未知的 buffer 不计入峰值，但必须登记缺口。

    关键：峰值因此只是**下界**，不足以断言不溢出。
    """
    known = _alloc("A", 1024, "%a")
    unknown = _alloc("B", None, "%b")
    nodes = (_node("n1", "%a"), _node("n2", "%b"))
    occ = analyze_occupancy(_module(nodes, (known, unknown)), {"ub": UB})
    so = occ.spaces["ub"]
    assert so.peak_bytes == 1024
    assert len(so.unsized) == 1
    assert any(k is GapKind.UNKNOWN_SIZE for k, _, _ in occ.gap_notes)
    assert any("下界" in d for _, d, _ in occ.gap_notes)


def test_never_used_buffer_is_conservatively_kept_alive() -> None:
    """从未被访问的 buffer 保守按"活到最后"处理。

    宁可高估峰值（误报溢出，人能看出来），不可低估（漏报溢出，静默放过）。
    """
    orphan = _alloc("ORPHAN", 1024, "%never")
    used = _alloc("U", 1024, "%u")
    nodes = (_node("n1", "%u"), _node("n2", "%u"), _node("n3", "%u"))
    occ = analyze_occupancy(_module(nodes, (orphan, used)), {"ub": UB})
    so = occ.spaces["ub"]
    orphan_iv = next(i for i in so.intervals if i.alloc.name == "ORPHAN")
    assert orphan_iv.never_used
    assert orphan_iv.end == len(list(nodes)) - 1, "应活到模块末尾"
    assert so.peak_bytes == 2048
    assert any("死代码" in d for _, d, _ in occ.gap_notes)


def test_peak_contributors_sorted_by_size() -> None:
    """溢出诊断需按尺寸降序给贡献者（FR5 可操作性）。"""
    a = _alloc("small", 1024, "%a")
    b = _alloc("big", 100 * 1024, "%b")
    c = _alloc("mid", 8 * 1024, "%c")
    nodes = (_node("n1", "%a", "%b", "%c"),)
    occ = analyze_occupancy(_module(nodes, (a, b, c)), {"ub": UB})
    names = [iv.alloc.name for iv in occ.spaces["ub"].peak_contributors]
    assert names == ["big", "mid", "small"]


def test_per_space_isolation() -> None:
    """不同地址空间独立统计——混算会产生毫无意义的结论。"""
    ub = _alloc("U", 100 * 1024, "%u", space="ub")
    cbuf = _alloc("C", 400 * 1024, "%c", space="cbuf")
    nodes = (_node("n1", "%u", "%c"),)
    occ = analyze_occupancy(_module(nodes, (ub, cbuf)), {"ub": UB, "cbuf": 512 * 1024})
    assert occ.spaces["ub"].peak_bytes == 100 * 1024
    assert occ.spaces["cbuf"].peak_bytes == 400 * 1024
    assert not occ.any_overflow


def test_overflow_reports_which_space() -> None:
    ub = _alloc("U", 300 * 1024, "%u", space="ub")
    nodes = (_node("n1", "%u"),)
    occ = analyze_occupancy(_module(nodes, (ub,)), {"ub": UB})
    assert occ.overflowing_spaces == ("ub",)
    assert occ.spaces["ub"].headroom == UB - 300 * 1024
    assert occ.spaces["ub"].headroom < 0


def test_curve_has_one_point_per_node() -> None:
    a = _alloc("A", 1024, "%a")
    nodes = (_node("n1", "%a"), _node("n2", "%a"), _node("n3", "%a"))
    occ = analyze_occupancy(_module(nodes, (a,)), {"ub": UB})
    assert len(occ.spaces["ub"].curve) == 3


def test_interval_contains() -> None:
    iv = Interval(alloc=_alloc("A", 8, "%a"), start=2, end=5)
    assert not iv.contains(1)
    assert iv.contains(2) and iv.contains(5)
    assert not iv.contains(6)


def test_empty_module_does_not_crash() -> None:
    m = _module((), ())
    occ = analyze_occupancy(m, {"ub": UB})
    assert occ.spaces == {}
    assert not occ.any_overflow


# ---------------------------------------------------------------------------
# verdict 契约
# ---------------------------------------------------------------------------


def test_gap_verdicts_are_not_problems() -> None:
    """缺口类 verdict 说的是"我没验成"，不是"发现了问题"。

    混淆二者会让 agent 把"看不懂"当成"没问题"。
    """
    assert Verdict.COVERAGE_GAP.is_gap
    assert not Verdict.COVERAGE_GAP.is_problem
    assert Verdict.UNTRUSTED_DESCRIPTION.is_gap
    assert Verdict.OVERFLOW.is_problem
    assert not Verdict.OVERFLOW.is_gap
    assert not Verdict.OK.is_problem and not Verdict.OK.is_gap


def test_gap_and_problem_have_different_exit_codes() -> None:
    """脚本必须能区分"验证失败"与"没验成"。"""
    assert verdict_exit_code(Verdict.OK) == 0
    assert verdict_exit_code(Verdict.OVERFLOW) == 1
    assert verdict_exit_code(Verdict.COVERAGE_GAP) == 4
    assert verdict_exit_code(Verdict.UNTRUSTED_DESCRIPTION) == 5
    assert verdict_exit_code(Verdict.COVERAGE_GAP) != verdict_exit_code(Verdict.OVERFLOW)


def test_every_verdict_has_an_exit_code() -> None:
    for v in Verdict:
        assert isinstance(verdict_exit_code(v), int)


def test_result_json_is_deterministic() -> None:
    r = ToolResult(
        tool="ub_occupancy",
        verdict=Verdict.OK,
        spec_hash="sha256:x",
        engine_version="0.1.0",
        trust="provisional",
        ir_fingerprint="abc",
    )
    assert r.to_json_bytes() == r.to_json_bytes()
    doc = json.loads(r.to_json_bytes())
    assert doc["verdict"] == "OK"


def test_untrusted_description_is_flagged_in_render() -> None:
    """信任降级必须出现在结论旁——否则"图画出来了"就显得权威（FR6）。"""
    r = ToolResult(
        tool="ub_occupancy",
        verdict=Verdict.OK,
        spec_hash="sha256:x",
        engine_version="0.1.0",
        trust="provisional",
    )
    text = r.render()
    assert "provisional" in text
    assert "未经对拍验证" in text


def test_anchored_trust_does_not_add_a_warning() -> None:
    r = ToolResult(
        tool="ub_occupancy",
        verdict=Verdict.OK,
        spec_hash="sha256:x",
        engine_version="0.1.0",
        trust="anchored",
    )
    assert "未经对拍验证" not in r.render()


def test_audit_coordinates_are_all_present() -> None:
    """复现历史结论需要 spec_hash + engine_version + IR 指纹三者。"""
    r = ToolResult(
        tool="ub_occupancy",
        verdict=Verdict.OK,
        spec_hash="sha256:abc",
        engine_version="0.1.0",
        trust="provisional",
        ir_fingerprint="deadbeef" * 8,
    )
    doc = json.loads(r.to_json_bytes())
    assert doc["spec_hash"] and doc["engine_version"] and doc["ir_fingerprint"]


def test_finding_renders_location() -> None:
    f = Finding(severity="error", message="boom", loc=Loc(file="k.mlir", line=7), rule="r/x")
    assert "k.mlir:7" in f.render()
    assert "[r/x]" in f.render()


# ---------------------------------------------------------------------------
# 装配：配置文档 + VIR → 结论
# ---------------------------------------------------------------------------

CONFIG = {
    "spec_name": "toy",
    "arch": "a3",
    "schema_version": 1,
    "vm": {"spaces": [{"name": "ub", "capacity": UB, "align": 32}]},
    "ops": [
        {
            "op": "hivm.hir.vadd",
            "trust": "provisional",
            "effects": [{"kind": "write", "space": "ub", "target": "out"}],
        }
    ],
    "checks": [{"name": "ub_occupancy", "params": {"spaces": ["ub"]}}],
}


def test_clean_module_yields_ok() -> None:
    a = _alloc("A", 1024, "%a")
    m = _module((_node("n1", "%a"),), (a,))
    r = run_ub_occupancy(CONFIG, m, "sha256:x")
    assert r.verdict is Verdict.OK
    assert r.exit_code == 0


def test_overflow_wins_over_ok() -> None:
    a = _alloc("A", 300 * 1024, "%a")
    m = _module((_node("n1", "%a"),), (a,))
    r = run_ub_occupancy(CONFIG, m, "sha256:x")
    assert r.verdict is Verdict.OVERFLOW
    assert any("超出容量" in d.message for d in r.diagnostics)
    assert any(d.rule == "occupancy/overflow" for d in r.diagnostics)


def test_unmodeled_op_downgrades_ok_to_coverage_gap() -> None:
    """有缺口时"没发现溢出"必须降级——否则等于用"我没看全"换一个 OK。"""
    a = _alloc("A", 1024, "%a")
    gap = Gap(kind=GapKind.UNMODELED_OP, op="hivm.hir.vexp", detail="未建模", loc=LOC)
    m = _module((_node("n1", "%a"),), (a,), gaps=(gap,))
    r = run_ub_occupancy(CONFIG, m, "sha256:x")
    assert r.verdict is Verdict.COVERAGE_GAP
    assert any("不能" in d.message for d in r.diagnostics)


def test_overflow_is_reported_even_with_gaps() -> None:
    """已确证的溢出不该被"还有别的看不懂"掩盖。"""
    a = _alloc("A", 300 * 1024, "%a")
    gap = Gap(kind=GapKind.UNMODELED_OP, op="hivm.hir.vexp", detail="未建模", loc=LOC)
    m = _module((_node("n1", "%a"),), (a,), gaps=(gap,))
    r = run_ub_occupancy(CONFIG, m, "sha256:x")
    assert r.verdict is Verdict.OVERFLOW


def test_unknown_capacity_cannot_yield_ok() -> None:
    """容量未知时不得给 OK——那是拿"不知道"换"没问题"。"""
    cfg = json.loads(json.dumps(CONFIG))
    cfg["vm"]["spaces"][0]["capacity"] = None
    a = _alloc("A", 1024, "%a")
    m = _module((_node("n1", "%a"),), (a,))
    r = run_ub_occupancy(cfg, m, "sha256:x")
    assert r.verdict is Verdict.COVERAGE_GAP


def test_missing_check_declaration_is_untrusted() -> None:
    cfg = json.loads(json.dumps(CONFIG))
    cfg["checks"] = []
    m = _module((_node("n1", "%a"),), (_alloc("A", 1024, "%a"),))
    r = run_ub_occupancy(cfg, m, "sha256:x")
    assert r.verdict is Verdict.UNTRUSTED_DESCRIPTION
    assert r.exit_code == 5


def test_details_carry_curve_and_contributors() -> None:
    a = _alloc("A", 1024, "%a")
    m = _module((_node("n1", "%a"), _node("n2", "%a")), (a,))
    r = run_ub_occupancy(CONFIG, m, "sha256:x")
    ub = r.details["spaces"]["ub"]
    assert ub["curve"] and ub["contributors"]
    assert ub["capacity"] == UB
    assert ub["utilization"] is not None


def test_run_tool_rejects_unknown_tool() -> None:
    m = _module((), ())
    with pytest.raises(KeyError, match="timeline"):
        run_tool("no_such_tool", CONFIG, m)


def test_spec_hash_is_computed_from_bytes_not_read_from_doc() -> None:
    """spec_hash 是配置文档的哈希，不可能存在文档内部（自指）。

    由字节串重算同时天然校验了文档未被篡改。
    """
    raw = b'{"a": 1}'
    h1 = config_spec_hash(raw)
    assert h1.startswith("sha256:")
    assert config_spec_hash(raw) == h1
    assert config_spec_hash(b'{"a": 2}') != h1


def test_trust_reflects_highest_declared_level() -> None:
    cfg = json.loads(json.dumps(CONFIG))
    cfg["ops"].append({"op": "hivm.hir.load", "trust": "cross-validated", "effects": []})
    m = _module((_node("n1", "%a"),), (_alloc("A", 1024, "%a"),))
    r = run_ub_occupancy(cfg, m, "sha256:x")
    assert r.trust == "cross-validated"
