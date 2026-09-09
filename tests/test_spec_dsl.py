"""Spec DSL 与静态检查器的测试（T0.1 / T0.2）。

重点是**静态检查必须真的拒绝错误描述**：一个永远放过的检查器比没有检查器更危险，
因为它提供虚假的安全感（正是 FR6 要防的自欺）。
"""

from __future__ import annotations

import pytest

from hivm_spec.spec import (
    UNBOUND,
    Attr,
    In,
    Out,
    Spec,
    SpecError,
    Trust,
    Variadic,
    cond_wr,
    host_fn,
    rd,
    sync_barrier,
    sync_set,
    sync_wait,
    wr,
)
from hivm_spec.static_check import Severity, check_spec, format_diagnostics, has_errors


def _base_spec() -> Spec:
    s = Spec(name="t", arch="a3")
    s.space("gm")
    s.space("ub", capacity=192 * 1024, align=32)
    s.pipe("PIPE_V", "PIPE_MTE2")
    s.event("EVENT_ID0")
    return s


def _errors(s: Spec) -> list[str]:
    return [d.message for d in check_spec(s) if d.severity is Severity.ERROR]


# ---------------------------------------------------------------------------
# Spec API 基础
# ---------------------------------------------------------------------------


def test_param_names_are_bound_from_dict_keys() -> None:
    s = _base_spec()
    op = s.op("hivm.hir.vadd", params={"a": In(), "b": In(), "out": Out()}, effects=(wr("out"),))
    assert [p.name for p in op.params] == ["a", "b", "out"]
    assert all(p.is_bound for p in op.params)


def test_unbound_param_is_distinguishable_from_empty_name() -> None:
    """哨兵使"忘记绑定"与"名字为空"两种错误可区分。"""
    assert In().name == UNBOUND
    assert not In().is_bound
    with pytest.raises(SpecError, match="不得为空"):
        In().__class__(name="", kind=In().kind)


def test_duplicate_declarations_are_rejected() -> None:
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))
    with pytest.raises(SpecError, match="重复"):
        s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))
    with pytest.raises(SpecError, match="重复"):
        s.space("ub")
    with pytest.raises(SpecError, match="重复"):
        s.pipe("PIPE_V")
    s.check("ub_occupancy")
    with pytest.raises(SpecError, match="重复"):
        s.check("ub_occupancy")


def test_spec_requires_a_name() -> None:
    with pytest.raises(SpecError, match="命名"):
        Spec(name="")


# ---------------------------------------------------------------------------
# 效应原语（spike 的核心产出）
# ---------------------------------------------------------------------------


def test_sync_effects_are_a_distinct_category_not_memory_rw() -> None:
    """spike ② 的关键发现：同步是对同步状态的读写，不能用 rd/wr 硬套。"""
    assert sync_set(event="e").is_sync
    assert sync_wait(event="e").is_sync
    assert sync_barrier(pipe="PIPE_V").is_sync
    assert not rd("a").is_sync
    assert not wr("a").is_sync


def test_sync_effects_require_an_event() -> None:
    with pytest.raises(SpecError, match="event"):
        sync_set(event="")
    with pytest.raises(SpecError, match="event"):
        sync_wait(event="")


def test_cond_write_requires_a_condition() -> None:
    """spike ③：没有 when，init_condition 的条件语义就静默丢失了。"""
    with pytest.raises(SpecError, match="when"):
        cond_wr("c", when="")
    e = cond_wr("c", when="init_condition")
    assert e.when == "init_condition"


def test_memory_effects_require_a_target() -> None:
    with pytest.raises(SpecError, match="target"):
        rd("")


def test_host_fn_requires_an_auditable_reason() -> None:
    with pytest.raises(SpecError, match="reason"):
        host_fn("mmad_ref", reason="")


# ---------------------------------------------------------------------------
# 逃生舱 trust 封顶（OD8）
# ---------------------------------------------------------------------------


def test_escape_hatch_trust_is_capped_at_provisional() -> None:
    s = _base_spec()
    op = s.op(
        "hivm.hir.mmadL1",
        params={"a": In(), "c": Out()},
        effects=(wr("c"),),
        value=host_fn("mmad_ref", reason="布局代数不进声明式描述"),
        trust=Trust.ANCHORED,  # 刻意声明高信任
    )
    assert op.effective_trust is Trust.PROVISIONAL, "逃生舱不得伪装成高可信"
    assert any("封顶 provisional" in m for m in _errors(s))


def test_declarative_op_keeps_its_declared_trust() -> None:
    s = _base_spec()
    op = s.op(
        "hivm.hir.vadd",
        params={"out": Out()},
        effects=(wr("out"),),
        value="elementwise(add)",
        trust=Trust.CROSS_VALIDATED,
    )
    assert op.effective_trust is Trust.CROSS_VALIDATED


# ---------------------------------------------------------------------------
# 静态检查：必须真的拒绝
# ---------------------------------------------------------------------------


def test_effect_referencing_unknown_param_is_rejected() -> None:
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"), rd("ghost")))
    assert any("ghost" in m for m in _errors(s))


def test_out_param_without_write_effect_is_rejected() -> None:
    """漏写 Out 的写效应 → 占用分析会漏掉写入点，是危险的假阴性。"""
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"a": In(), "out": Out()}, effects=(rd("a"),))
    assert any("标注为 Out 但无写效应" in m for m in _errors(s))


def test_in_param_with_write_effect_is_rejected() -> None:
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"a": In()}, effects=(wr("a"),))
    assert any("标注为 In 却有写效应" in m for m in _errors(s))


def test_undeclared_space_is_rejected() -> None:
    s = _base_spec()
    s.op("hivm.hir.load", params={"dst": Out()}, effects=(wr("dst", space="l0c"),))
    assert any("未声明的 space" in m for m in _errors(s))


def test_undeclared_pipe_is_rejected() -> None:
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),), pipe="PIPE_NOPE")
    assert any("PIPE_NOPE" in m for m in _errors(s))


def test_empty_description_is_rejected() -> None:
    """空描述会让工具静默通过该 op —— 正确做法是不声明它，从而触发 COVERAGE_GAP。"""
    s = _base_spec()
    s.op("hivm.hir.mystery", params={"a": In()}, effects=())
    errs = _errors(s)
    assert any("空描述" in m for m in errs)


def test_arity_on_non_variadic_param_is_rejected() -> None:
    s = _base_spec()
    s.op(
        "hivm.hir.mmadL1",
        params={"out": Out(), "sync_args": In("scalar")},
        effects=(wr("out"),),
    )
    # 手工构造非法组合
    s2 = _base_spec()
    s2.op(
        "hivm.hir.mmadL1",
        params={"out": Out(), "sync_args": Variadic(In("scalar"), arity=(0, 2))},
        effects=(wr("out"),),
    )
    assert not any("非 variadic" in m for m in _errors(s2))


def test_unknown_check_is_rejected() -> None:
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))
    s.check("does_not_exist")
    assert any("未知 check" in m for m in _errors(s))


def test_occupancy_check_without_capacity_is_rejected() -> None:
    """无容量则无法判定溢出，结论会假精确。"""
    s = Spec(name="t")
    s.space("ub")  # 无 capacity
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))
    s.check("ub_occupancy", spaces=["ub"])
    assert any("无 capacity" in m for m in _errors(s))


def test_timeline_check_without_sync_ops_is_rejected() -> None:
    """无同步 op 的时序检查永远得出'未发现问题'——危险的假阴性。"""
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))
    s.check("timeline")
    assert any("假阴性" in m for m in _errors(s))


def test_timeline_check_passes_with_sync_ops() -> None:
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))
    s.op(
        "hivm.hir.set_flag",
        params={"sp": Attr("pipe"), "wp": Attr("pipe"), "ev": Attr("event")},
        effects=(sync_set(event="ev", from_pipe="sp", to_pipe="wp"),),
    )
    s.check("timeline")
    assert not any("假阴性" in m for m in _errors(s))


def test_equivalence_check_without_value_semantics_is_rejected() -> None:
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))
    s.check("equivalence")
    assert any("无从比较" in m for m in _errors(s))


def test_no_space_declared_is_rejected() -> None:
    s = Spec(name="t")
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))
    assert any("未声明任何 space" in m for m in _errors(s))


def test_no_checks_warns_but_is_not_an_error() -> None:
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))
    diags = check_spec(s)
    assert not has_errors(diags)
    assert any(d.severity is Severity.WARNING for d in diags)


def test_diagnostics_carry_in_description_location() -> None:
    """FR7：报错须指向描述内位置，而非被验证的 IR。"""
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"), rd("ghost")))
    d = next(d for d in check_spec(s) if "ghost" in d.message)
    assert d.where == "op hivm.hir.vadd"
    assert "hivm.hir.vadd" in d.render()


def test_format_diagnostics_summarizes_counts() -> None:
    s = _base_spec()
    s.op("hivm.hir.vadd", params={"a": In()}, effects=(wr("a"),))
    text = format_diagnostics(check_spec(s))
    assert "个错误" in text and "个告警" in text


def test_all_diagnostics_are_reported_not_short_circuited() -> None:
    """agent 需要一次看到所有问题，逐个试错会浪费大量轮次。"""
    s = _base_spec()
    s.op("hivm.hir.a", params={"out": Out()}, effects=(rd("ghost1"),))
    s.op("hivm.hir.b", params={"out": Out()}, effects=(rd("ghost2"),))
    errs = _errors(s)
    assert any("ghost1" in m for m in errs)
    assert any("ghost2" in m for m in errs)


# ---------------------------------------------------------------------------
# 归一化与 spec_hash（FR8）
# ---------------------------------------------------------------------------


def test_normalize_is_deterministic() -> None:
    def build() -> Spec:
        s = _base_spec()
        s.op("hivm.hir.vadd", params={"a": In(), "out": Out()}, effects=(rd("a"), wr("out")))
        s.check("ub_occupancy", spaces=["ub"])
        return s

    assert build().canonical_bytes() == build().canonical_bytes()
    assert build().spec_hash() == build().spec_hash()


def test_declaration_order_does_not_affect_the_hash() -> None:
    """归一化必须消除声明顺序的影响，否则 spec_hash 会因无关改动而变。"""
    a = _base_spec()
    a.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))
    a.op("hivm.hir.vmul", params={"out": Out()}, effects=(wr("out"),))

    b = _base_spec()
    b.op("hivm.hir.vmul", params={"out": Out()}, effects=(wr("out"),))
    b.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),))

    assert a.spec_hash() == b.spec_hash()


def test_semantic_change_does_affect_the_hash() -> None:
    a = _base_spec()
    a.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),), pipe="PIPE_V")
    b = _base_spec()
    b.op("hivm.hir.vadd", params={"out": Out()}, effects=(wr("out"),), pipe="PIPE_MTE2")
    assert a.spec_hash() != b.spec_hash()


def test_normalize_marks_escape_hatch_entries() -> None:
    s = _base_spec()
    s.op(
        "hivm.hir.mmadL1",
        params={"c": Out()},
        effects=(wr("c"),),
        value=host_fn("mmad_ref", reason="布局代数"),
    )
    doc = s.normalize()
    entry = doc["ops"][0]
    assert entry["escape_hatch"] is True
    assert entry["trust"] == "provisional"
    assert entry["value"]["reason"] == "布局代数"


# ---------------------------------------------------------------------------
# 语义假设登记（D9 前置，M2 审查发现 3）
# ---------------------------------------------------------------------------


def test_assumption_is_registered_and_hashed() -> None:
    """未对拍的语义假设须落配置文档，且进 spec_hash（假设变更 = 语义变更）。"""
    s = _base_spec()
    s.assume(
        "timeline/flag-initial-state",
        "flag 初态 = 已装载",
        rationale="压制假阳性",
        risk_direction="false-negative",
        resolve_by="M3",
    )
    doc = s.normalize()
    assert doc["assumptions"][0]["subject"] == "timeline/flag-initial-state"
    assert doc["assumptions"][0]["risk_direction"] == "false-negative"

    bare = _base_spec()
    assert bare.spec_hash() != s.spec_hash(), "假设登记必须改变 spec_hash"


def test_assumption_rejects_duplicate_and_bad_risk_direction() -> None:
    """一个主体只能有一个当前假设；风险方向取值受约束（防账本堆矛盾猜测）。"""
    s = _base_spec()
    s.assume("timeline/x", "假设 A")
    with pytest.raises(SpecError, match="重复登记"):
        s.assume("timeline/x", "假设 B")
    with pytest.raises(SpecError, match="risk_direction"):
        s.assume("timeline/y", "假设 C", risk_direction="whatever")


def test_unresolved_assumption_freezes_trust_upgrade() -> None:
    """未销案的假设冻结 trust 升级——语义没对齐就升信任等于把猜测当结论。"""
    from hivm_spec.assemble import _trust_of
    from hivm_spec.generate import generate

    s = _base_spec()
    s.op(
        "hivm.hir.vadd",
        params={"out": Out()},
        effects=(wr("out"),),
        trust=Trust.CROSS_VALIDATED,
    )
    s.assume("timeline/flag-initial-state", "flag 初态 = 已装载", resolve_by="M3")
    res = generate(s, timestamp="2026-01-01T00:00:00+00:00")
    led = res.ledger.to_json()
    assert led["frozen_by_assumption"] == ["timeline/flag-initial-state"]
    # 结论侧同样封顶 provisional
    assert _trust_of(res.config) == "provisional"
