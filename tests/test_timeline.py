"""时序引擎与死锁判定测试（T2.1–T2.5）。

全部用**手写 VIR**，不需 bindings——D6/D7 要求核心层可独立测试。
口径见 timeline 模块 docstring：泳道=pipe、事件计数语义、两层判定。
"""

from __future__ import annotations

import json
from pathlib import Path

from hivm_spec.assemble import run_timeline
from hivm_spec.render import chrome_trace_json, render_timeline_chart
from hivm_spec.timeline import (
    DEFAULT_BOUND,
    expand_steps,
    simulate,
)
from hivm_spec.verdict import Verdict
from hivm_spec.vir import (
    Coverage,
    Loc,
    SyncKind,
    VLoop,
    VModule,
    VNode,
    VRegion,
    VSync,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LOC = Loc(file="t.mlir", line=1)
PIPES = ("PIPE_V", "PIPE_MTE2")


# ---------------------------------------------------------------------------
# VIR 构造助手（手写微例，M0 惯例）
# ---------------------------------------------------------------------------


def _op(nid: str, pipe: str = "PIPE_V") -> VNode:
    return VNode(id=nid, op="hivm.hir.vadd", loc=LOC, pipe=pipe)


def _sync(nid: str, kind: str, ev: int | None, pipe: str) -> tuple[VNode, VSync]:
    node = VNode(id=nid, op=f"hivm.hir.{kind}", loc=Loc(file="t.mlir", line=len(nid)), pipe=pipe)
    vs = VSync(
        node_id=nid,
        kind=SyncKind(kind),
        loc=node.loc,
        event_id=ev,
        pipe=pipe,
    )
    return node, vs


def _loop(items: tuple, trip: int | None, lid: str = "L") -> VRegion:
    return VRegion(id=lid, kind="for", loc=LOC, items=items, loop=VLoop(iv="i", trip_count=trip))


def _module(items: tuple, syncs: tuple = (), gaps: tuple = ()) -> VModule:
    return VModule(
        source="t.mlir",
        items=(VRegion(id="f0", kind="func", loc=LOC, items=items),),
        syncs=tuple(syncs),
        coverage=Coverage(modeled_ops=("hivm.hir.vadd",), gaps=gaps),
        arch="a3",
    )


def _cfg(bound: int = DEFAULT_BOUND) -> dict:
    return {
        "ops": [{"op": "hivm.hir.vadd", "trust": "provisional"}],
        "vm": {"pipes": list(PIPES)},
        "checks": [{"name": "timeline", "options": {"unroll_bound": bound}}],
    }


def _run(module: VModule, **kw) -> object:
    return run_timeline(_cfg(**{k: v for k, v in kw.items() if k == "bound"}), module, "sha:test")


# ---------------------------------------------------------------------------
# T2.1/T2.3：健康样例与两类结构性死锁
# ---------------------------------------------------------------------------


def test_cross_iter_pair_is_healthy() -> None:
    """跨迭代事件对：set 在 MTE2 泳道、wait 在 V 泳道——流水线健康。

    这是 D11 spike op ② 的最小形态，也是"计数配平且顺序可行"的对照面。
    """
    w, ws = _sync("w0", "wait_flag", 0, "PIPE_V")
    s, ss = _sync("s0", "set_flag", 0, "PIPE_MTE2")
    m = _module((_loop((w, s), 8),), (ws, ss))
    r = _run(m)  # type: ignore[arg-type]
    assert r.verdict is Verdict.OK  # type: ignore[attr-defined]
    claim = r.details["exploration"]["claim"]  # type: ignore[attr-defined]
    assert "未发现死锁" in claim and "展开界" in claim  # T2.5 措辞
    # events 统计是**模块全量**口径（与展开界无关）
    assert r.details["events"] == {"0": {"sets": 1, "waits": 1}}  # type: ignore[attr-defined]
    assert r.exit_code == 0  # type: ignore[attr-defined]


def test_unpaired_wait_is_deterministic_deadlock() -> None:
    """规则 A：事件全模块无人 set → 消费完初始装载后必然饿死（waits_before_sets 形态）。

    set/wait 计数"配平"（对各自 flag 而言）不等于顺序可行——这正是
    FileCheck 必然漏过、本工具必须抓出的立论案例之一。re-arm 模型下首个
    wait 消费初始装载后放行，第二个 wait 无 arm 可用 → 死锁。
    """
    w1, ws1 = _sync("w1", "sync_block_wait", 15, "PIPE_S")
    w2, ws2 = _sync("w2", "sync_block_wait", 15, "PIPE_S")
    s1, ss1 = _sync("s1", "sync_block_set", 14, "PIPE_S")
    m = _module((w1, w2, s1), (ws1, ws2, ss1))
    r = _run(m)  # type: ignore[arg-type]
    assert r.verdict is Verdict.DEADLOCK  # type: ignore[attr-defined]
    assert r.exit_code == 1  # type: ignore[attr-defined]
    rule_findings = [d for d in r.diagnostics if d.rule == "timeline/unpaired-wait"]  # type: ignore[attr-defined]
    assert len(rule_findings) == 1  # 首个 wait 消费初始装载后放行
    assert rule_findings[0].loc == w2.loc  # FR5：可定位
    assert r.details["deadlocks"][0]["rule"] == "unpaired-wait"  # type: ignore[attr-defined]


def test_same_lane_wait_outpaces_rearm_is_deadlock() -> None:
    """规则 B：同泳道 2 wait : 1 set（wait 消费速度超过 re-arm）→ wait-for 环。

    跨迭代流水线 re-arm 不足的注入形态：16 次 wait 只有 8 次 set + 1 初始
    装载，第 2 次 wait 起无 arm 可用；泳道程序序门控使 set 不可达。
    静态 trip 全量展开（无截断），确定性判定成立。
    """
    w1, ws1 = _sync("w1", "wait_flag", 0, "PIPE_V")
    w2, ws2 = _sync("w2", "wait_flag", 0, "PIPE_V")
    s, ss = _sync("s", "set_flag", 0, "PIPE_V")
    m = _module((_loop((w1, w2, s), 8),), (ws1, ws2, ss))
    r = _run(m)  # type: ignore[arg-type]
    assert r.verdict is Verdict.DEADLOCK  # type: ignore[attr-defined]
    assert any(d.rule == "timeline/wait-cycle" for d in r.diagnostics)  # type: ignore[attr-defined]


def test_same_lane_balanced_pair_with_rearm_is_healthy() -> None:
    """对照：同泳道 1 wait : 1 set + 初始装载 = 自续握手 → 健康（不误判）。

    这是 re-arm 口径的关键回归面：若初态建模为空，此形态会被误判死锁
    （假阳性）——NFR2 优先压制此类误判。
    """
    w, ws = _sync("w", "wait_flag", 0, "PIPE_V")
    s, ss = _sync("s", "set_flag", 0, "PIPE_V")
    m = _module((_loop((w, s), 8),), (ws, ss))
    r = _run(m)  # type: ignore[arg-type]
    assert r.verdict is Verdict.OK  # type: ignore[attr-defined]


def test_truncation_prevents_false_cycle_deadlock() -> None:
    """截断保护：同一 re-arm 饥饿形态但 trip 未知 → 不得判 DEADLOCK。

    截断可能人为切断 set 的可达性——误判（把对的判错）优先压制（NFR2），
    降级为探索层观察 + 截断限定语。
    """
    w1, ws1 = _sync("w1", "wait_flag", 0, "PIPE_V")
    w2, ws2 = _sync("w2", "wait_flag", 0, "PIPE_V")
    s, ss = _sync("s", "set_flag", 0, "PIPE_V")
    m = _module((_loop((w1, w2, s), None),), (ws1, ws2, ss))
    r = _run(m, bound=4)  # type: ignore[arg-type]
    assert r.verdict is not Verdict.DEADLOCK  # type: ignore[attr-defined]
    assert r.details["truncated"] is True  # type: ignore[attr-defined]
    assert "截断" in r.details["exploration"]["claim"]  # type: ignore[attr-defined]


def test_vacuous_module_is_coverage_gap() -> None:
    """空洞 OK 防线：无任何同步结构时，"未发现死锁"不是结论而是无知。"""
    m = _module((_op("n1"),), ())
    r = _run(m)  # type: ignore[arg-type]
    assert r.verdict is Verdict.COVERAGE_GAP  # type: ignore[attr-defined]
    assert any(d.rule == "timeline/vacuous" for d in r.diagnostics)  # type: ignore[attr-defined]


def test_missing_timeline_check_is_untrusted() -> None:
    """描述未声明 timeline check → UNTRUSTED_DESCRIPTION（与占用工具同口径）。"""
    cfg = {"ops": [], "vm": {"pipes": []}, "checks": []}
    m = _module((_op("n1"),), ())
    r = run_timeline(cfg, m, "sha:test")
    assert r.verdict is Verdict.UNTRUSTED_DESCRIPTION


def test_unmodeled_ops_downgrade_ok() -> None:
    """存在未建模 op 时 OK 降级为 COVERAGE_GAP（与占用引擎同一裁决哲学）。"""
    from hivm_spec.vir import Gap, GapKind

    w, ws = _sync("w", "wait_flag", 0, "PIPE_V")
    s, ss = _sync("s", "set_flag", 0, "PIPE_MTE2")
    m = _module(
        (_loop((w, s), 2),),
        (ws, ss),
        gaps=(Gap(kind=GapKind.UNMODELED_OP, op="hivm.hir.mystery", detail="未建模", loc=LOC),),
    )
    r = _run(m)  # type: ignore[arg-type]
    assert r.verdict is Verdict.COVERAGE_GAP  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# T2.2：策略集与确定性
# ---------------------------------------------------------------------------


def _healthy_multi_lane() -> VModule:
    steps = []
    syncs = []
    for k in range(4):
        w, ws = _sync(f"w{k}", "wait_flag", 0, "PIPE_V")
        s, ss = _sync(f"s{k}", "set_flag", 0, "PIPE_MTE2")
        steps += [w, s]
        syncs += [ws, ss]
    return _module(tuple(steps), tuple(syncs))


def test_same_seed_is_reproducible() -> None:
    """同策略同种子两次模拟执行序逐字节一致（FR8）。"""
    m = _healthy_multi_lane()
    exp = expand_steps(m, DEFAULT_BOUND)
    a = simulate(exp.steps, "random(seed=7)", {})
    b = simulate(exp.steps, "random(seed=7)", {})
    assert a.order == b.order


def test_different_seeds_explore_different_orders() -> None:
    """不同种子的随机策略探索不同执行序——策略集不是摆设。"""
    m = _healthy_multi_lane()
    exp = expand_steps(m, DEFAULT_BOUND)
    orders = {simulate(exp.steps, f"random(seed={k})", {}).order for k in range(8)}
    assert len(orders) >= 2, "8 个种子只产生一种执行序——随机策略失效"


def test_run_timeline_is_deterministic() -> None:
    """同一输入两次运行 details JSON 逐字节一致（FR8 延伸到时序工具）。"""
    m = _healthy_multi_lane()
    r1 = _run(m)  # type: ignore[arg-type]
    r2 = _run(m)  # type: ignore[arg-type]
    assert r1.to_json_bytes() == r2.to_json_bytes()  # type: ignore[attr-defined]


def test_strategy_blocked_is_warning_not_verdict() -> None:
    """一个事件三个等待者、只有一次 set：必有一个饿死，但属 schedule 依赖 → 只进 diagnostics。

    NFR2：误判（假阳性死锁）优先压制。set 在其泳道首位（可达、会执行），
    确定性层不判死锁；初始装载 + 1 次 set 共 2 个 arm，第 3 个 wait 依
    调度顺序决定谁饿死——策略层受阻作为观察报告。
    """
    s, ss = _sync("s", "set_flag", 0, "PIPE_MTE2")  # 泳道首位——可达
    wv, wsv = _sync("wv", "wait_flag", 0, "PIPE_V")
    wv2, wsv2 = _sync("wv2", "wait_flag", 0, "PIPE_V")
    wm, wsm = _sync("wm", "wait_flag", 0, "PIPE_MTE2")
    m = _module((s, wv, wv2, wm), (ss, wsv, wsv2, wsm))
    r = _run(m)  # type: ignore[arg-type]
    assert r.verdict is Verdict.OK  # type: ignore[attr-defined]
    assert r.details["exploration"]["strategy_blocked"], "应有策略层受阻观察"  # type: ignore[attr-defined]
    assert any(  # type: ignore[attr-defined]
        d.rule == "timeline/strategy-blocked" for d in r.diagnostics
    )


def test_gantt_marks_blocked_wait() -> None:
    """甘特图把受阻 wait 画成 `×`——坏消息在视图层不可隐藏（FR6 视图侧）。"""
    s, ss = _sync("s", "set_flag", 0, "PIPE_MTE2")
    wv, wsv = _sync("wv", "wait_flag", 0, "PIPE_V")
    wv2, wsv2 = _sync("wv2", "wait_flag", 0, "PIPE_V")
    wm, wsm = _sync("wm", "wait_flag", 0, "PIPE_MTE2")
    m = _module((s, wv, wv2, wm), (ss, wsv, wsv2, wsm))
    r = _run(m)  # type: ignore[arg-type]
    chart = render_timeline_chart(r.details)  # type: ignore[attr-defined]
    assert chart, "有执行步骤时甘特图不得为空"
    assert "×" in chart
    assert "▲" in chart and "▽" in chart
    assert all(len(line) <= 80 for line in chart.splitlines())


# ---------------------------------------------------------------------------
# barrier 汇合语义
# ---------------------------------------------------------------------------


def test_barrier_gates_later_ops_across_lanes() -> None:
    """barrier 前的 re-arm 饥饿 wait 阻塞 barrier，并经汇合门控阻塞后继 op。

    事件 5 无人 set：首个 wait 消费初始装载放行，第二个 wait 饿死 →
    barrier（前置步未完）不放行 → post 被 barrier 门控。
    """
    w1, ws1 = _sync("w1", "wait_flag", 5, "PIPE_MTE2")
    w2, ws2 = _sync("w2", "wait_flag", 5, "PIPE_MTE2")  # 无 arm 可用
    b, bs = _sync("b", "pipe_barrier", None, "PIPE_V")
    post = _op("post", "PIPE_V")
    m = _module((w1, w2, b, post), (ws1, ws2, bs))
    r = _run(m)  # type: ignore[arg-type]
    assert r.verdict is Verdict.DEADLOCK  # type: ignore[attr-defined]
    order = r.details["timelines"]["sequential"]["order"]  # type: ignore[attr-defined]
    assert order == [0], "只有首个 wait 放行；barrier 与后继都被挡"


def test_barrier_releases_when_prefix_completes() -> None:
    """前缀完成后 barrier 放行，后继 op 正常执行。"""
    s, ss = _sync("s", "set_flag", 5, "PIPE_V")
    w, ws = _sync("w", "wait_flag", 5, "PIPE_MTE2")
    b, bs = _sync("b", "pipe_barrier", None, "PIPE_V")
    post_m = _op("post_m", "PIPE_MTE2")
    m = _module((s, w, b, post_m), (ss, ws, bs))
    r = _run(m)  # type: ignore[arg-type]
    assert r.verdict is Verdict.OK  # type: ignore[attr-defined]
    order = r.details["timelines"]["sequential"]["order"]  # type: ignore[attr-defined]
    assert len(order) == 4


# ---------------------------------------------------------------------------
# 渲染（T2.4）与 JSON 契约
# ---------------------------------------------------------------------------


def test_gantt_is_deterministic_and_within_width() -> None:
    w, ws = _sync("w", "sync_block_wait", 15, "PIPE_S")
    s, ss = _sync("s", "sync_block_set", 14, "PIPE_S")
    m = _module((w, s), (ws, ss))
    r = _run(m)  # type: ignore[arg-type]
    chart1 = render_timeline_chart(r.details)  # type: ignore[attr-defined]
    chart2 = render_timeline_chart(r.details)  # type: ignore[attr-defined]
    assert chart1 == chart2
    assert all(len(line) <= 80 for line in chart1.splitlines())


def test_chrome_trace_is_valid_json_with_required_fields() -> None:
    m = _healthy_multi_lane()
    r = _run(m)  # type: ignore[arg-type]
    trace = json.loads(chrome_trace_json(r.details))  # type: ignore[attr-defined]
    events = [e for e in trace["traceEvents"] if e["ph"] == "X"]
    assert events, "traceEvents 为空"
    for e in events:
        assert {"name", "pid", "tid", "ts", "dur"} <= set(e)
        assert e["dur"] == 1


def test_details_json_is_serializable() -> None:
    m = _healthy_multi_lane()
    r = _run(m)  # type: ignore[arg-type]
    parsed = json.loads(r.to_json_bytes().decode("utf-8"))  # type: ignore[attr-defined]
    assert parsed["tool"] == "timeline"
    assert parsed["verdict"] == "OK"


def test_timeline_module_never_imports_bishengir() -> None:
    """D6 守卫：时序引擎不得接触 bindings——防止第二个遍历核。"""
    source = (REPO_ROOT / "src" / "hivm_spec" / "timeline.py").read_text(encoding="utf-8")
    assert "bishengir" not in source


# ---------------------------------------------------------------------------
# 展开口径
# ---------------------------------------------------------------------------


def test_static_trip_expands_fully_and_unknown_uses_bound() -> None:
    """静态 trip 全量展开（不截断）；未知 trip 按界截断并标注。"""
    w, ws = _sync("w", "wait_flag", 0, "PIPE_V")
    s, ss = _sync("s", "set_flag", 0, "PIPE_MTE2")
    m_static = _module((_loop((w, s), 5, "Ls"),), (ws, ss))
    exp = expand_steps(m_static, 16)
    assert exp.truncated is False
    assert len(exp.steps) == 10
    assert exp.static_full_loops == 1

    m_unknown = _module((_loop((w, s), None, "Lu"),), (ws, ss))
    exp2 = expand_steps(m_unknown, 16)
    assert exp2.truncated is True
    assert len(exp2.steps) == 32
    assert any("展开界" in n for n in exp2.truncation_notes)


def test_nested_loops_expand_cartesian() -> None:
    """嵌套循环按笛卡尔积展开（外层 2 × 内层 3）。"""
    w, ws = _sync("w", "wait_flag", 0, "PIPE_V")
    inner = _loop((w,), 3, "Li")
    outer = _loop((inner,), 2, "Lo")
    m = _module((outer,), (ws,))
    exp = expand_steps(m, 16)
    assert len(exp.steps) == 6
    paths = {st.loop_path for st in exp.steps}
    assert paths == {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}


def test_total_step_cap_truncates_honestly() -> None:
    """步数硬上限触发截断且如实标注（防步数爆炸）。"""
    w, ws = _sync("w", "wait_flag", 0, "PIPE_V")
    s, ss = _sync("s", "set_flag", 0, "PIPE_MTE2")
    m = _module((_loop((w, s), None, "Lx"),), (ws, ss))
    exp = expand_steps(m, 100000)
    assert exp.truncated is True
    assert len(exp.steps) <= 4096 + 2  # 上限 + 少量缓冲
