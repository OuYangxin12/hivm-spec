"""spec 工具装配与运行（T1.6）：配置文档 + IR → ToolResult。

**verdict 优先级**在此实现：`UNTRUSTED_DESCRIPTION` > `COVERAGE_GAP` > 具体问题 > `OK`。
这个顺序是 FR6/FR7 的落点——描述不可信时后续分析无意义；覆盖不全时
"没发现问题"不能被读作"没有问题"。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from hivm_spec.occupancy import ENGINE_VERSION as OCC_VERSION
from hivm_spec.occupancy import OccupancyResult, analyze_occupancy
from hivm_spec.verdict import Finding, ToolResult, Verdict
from hivm_spec.vir import SyncKind, VModule

__all__ = ["config_spec_hash", "load_config", "run_tool", "run_ub_occupancy"]


def config_spec_hash(config_bytes: bytes) -> str:
    """从配置文档的**字节串**算 spec_hash（框架 §7.3）。

    哈希对象是配置文档而非描述源文件：归一化已消除书写顺序与风格差异，故
    配置文档是语义的规范表示。因此 spec_hash 不可能存放在配置文档**内部**
    （自指），必须由读取方对字节串重算——这同时天然校验了文档未被篡改。
    """
    return "sha256:" + hashlib.sha256(config_bytes).hexdigest()


def _trust_of(config: dict[str, Any]) -> str:
    """配置文档中的最高信任级别。

    取**最高**而非最低是刻意的：结论旁的信任标注要回答"这个结论最多能有多可信"。
    """
    order = ["provisional", "cross-validated", "anchored"]
    best = "provisional"
    for op in config.get("ops", []):
        t = op.get("trust", "provisional")
        if t in order and order.index(t) > order.index(best):
            best = t
    return best


def _capacities_from_config(config: dict[str, Any]) -> dict[str, int | None]:
    caps: dict[str, int | None] = {}
    for sp in config.get("vm", {}).get("spaces", []):
        caps[sp["name"]] = sp.get("capacity")
    return caps


def _gap_findings(module: VModule, occ: OccupancyResult) -> list[Finding]:
    findings: list[Finding] = []
    for gap in module.coverage.gaps:
        findings.append(
            Finding(
                severity="warning",
                message=gap.detail,
                loc=gap.loc,
                rule=f"coverage/{gap.kind.value}",
                extra={"op": gap.op} if gap.op else {},
            )
        )
    for kind, detail, loc in occ.gap_notes:
        findings.append(
            Finding(severity="warning", message=detail, loc=loc, rule=f"occupancy/{kind.value}")
        )
    return findings


def run_ub_occupancy(config: dict[str, Any], module: VModule, spec_hash: str = "") -> ToolResult:
    """UB 占用图工具（§7.1）。

    不需要锚点 IR：溢出是待验 IR 的**内在性质**，对照物是硬件容量线（D12）。
    """
    trust = _trust_of(config)
    caps = _capacities_from_config(config)

    check_cfg = next((c for c in config.get("checks", []) if c["name"] == "ub_occupancy"), None)
    if check_cfg is None:
        return ToolResult(
            tool="ub_occupancy",
            verdict=Verdict.UNTRUSTED_DESCRIPTION,
            spec_hash=spec_hash,
            engine_version=OCC_VERSION,
            trust=trust,
            ir_fingerprint=module.fingerprint(),
            diagnostics=[
                Finding(
                    severity="error",
                    message="描述未声明 ub_occupancy check——无法确定要分析哪些地址空间",
                    rule="assemble/missing-check",
                )
            ],
        )

    spaces = tuple(check_cfg.get("options", {}).get("spaces", ()) or ())
    occ = analyze_occupancy(module, caps, spaces_of_interest=spaces)

    findings = _gap_findings(module, occ)
    details: dict[str, Any] = {"spaces": {}}
    for name, so in sorted(occ.spaces.items()):
        details["spaces"][name] = {
            "capacity": so.capacity,
            "peak_bytes": so.peak_bytes,
            "peak_at_node": so.peak_at,
            "headroom": so.headroom,
            "utilization": round(so.utilization, 4) if so.utilization is not None else None,
            "buffers": len(so.intervals),
            "unsized_buffers": len(so.unsized),
            "curve": so.curve,
            "contributors": [
                {
                    "name": iv.alloc.name,
                    "nbytes": iv.nbytes,
                    "interval": [iv.start, iv.end],
                    "shape": iv.alloc.shape_text,
                }
                for iv in so.peak_contributors
            ],
        }

    # 溢出诊断（含贡献者排序，FR5）
    for name in occ.overflowing_spaces:
        so = occ.spaces[name]
        top = ", ".join(f"{iv.alloc.name}({iv.nbytes}B)" for iv in so.peak_contributors[:5])
        findings.append(
            Finding(
                severity="error",
                message=(
                    f"space '{name}' 峰值 {so.peak_bytes}B 超出容量 {so.capacity}B"
                    f"（超 {so.peak_bytes - (so.capacity or 0)}B）；主要贡献者：{top}"
                ),
                loc=so.peak_contributors[0].alloc.loc if so.peak_contributors else None,
                rule="occupancy/overflow",
                extra={
                    "space": name,
                    "peak_bytes": so.peak_bytes,
                    "capacity": so.capacity,
                },
            )
        )

    verdict = _decide_verdict(module, occ, trust)

    # 缺口存在时，必须说清"没发现溢出"不等于"不会溢出"
    if verdict is Verdict.COVERAGE_GAP:
        unsized = sum(len(s.unsized) for s in occ.spaces.values())
        unmodeled = module.coverage.unmodeled_ops()
        reasons = []
        if occ.is_vacuous or not occ.spaces:
            reasons.append(
                "在被检查的地址空间中未找到任何 buffer——本结论不代表『不溢出』，而代表『未能分析』"
            )
        if unsized:
            reasons.append(f"{unsized} 个 buffer 尺寸未知（峰值仅为下界）")
        if unmodeled:
            reasons.append(f"{len(unmodeled)} 个 op 未建模：{list(unmodeled[:5])}")
        # reasons 可能为空（如仅存在 unknown-space 效应缺口）——此时必须说明
        # 缺口在 diagnostics 里，不能输出"原因："后接空串。
        if not reasons:
            reasons.append("存在若干效应空间/尺寸无法确定的缺口，详见 diagnostics")
        findings.insert(
            0,
            Finding(
                severity="error",
                message=("覆盖不完整，**不能**据此断言不溢出。原因：" + "；".join(reasons)),
                rule="verdict/coverage-gap",
            ),
        )

    return ToolResult(
        tool="ub_occupancy",
        verdict=verdict,
        spec_hash=spec_hash,
        engine_version=OCC_VERSION,
        trust=trust,
        ir_fingerprint=module.fingerprint(),
        details=details,
        diagnostics=findings,
    )


def _decide_verdict(module: VModule, occ: OccupancyResult, trust: str) -> Verdict:
    """verdict 优先级裁决。

    **溢出优先于缺口**：已经确证的溢出是真问题，不该被"还有别的看不懂"掩盖。
    但"没发现溢出"在有缺口时必须降级为 COVERAGE_GAP——否则等于用"我没看全"
    换取一个 OK，正是 FR6 要防的自欺。
    """
    if occ.any_overflow:
        return Verdict.OVERFLOW

    # **空洞 OK 的防线**：一个 buffer 都没找到时，"未溢出"不是结论而是无知。
    # 实测 annotate-vf-alias.mlir 曾因此拿到干净的 OK——它的 3 个 UB buffer
    # 是函数参数而非 memref.alloc，工具什么都没分析却给了绿灯。
    if occ.is_vacuous or not occ.spaces:
        return Verdict.COVERAGE_GAP

    has_unsized = any(s.unsized for s in occ.spaces.values())
    if not module.coverage.is_complete or has_unsized:
        return Verdict.COVERAGE_GAP

    # 容量未知时同样不能给 OK
    if any(s.capacity is None for s in occ.spaces.values()):
        return Verdict.COVERAGE_GAP

    return Verdict.OK


# ---------------------------------------------------------------------------
# 时序图工具（T2.1–T2.5，§7.2）：不需要锚点 IR——死锁是待验 IR 同步结构的
# 内在性质（D12/R2）。判定分两层：确定性层（结构性死锁 → DEADLOCK）与
# 探索层（策略集 × 展开界，结论恒带限定语）。
# ---------------------------------------------------------------------------


def run_timeline(
    config: dict[str, Any],
    module: VModule,
    spec_hash: str = "",
    *,
    bound: int | None = None,
    strategies: tuple[str, ...] | None = None,
) -> ToolResult:
    """时序图工具：pipe/event 时间线 + 结构性死锁判定。"""
    from hivm_spec import timeline as tl

    trust = _trust_of(config)

    check_cfg = next((c for c in config.get("checks", []) if c["name"] == "timeline"), None)
    if check_cfg is None:
        return ToolResult(
            tool="timeline",
            verdict=Verdict.UNTRUSTED_DESCRIPTION,
            spec_hash=spec_hash,
            engine_version=tl.TIMELINE_ENGINE_VERSION,
            trust=trust,
            ir_fingerprint=module.fingerprint(),
            diagnostics=[
                Finding(
                    severity="error",
                    message="描述未声明 timeline check——无法确定调度口径（策略集/展开界）",
                    rule="assemble/missing-check",
                )
            ],
        )

    opts = check_cfg.get("options", {}) or {}
    bound = bound if bound is not None else int(opts.get("unroll_bound", tl.DEFAULT_BOUND))
    if bound < 1:
        return ToolResult(
            tool="timeline",
            verdict=Verdict.UNTRUSTED_DESCRIPTION,
            spec_hash=spec_hash,
            engine_version=tl.TIMELINE_ENGINE_VERSION,
            trust=trust,
            ir_fingerprint=module.fingerprint(),
            diagnostics=[
                Finding(
                    severity="error",
                    message=f"展开界非法：{bound}（须 ≥1）",
                    rule="assemble/bad-bound",
                )
            ],
        )
    if strategies is None:
        strategies = (*tl.STRATEGIES, *(f"random(seed={k})" for k in tl.RANDOM_SEEDS))
    pipe_priority = {name: idx for idx, name in enumerate(config.get("vm", {}).get("pipes", []))}

    expansion = tl.expand_steps(module, bound)
    deadlocks = tl.structural_fixpoint(
        expansion.steps, expansion.truncated, tl.events_with_any_set(module)
    )
    sims = [tl.simulate(expansion.steps, s, pipe_priority) for s in strategies]

    # -- 分析缺口盘点（FR4：缺口可见，不静默） ----------------------------
    sync_steps = [st for st in expansion.steps if st.sync is not None]
    unanalyzable = [
        st
        for st in sync_steps
        if st.sync is not None and tl.event_of(st) is None and st.sync.kind != SyncKind.PIPE_BARRIER
    ]
    n_sets = sum(1 for s in module.syncs if s.kind in _TL_SET_KINDS)
    n_waits = sum(1 for s in module.syncs if s.kind in _TL_WAIT_KINDS)
    n_barriers = sum(1 for s in module.syncs if s.kind is SyncKind.PIPE_BARRIER)
    events_stat: dict[str, dict[str, int]] = {}
    for s in module.syncs:
        if s.event_id is None:
            continue
        slot = events_stat.setdefault(str(s.event_id), {"sets": 0, "waits": 0})
        if s.kind in _TL_SET_KINDS:
            slot["sets"] += 1
        elif s.kind in _TL_WAIT_KINDS:
            slot["waits"] += 1

    # -- 策略层观察（schedule 依赖，不进 verdict） -------------------------
    strategy_blocked: list[dict[str, Any]] = []
    for sim in sims:
        for seq in sim.blocked:
            st = expansion.steps[seq]
            ev = tl.event_of(st)
            # 确定性层已判定为死锁的 wait 不重复报告
            if any(d.step_seq == seq for d in deadlocks):
                continue
            strategy_blocked.append(
                {
                    "strategy": sim.strategy,
                    "step_seq": seq,
                    "event_id": ev,
                    "loc": st.node.loc.describe(),
                    "message": (
                        f"策略 {sim.strategy} 下 wait(事件 {ev}) 受阻——schedule 依赖观察，"
                        "非确定性死锁判定"
                    ),
                }
            )

    # -- verdict 裁决：DEADLOCK > COVERAGE_GAP > OK ------------------------
    if deadlocks:
        verdict = Verdict.DEADLOCK
    elif not module.syncs:
        # 空洞 OK 防线（同占用引擎的 is_vacuous）：没有任何同步结构时，
        # "未发现死锁"不是结论而是无知。
        verdict = Verdict.COVERAGE_GAP
    elif not module.coverage.is_complete or unanalyzable:
        verdict = Verdict.COVERAGE_GAP
    else:
        verdict = Verdict.OK

    # -- 探索层结论措辞（T2.5：禁止无条件"无死锁"） ------------------------
    strategy_label = "/".join(strategies)
    claim = f"在{{{strategy_label}}}×{{展开界={bound}}}内未发现死锁"
    if expansion.truncated:
        claim += "（循环展开截断——负向结论仅在该界内成立）"
    if verdict is Verdict.DEADLOCK:
        claim += "；另有确定性死锁判定，见 deadlocks/诊断"
    elif verdict is Verdict.COVERAGE_GAP:
        claim = "覆盖不完整，探索层结论已降级为 COVERAGE_GAP，不构成'无死锁'依据"

    # -- diagnostics -------------------------------------------------------
    findings: list[Finding] = []
    for d in deadlocks:
        findings.append(
            Finding(
                severity="error",
                message=d.message,
                loc=d.loc,
                rule=f"timeline/{d.rule}",
                extra={"event_id": d.event_id, "step_seq": d.step_seq},
            )
        )
    for gap in module.coverage.gaps:
        findings.append(
            Finding(
                severity="warning",
                message=gap.detail,
                loc=gap.loc,
                rule=f"coverage/{gap.kind.value}",
                extra={"op": gap.op} if gap.op else {},
            )
        )
    if not module.syncs:
        findings.append(
            Finding(
                severity="error",
                message=(
                    "未在 IR 中发现任何同步结构（set/wait/barrier）——本结论不代表"
                    "『无死锁风险』，而代表『未能分析』"
                ),
                rule="timeline/vacuous",
            )
        )
    elif unanalyzable:
        findings.append(
            Finding(
                severity="warning",
                message=(
                    f"{len(unanalyzable)} 个 set/wait 未标注事件 id，无法配对分析"
                    "——涉及它们的结论不可信"
                ),
                loc=unanalyzable[0].node.loc,
                rule="timeline/unanalyzable-event",
            )
        )
    if expansion.truncated:
        for note in expansion.truncation_notes:
            findings.append(Finding(severity="warning", message=note, rule="timeline/truncation"))
    if verdict is not Verdict.DEADLOCK:
        for item in strategy_blocked:
            findings.append(
                Finding(
                    severity="warning",
                    message=item["message"],
                    loc=None,
                    rule="timeline/strategy-blocked",
                    extra={"strategy": item["strategy"], "event_id": item["event_id"]},
                )
            )
    if verdict is Verdict.OK:
        findings.append(Finding(severity="info", message=claim, rule="timeline/exploration-claim"))

    # -- details（JSON 契约） ----------------------------------------------
    lane_order = list(config.get("vm", {}).get("pipes", []))
    for st in expansion.steps:
        if st.lane not in lane_order:
            lane_order.append(st.lane)
    details: dict[str, Any] = {
        "bound": bound,
        "strategies": list(strategies),
        "pipe_order": lane_order,
        "truncated": expansion.truncated,
        "truncation_notes": list(expansion.truncation_notes),
        "static_full_loops": expansion.static_full_loops,
        "steps_total": len(expansion.steps),
        "sync_counts": {"set": n_sets, "wait": n_waits, "barrier": n_barriers},
        "events": dict(sorted(events_stat.items(), key=lambda kv: int(kv[0]))),
        "deadlocks": [
            {
                "rule": d.rule,
                "event_id": d.event_id,
                "step_seq": d.step_seq,
                "loc": d.loc.describe(),
                "message": d.message,
            }
            for d in deadlocks
        ],
        "exploration": {
            "claim": claim,
            "strategy_blocked": strategy_blocked,
        },
        "timelines": {},
    }
    for sim in sims:
        details["timelines"][sim.strategy] = _timeline_details(expansion, sim)

    return ToolResult(
        tool="timeline",
        verdict=verdict,
        spec_hash=spec_hash,
        engine_version=tl.TIMELINE_ENGINE_VERSION,
        trust=trust,
        ir_fingerprint=module.fingerprint(),
        details=details,
        diagnostics=findings,
    )


#: details 中每条时间线的步数上限（JSON 体积护栏；完整轨迹走 Chrome Trace）
_TIMELINE_STEP_CAP = 400
_TL_SET_KINDS = frozenset({SyncKind.SET_FLAG, SyncKind.SYNC_BLOCK_SET})
_TL_WAIT_KINDS = frozenset({SyncKind.WAIT_FLAG, SyncKind.SYNC_BLOCK_WAIT})


def _timeline_details(expansion: Any, sim: Any) -> dict[str, Any]:
    """单策略时间线的可序列化视图（执行序前 _TIMELINE_STEP_CAP 步）。

    **受阻 wait 也必须出现在视图里**（坏消息不可隐藏，FR6 视图侧延伸）：
    追加在执行序之后并带 blocked 标记，甘特图据此画 `×`、trace 据此标注。
    """
    from hivm_spec import timeline as tl

    steps: list[dict[str, Any]] = []
    for pos, seq in enumerate(sim.order[:_TIMELINE_STEP_CAP]):
        st = expansion.steps[seq]
        steps.append(
            {
                "pos": pos,
                "seq": seq,
                "op": st.node.op,
                "lane": st.lane,
                "kind": st.sync.kind.value if st.sync is not None else "op",
                "event": tl.event_of(st),
                "iter": ".".join(str(x) for x in st.loop_path),
                "loc": st.node.loc.describe(),
                "blocked": False,
            }
        )
    room = max(0, _TIMELINE_STEP_CAP - len(steps))
    for k, seq in enumerate(sim.blocked[:room]):
        st = expansion.steps[seq]
        steps.append(
            {
                "pos": len(steps) + k,
                "seq": seq,
                "op": st.node.op,
                "lane": st.lane,
                "kind": st.sync.kind.value if st.sync is not None else "op",
                "event": tl.event_of(st),
                "iter": ".".join(str(x) for x in st.loop_path),
                "loc": st.node.loc.describe(),
                "blocked": True,
            }
        )
    return {
        "order": list(sim.order[:_TIMELINE_STEP_CAP]),
        "order_truncated": len(sim.order) > _TIMELINE_STEP_CAP,
        "steps": steps,
        "blocked": list(sim.blocked),
    }


#: 工具名 → 运行函数（timeline 走 run_timeline 的 kwargs 分发）
_TOOLS = {"ub_occupancy": run_ub_occupancy}


def run_tool(
    name: str, config: dict[str, Any], module: VModule, spec_hash: str = "", **kwargs: Any
) -> ToolResult:
    if name == "timeline":
        return run_timeline(config, module, spec_hash, **kwargs)
    fn = _TOOLS.get(name)
    if fn is None:
        raise KeyError(f"未知工具 {name!r}；已实现：ub_occupancy / timeline。equivalence 见 M3")
    return fn(config, module, spec_hash)


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    """读配置文档，返回 (文档, spec_hash)。

    spec_hash 由字节串重算而非从文档内读取——见 `config_spec_hash`。
    """
    raw = path.read_bytes()
    data: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return data, config_spec_hash(raw)
