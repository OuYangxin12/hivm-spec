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

    **未决语义假设封顶**（D9 前置，M2 审查发现 3）：只要描述里还有未对拍的
    语义假设（`assumptions` 非空），信任一律封顶 `provisional`——语义尚未与
    权威链对齐就宣称更高信任，等于把猜测当结论。
    """
    order = ["provisional", "cross-validated", "anchored"]
    best = "provisional"
    for op in config.get("ops", []):
        t = op.get("trust", "provisional")
        if t in order and order.index(t) > order.index(best):
            best = t
    if config.get("assumptions"):
        return "provisional"
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

    # -- 策略层观察（不进 verdict；须区分 schedule 依赖 / 不变） ------------
    # 受阻性质判别（审查发现 1b）：全策略受阻集合一致 = schedule-invariant，
    # 不得再表述为"schedule 依赖观察"——那是把已观测到的不变量说成偶然。
    blocked_kind = tl.classify_blocked({sim.strategy: sim.blocked for sim in sims})
    deadlock_seqs = {d.step_seq for d in deadlocks}
    #: 按 (step_seq) 聚合，避免 策略数×受阻步数 的笛卡尔噪声淹没摘要
    blocked_by_step: dict[int, list[str]] = {}
    for sim in sims:
        for seq in sim.blocked:
            if seq in deadlock_seqs:
                continue  # 确定性层已判定为死锁的 wait 不重复报告
            blocked_by_step.setdefault(seq, []).append(sim.strategy)
    strategy_blocked: list[dict[str, Any]] = []
    for seq, strats in sorted(blocked_by_step.items()):
        st = expansion.steps[seq]
        ev = tl.event_of(st)
        invariant = len(strats) == len(sims)
        nature = (
            "全部策略下均受阻——与调度选择无关（schedule-invariant）"
            if invariant
            else f"{len(strats)}/{len(sims)} 个策略下受阻——schedule 依赖观察"
        )
        strategy_blocked.append(
            {
                "strategy": strats[0] if len(strats) == 1 else "|".join(strats),
                "strategies": strats,
                "schedule_invariant": invariant,
                "step_seq": seq,
                "event_id": ev,
                "loc": st.node.loc.describe(),
                "message": f"wait(事件 {ev}) {nature}，非确定性死锁判定",
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
    # 审查发现 1c：claim 不得与自身 diagnostics 矛盾——存在全策略一致受阻时，
    # "未发现死锁"必须附带该保留（截断是其最常见成因，见 truncation_notes）
    if verdict is not Verdict.DEADLOCK and blocked_kind == "schedule-invariant":
        n_inv = sum(1 for item in strategy_blocked if item["schedule_invariant"])
        claim += (
            f"；但有 {n_inv} 个 wait 在**全部策略**下均受阻（schedule-invariant，"
            "确定性层未能归类）——不可读作'无死锁'"
        )

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
                    rule=(
                        "timeline/blocked-schedule-invariant"
                        if item["schedule_invariant"]
                        else "timeline/strategy-blocked"
                    ),
                    extra={
                        "strategy": item["strategy"],
                        "strategies": item["strategies"],
                        "event_id": item["event_id"],
                        "schedule_invariant": item["schedule_invariant"],
                    },
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
            "blocked_kind": blocked_kind,
            "strategy_blocked": strategy_blocked,
        },
        #: 规则 C 的算术证据（无论是否触发判定都输出——供需账目本身即诊断材料）
        "arm_supply": [
            {
                "event_id": d.event_id,
                "waits": d.waits,
                "sets": d.sets,
                "initial_arm": d.initial_arm,
                "starved": d.starved,
            }
            for d in tl.arm_deficits(expansion.steps)
        ],
        #: 本结论所依赖的未对拍语义假设（D9 前置）——判定前提必须随结论可见，
        #: 否则读者无法判断"DEADLOCK/OK"是在哪套语义口径下成立的
        "assumptions": [
            {
                "subject": a.get("subject", ""),
                "assumed": a.get("assumed", ""),
                "risk_direction": a.get("risk_direction", ""),
                "resolve_by": a.get("resolve_by", ""),
            }
            for a in config.get("assumptions", [])
            if str(a.get("subject", "")).startswith("timeline/")
        ],
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


def _run_symbolic_equivalence(
    config: dict[str, Any],
    module: VModule,
    anchor: VModule,
    spec_hash: str,
    trust: Any,
    *,
    bound: int | None = None,
) -> ToolResult:
    """有界符号等价（T4.2）。

    两侧用**同名**符号输入：符号名按位置派生（in0/in1/…）而非用 SSA 文本，
    因为两份 IR 的 SSA 名可能不同，用它做符号名会让两侧变成不同的自由变量，
    等价判定就恒能找到"反例"（其实只是变量不同）。
    """
    from hivm_spec import symbolic_equiv as se
    from hivm_spec.inputs import specs_from_module
    from hivm_spec.interpret import interpret
    from hivm_spec.timeline import DEFAULT_BOUND
    from hivm_spec.values import symbol

    eff_bound = bound if bound is not None else DEFAULT_BOUND
    diagnostics: list[Finding] = []

    def _symbolic_inputs(mod: VModule) -> dict[str, Any]:
        specs, problems = specs_from_module(mod)
        for why in problems:
            diagnostics.append(
                Finding(
                    severity="warning",
                    message=f"输入无法推导：{why}",
                    rule="equivalence/undeducible-input",
                )
            )
        return {s.name: symbol(f"in{i}") for i, s in enumerate(specs)}

    left = interpret(module, config, _symbolic_inputs(module), bound=eff_bound)
    right = interpret(anchor, config, _symbolic_inputs(anchor), bound=eff_bound)

    # 覆盖缺口先行（与具体档 M3/equivalence.py 同一口径）：
    # 有未建模 op 时，"没发现发散"可能只是因为没算那一步。
    if left.has_unmodeled or right.has_unmodeled:
        gaps = left.gaps + right.gaps
        gnames = sorted({g.op for g in gaps if g.kind.value == "unmodeled_op" and g.op})
        diagnostics.append(
            Finding(
                severity="warning",
                message=(
                    f"存在未建模 op（{', '.join(gnames) or '未具名'}），"
                    "未建模的步骤不参与比较——'没发现发散'不成立"
                ),
                rule="equivalence/coverage-gap",
            )
        )
        return ToolResult(
            tool="equivalence",
            verdict=Verdict.COVERAGE_GAP,
            spec_hash=spec_hash,
            engine_version=se.SYMBOLIC_ENGINE_VERSION,
            trust=trust,
            ir_fingerprint=module.fingerprint(),
            diagnostics=diagnostics,
            details={
                "mode": "symbolic",
                "bound": eff_bound,
                "anchor_fingerprint": anchor.fingerprint(),
            },
        )

    diff = se.compare_symbolic(
        left.values_by_seq,
        right.values_by_seq,
        bound=eff_bound,
        op_names={tr.seq: tr.op for tr in left.traces},
    )

    for note in diff.notes:
        diagnostics.append(Finding(severity="info", message=note, rule="symbolic/note"))

    if diff.outcome is se.SymbolicOutcome.COUNTEREXAMPLE:
        verdict = Verdict.MISMATCH
        diagnostics.append(
            Finding(
                severity="error",
                message=diff.describe(),
                rule="symbolic/counterexample",
            )
        )
    elif diff.outcome is se.SymbolicOutcome.UNKNOWN:
        # 超时/不可解/无可比步 → **缺口**，绝不按通过处理（M4 卡 §4.4）
        verdict = Verdict.COVERAGE_GAP
        diagnostics.append(
            Finding(
                severity="warning",
                message=diff.describe(),
                rule="symbolic/unknown",
            )
        )
    else:
        verdict = Verdict.OK
        diagnostics.append(
            Finding(severity="info", message=diff.describe(), rule="symbolic/proven")
        )

    return ToolResult(
        tool="equivalence",
        verdict=verdict,
        spec_hash=spec_hash,
        engine_version=se.SYMBOLIC_ENGINE_VERSION,
        trust=trust,
        ir_fingerprint=module.fingerprint(),
        diagnostics=diagnostics,
        details={
            "mode": "symbolic",
            "outcome": diff.outcome.value,
            "bound": diff.bound,
            "compared_steps": diff.compared,
            "counterexample": diff.counterexample,
            "uninterpreted": list(diff.uninterpreted),
            "has_division": diff.has_division,
            "anchor_fingerprint": anchor.fingerprint(),
            # 结论强度声明：与具体档并列，不可混为一谈
            "semantics": "Real（无限精度有理数）——不覆盖浮点舍入与结合律缺失",
        },
    )


def run_equivalence(
    config: dict[str, Any],
    module: VModule,
    spec_hash: str = "",
    *,
    anchor: VModule | None = None,
    bound: int | None = None,
    mode: str = "concrete",
) -> ToolResult:
    """等价验证工具：具体执行差分（T3.5/T3.6）或有界符号等价（T4.2）。

    `mode="symbolic"` 切到符号档。二者结论**并列而非替代**：具体档说的是
    "这组输入上没发现发散"，符号档说的是"有界内所有输入上无反例"，且符号档
    用 Real 语义、不覆盖浮点精度（M4 卡 §4.1）。

    `anchor is None` 时**不自比**——报 COVERAGE_GAP。拿同一份 IR 自比恒等于
    "通过"，却什么都没验证，而报告上的 OK 与真验证过的 OK 长得一模一样
    （M3 卡 §4 要点 2：最典型的自欺形态）。
    """
    from hivm_spec import equivalence as eq
    from hivm_spec.inputs import InputStrategy, specs_from_module
    from hivm_spec.interpret import interpret
    from hivm_spec.values import concrete

    trust = _trust_of(config)
    check_cfg = next((c for c in config.get("checks", []) if c["name"] == "equivalence"), None)
    if check_cfg is None:
        return ToolResult(
            tool="equivalence",
            verdict=Verdict.UNTRUSTED_DESCRIPTION,
            spec_hash=spec_hash,
            engine_version=eq.EQUIVALENCE_ENGINE_VERSION,
            trust=trust,
            ir_fingerprint=module.fingerprint(),
            diagnostics=[
                Finding(
                    severity="error",
                    message="描述未声明 equivalence check——无法确定容差口径（rtol/atol）",
                    rule="assemble/missing-check",
                )
            ],
        )

    tol = eq.tolerance_from_config(config)
    if anchor is None:
        return ToolResult(
            tool="equivalence",
            verdict=Verdict.COVERAGE_GAP,
            spec_hash=spec_hash,
            engine_version=eq.EQUIVALENCE_ENGINE_VERSION,
            trust=trust,
            ir_fingerprint=module.fingerprint(),
            diagnostics=[
                Finding(
                    severity="error",
                    message=(
                        "未提供对拍锚点（--anchor）：等价验证需要一份参照 IR。"
                        "本工具不做自比——同一份 IR 自比恒为 OK，却什么都没验证"
                    ),
                    rule="equivalence/no-anchor",
                )
            ],
            details={"tolerance": tol.as_dict()},
        )

    if mode == "symbolic":
        return _run_symbolic_equivalence(config, module, anchor, spec_hash, trust, bound=bound)

    # 两侧必须拿到**同一组**输入，否则"结果不同"可能只是输入不同。
    # 输入规格取自待验侧；锚点侧缺哪个输入就按缺口处理（不另生成）。
    specs, problems = specs_from_module(module)
    strategy = InputStrategy()
    raw = strategy.generate_all(specs)
    dtype_of_name = {s.name: s.dtype for s in specs}
    inputs = {k: concrete(v, dtype_of_name[k]) for k, v in raw.items()}

    diagnostics: list[Finding] = [
        Finding(
            severity="warning",
            message=f"输入无法推导：{why}",
            rule="equivalence/undeducible-input",
        )
        for why in problems
    ]

    left = interpret(module, config, inputs, bound=bound)
    # 锚点侧用**同一批输入对象**：按名字取，缺失的不补
    anchor_specs, _ = specs_from_module(anchor)
    anchor_inputs = {s.name: inputs[s.name] for s in anchor_specs if s.name in inputs}
    right = interpret(anchor, config, anchor_inputs, bound=bound)

    diff = eq.compare(left, right, tol)

    for note in diff.notes:
        diagnostics.append(Finding(severity="warning", message=note, rule="equivalence/note"))
    if diff.first is not None:
        diagnostics.append(
            Finding(
                severity="error",
                message=f"首个发散：{diff.first.describe()}（另有 {diff.impacted} 步受影响）",
                rule="equivalence/first-divergence",
            )
        )

    details: dict[str, Any] = {
        "tolerance": tol.as_dict(),
        "compared_steps": diff.compared,
        "impacted_steps": diff.impacted,
        "inputs": strategy.provenance(),
        "anchor_fingerprint": anchor.fingerprint(),
    }
    # 逐 op 值哈希 trace（T3.6）：随结论输出，供人工比对与二次定位。
    # 只放哈希与标签，不放全量张量——后者动辄上百 MB，且真正要看的是"从哪一步
    # 起两侧不同"，哈希足够回答。
    details["trace"] = [
        {
            "seq": lt.seq,
            "label": lt.label,
            "op": lt.op,
            "line": lt.loc.line,
            "left": lt.out_hash,
            "right": rt.out_hash,
            "same": lt.out_hash == rt.out_hash,
            "unmodeled": lt.unmodeled or rt.unmodeled,
        }
        for lt, rt in zip(left.traces, right.traces, strict=False)
    ]

    if diff.first is not None:
        details["first_divergence"] = {
            "seq": diff.first.seq,
            "op": diff.first.op,
            "label": diff.first.label,
            "file": diff.first.file,
            "line": diff.first.line,
            "left": diff.first.left_summary,
            "right": diff.first.right_summary,
            "max_abs": diff.first.max_abs,
            "max_rel": diff.first.max_rel,
        }

    return ToolResult(
        tool="equivalence",
        verdict=diff.verdict,
        spec_hash=spec_hash,
        engine_version=eq.EQUIVALENCE_ENGINE_VERSION,
        trust=trust,
        ir_fingerprint=module.fingerprint(),
        diagnostics=diagnostics,
        details=details,
    )


def run_sync_pairing(config: dict[str, Any], module: VModule, spec_hash: str = "") -> ToolResult:
    """同步静态配对检查（T4.4）——账本级快速检查。

    不需要锚点：配对完整性是待验 IR 的**内在性质**（与 ub_occupancy 同理，D12）。

    **本工具不产出 DEADLOCK。** 账目异常在语义上是"可疑"而非"必然死锁"——
    真实案例（CreatePreload stage-major）里顺序不可行时计数照样平衡。死锁判定
    归 timeline 工具；二者不一致时以 timeline 为准（M4 卡 §4.6）。
    """
    from hivm_spec import pairing as pr

    trust = _trust_of(config)
    check_cfg = next((c for c in config.get("checks", []) if c["name"] == "sync_pairing"), None)
    if check_cfg is None:
        return ToolResult(
            tool="sync_pairing",
            verdict=Verdict.UNTRUSTED_DESCRIPTION,
            spec_hash=spec_hash,
            engine_version=pr.PAIRING_ENGINE_VERSION,
            trust=trust,
            ir_fingerprint=module.fingerprint(),
            diagnostics=[
                Finding(
                    severity="error",
                    message="描述未声明 sync_pairing check——无法确定配对口径",
                    rule="assemble/missing-check",
                )
            ],
        )

    report = pr.analyze_pairing(module)

    diagnostics = [
        Finding(severity=f.severity, message=f.message, rule=f.rule, loc=f.loc)
        for f in report.findings
    ]
    for note in report.notes:
        diagnostics.append(Finding(severity="info", message=note, rule="pairing/note"))

    # verdict 归属：
    # - 有 wait 却全无 set → 该 wait 必然等不到，是确定的问题（DEADLOCK 由
    #   timeline 判，这里用 MISMATCH 表达"账目与语义要求不符"）；
    # - 只有 warning（orphan-set / 不平）→ 不构成验证失败；
    # - 一个事件都没有 → 无从检查，报缺口而非 OK。
    if report.has_error:
        verdict = Verdict.MISMATCH
    elif not report.ledgers:
        verdict = Verdict.COVERAGE_GAP
        diagnostics.append(
            Finding(
                severity="warning",
                message="模块内无可配对的同步事件——本检查未实际发生，不得据此判定同步正确",
                rule="pairing/vacuous",
            )
        )
    else:
        verdict = Verdict.OK

    return ToolResult(
        tool="sync_pairing",
        verdict=verdict,
        spec_hash=spec_hash,
        engine_version=pr.PAIRING_ENGINE_VERSION,
        trust=trust,
        ir_fingerprint=module.fingerprint(),
        diagnostics=diagnostics,
        details={
            "events": [
                {
                    "event_id": led.event_id,
                    "sets": led.sets,
                    "waits": led.waits,
                    "balanced": led.balanced,
                    "pipes": list(led.pipes),
                    "has_implicit": led.has_implicit,
                }
                for led in report.ledgers
            ],
            "unattributable": report.unattributable,
            "implicit_events": report.implicit_events,
        },
    )


def run_uninit_read(
    config: dict[str, Any], module: VModule, spec_hash: str = "", *, bound: int | None = None
) -> ToolResult:
    """未初始化读检查（T4.3）。

    不需要锚点：未初始化读是待验 IR 的**内在性质**（D12）。
    用独立 tainted 标记位而非 poison 魔数——魔数可能是合法计算结果，NaN 更不
    安全（真实计算本就会产生 NaN），届时无法区分"读了未初始化"与"算出了 NaN"。
    """
    from hivm_spec import uninit as un

    trust = _trust_of(config)
    check_cfg = next((c for c in config.get("checks", []) if c["name"] == "uninit_read"), None)
    if check_cfg is None:
        return ToolResult(
            tool="uninit_read",
            verdict=Verdict.UNTRUSTED_DESCRIPTION,
            spec_hash=spec_hash,
            engine_version=un.UNINIT_ENGINE_VERSION,
            trust=trust,
            ir_fingerprint=module.fingerprint(),
            diagnostics=[
                Finding(
                    severity="error",
                    message="描述未声明 uninit_read check",
                    rule="assemble/missing-check",
                )
            ],
        )

    report = un.analyze_uninit_reads(module, config, bound=bound)

    diagnostics = [
        Finding(
            severity="error",
            message=r.describe(with_loc=False),
            rule="uninit/read-before-write",
            loc=r.loc,
            extra={"buffer": r.buffer, "param": r.param, "seq": r.seq},
        )
        for r in report.reads
    ]
    for note in report.notes:
        diagnostics.append(Finding(severity="info", message=note, rule="uninit/note"))

    if report.reads:
        verdict = Verdict.MISMATCH
    elif report.tracked_buffers == 0:
        # 一个本地缓冲都没有 → 检查未实际发生。报 OK 等于用"没查"冒充"没问题"。
        verdict = Verdict.COVERAGE_GAP
        diagnostics.append(
            Finding(
                severity="warning",
                message="无本地分配的缓冲可供分析——本检查未实际发生",
                rule="uninit/vacuous",
            )
        )
    else:
        verdict = Verdict.OK

    return ToolResult(
        tool="uninit_read",
        verdict=verdict,
        spec_hash=spec_hash,
        engine_version=un.UNINIT_ENGINE_VERSION,
        trust=trust,
        ir_fingerprint=module.fingerprint(),
        diagnostics=diagnostics,
        details={
            "tracked_buffers": report.tracked_buffers,
            "unmodeled_ops": list(report.unmodeled_ops),
            "truncated": report.truncated,
            "reads": [
                {"seq": r.seq, "op": r.op, "buffer": r.buffer, "param": r.param}
                for r in report.reads
            ],
        },
    )


#: 工具名 → 运行函数（timeline 走 run_timeline 的 kwargs 分发）
_TOOLS = {
    "ub_occupancy": run_ub_occupancy,
    "sync_pairing": run_sync_pairing,
    "uninit_read": run_uninit_read,
}


def run_tool(
    name: str, config: dict[str, Any], module: VModule, spec_hash: str = "", **kwargs: Any
) -> ToolResult:
    if name == "timeline":
        return run_timeline(config, module, spec_hash, **kwargs)
    if name == "equivalence":
        return run_equivalence(config, module, spec_hash, **kwargs)
    fn = _TOOLS.get(name)
    if fn is None:
        raise KeyError(
            f"未知工具 {name!r}；已实现：ub_occupancy / timeline / equivalence / "
            "sync_pairing / uninit_read"
        )
    return fn(config, module, spec_hash)


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    """读配置文档，返回 (文档, spec_hash)。

    spec_hash 由字节串重算而非从文档内读取——见 `config_spec_hash`。
    """
    raw = path.read_bytes()
    data: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return data, config_spec_hash(raw)
