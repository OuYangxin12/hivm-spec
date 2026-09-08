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
from hivm_spec.vir import VModule

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


#: 工具名 → 运行函数
_TOOLS = {"ub_occupancy": run_ub_occupancy}


def run_tool(name: str, config: dict[str, Any], module: VModule, spec_hash: str = "") -> ToolResult:
    fn = _TOOLS.get(name)
    if fn is None:
        raise KeyError(
            f"未知工具 {name!r}；已实现：{sorted(_TOOLS)}。timeline/equivalence 见 M2/M3"
        )
    return fn(config, module, spec_hash)


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    """读配置文档，返回 (文档, spec_hash)。

    spec_hash 由字节串重算而非从文档内读取——见 `config_spec_hash`。
    """
    raw = path.read_bytes()
    data: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return data, config_spec_hash(raw)
