"""`hivm-spec run`：一条命令跑完全部适用检查。

## 本模块只做编排

**不新增任何判定逻辑。** 每个子检查的结论原样透传，聚合层只做一件事：按既有
优先级取最高严重度。特别地——**绝不把子检查的 `COVERAGE_GAP` 吞成 `OK`**。

这条是本模块最容易写错也最致命的地方：一个"跑了 6 项检查，5 项 OK、1 项没验成"
的 IR，聚合结论必须是 `COVERAGE_GAP` 而不是 `OK`。有测试锁定。

## 哪些检查跑、哪些不跑

`ub_occupancy` / `timeline` / `sync_pairing` / `uninit_read` 恒跑。

`equivalence` 需要 `--anchor`：**没有锚点时不静默消失，而是留一条
`COVERAGE_GAP` 记录**——否则报告上"5 项全 OK"会被读成"这份 IR 没问题"，
而实际上最重要的语义等价压根没验。这与 `run_equivalence` 拒绝自比是同一条
纪律（M3 卡 §4 要点 2）。

符号档额外要 z3；缺 z3 时同样留缺口记录而非假装它不存在（T4.0 决策）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hivm_spec.verdict import Finding, ToolResult, Verdict, verdict_exit_code

if TYPE_CHECKING:
    from hivm_spec.vir import VModule

__all__ = ["ORCHESTRATOR_VERSION", "RunReport", "aggregate", "run_checks"]

ORCHESTRATOR_VERSION = "0.1.0"

#: 跨检查聚合的裁决顺序。
#:
#: **已确证的问题优先于别处的覆盖缺口**——这是 `assemble._decide_verdict`
#: 早就定下的口径（"溢出优先于缺口：已经确证的溢出是真问题，不该被'还有
#: 别的看不懂'掩盖"），此处原样搬到跨检查层。若一个检查确证了 MISMATCH，
#: 另一个检查因 IR 无同步结构而"未能分析"，整体必须是 MISMATCH：让一个无关
#: 检查的无知掩盖已发现的缺陷，退出码会变绿，CI 就漏放了真 bug。
#:
#: 缺口仍然有牙齿：它挡在 OK 前面——只要有一项没验成，"全部检查通过"这个
#: **干净**结论就给不出。所以缺口的作用是"禁止假绿灯"，而不是"压下真红灯"。
#:
#: 顺序：
#:   1. UNTRUSTED_DESCRIPTION —— 描述本身不可信，所有结论失去根基（FR6）
#:   2. OVERFLOW/DEADLOCK/MISMATCH —— 已确证的具体问题，不被别处缺口掩盖
#:   3. COVERAGE_GAP —— 没发现问题但也没验全：禁止干净结论，却不盖过真问题
#:   4. OK
_BLOCKERS = (Verdict.OVERFLOW, Verdict.DEADLOCK, Verdict.MISMATCH)


def aggregate(results: list[ToolResult]) -> Verdict:
    """聚合多检查结论。

    - 任一 `UNTRUSTED_DESCRIPTION` → 它（描述不可信，结论失去根基）；
    - 否则任一具体问题（溢出/死锁/不匹配）→ 取该问题（**不被别处缺口掩盖**）；
    - 否则任一 `COVERAGE_GAP` → 它（没验全，禁止给干净 OK）；
    - 全 `OK` → `OK`；
    - 空列表 → `COVERAGE_GAP`（一项都没跑就说"没问题"是自欺）。
    """
    if not results:
        return Verdict.COVERAGE_GAP
    seen = {r.verdict for r in results}

    if Verdict.UNTRUSTED_DESCRIPTION in seen:
        return Verdict.UNTRUSTED_DESCRIPTION

    for blocker in _BLOCKERS:
        if blocker in seen:
            return blocker

    if Verdict.COVERAGE_GAP in seen:
        return Verdict.COVERAGE_GAP
    return Verdict.OK


class RunReport:
    """一次 run 的完整结论。"""

    __slots__ = ("results", "verdict")

    def __init__(self, results: list[ToolResult]) -> None:
        self.results = results
        self.verdict = aggregate(results)

    @property
    def exit_code(self) -> int:
        """沿用既有的 verdict → 退出码映射，不新造一套。"""
        return verdict_exit_code(self.verdict)

    def render(self) -> str:
        lines = [f"[run] 综合结论：{self.verdict.value}"]
        for r in self.results:
            lines.append(f"  {r.verdict.value:<22} {r.tool}")
            for d in r.diagnostics:
                if d.severity == "error":
                    lines.append(f"      error   {d.render()}")
                elif d.severity == "warning":
                    lines.append(f"      warning {d.render()}")
                elif d.rule.startswith("run/"):
                    lines.append(f"      skip    {d.message}")

        gaps = [r for r in self.results if r.verdict is Verdict.COVERAGE_GAP]
        if gaps and self.verdict is Verdict.COVERAGE_GAP:
            lines.append("")
            lines.append(
                f"注意：{len(gaps)}/{len(self.results)} 项未能完成验证。"
                "「没发现问题」不等于「没有问题」"
            )
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "exit_code": self.exit_code,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "checks": [r.to_json() for r in self.results],
        }

    def write_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _skipped(tool: str, spec_hash: str, message: str) -> ToolResult:
    """构造一条"因条件不满足而未跑"的缺口记录。

    刻意用 `COVERAGE_GAP` 而非把这项从报告里删掉：删掉会让报告显得全绿，
    而 `COVERAGE_GAP` 会把整体结论也拉成缺口——这正是我们要的。
    """
    return ToolResult(
        tool=tool,
        verdict=Verdict.COVERAGE_GAP,
        spec_hash=spec_hash,
        engine_version=ORCHESTRATOR_VERSION,
        trust="n/a",
        diagnostics=[Finding(severity="info", message=message, rule="run/skipped")],
    )


def run_checks(
    module: VModule,
    config: dict[str, Any],
    spec_hash: str,
    *,
    anchor: VModule | None = None,
    bound: int | None = None,
    mode: str = "concrete",
) -> RunReport:
    """跑全部适用检查。

    `mode` 只影响等价验证：`concrete` 跑具体档，`symbolic` 跑符号档。两档
    结论并列而非替代（M4 卡 §4.1），故不默认两个都跑——那会让报告出现两条
    强度不同却长得一样的结论。
    """
    from hivm_spec.assemble import run_tool

    results = [
        run_tool(name, config, module, spec_hash)
        for name in ("ub_occupancy", "timeline", "sync_pairing", "uninit_read")
    ]

    label = "equivalence" if mode == "concrete" else "equivalence(symbolic)"
    if anchor is None:
        results.append(
            _skipped(
                label,
                spec_hash,
                "未提供 --anchor，等价验证未执行——本工具不做自比"
                "（同一份 IR 自比恒为 OK 却什么都没验证）",
            )
        )
    elif mode == "symbolic" and not _z3_ready():
        results.append(
            _skipped(
                label,
                spec_hash,
                "z3 不可用，符号等价未执行——具体档结论不能替代符号档"
                "（安装：pip install -e '.[symbolic]'）",
            )
        )
    else:
        results.append(
            run_tool(
                "equivalence", config, module, spec_hash, anchor=anchor, bound=bound, mode=mode
            )
        )

    return RunReport(results)


def _z3_ready() -> bool:
    from hivm_spec.symbolic import z3_available

    return z3_available()
