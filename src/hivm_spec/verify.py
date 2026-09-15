"""一键验证入口（`hivm-spec verify`）。

## 为什么值得一个子命令

`run` 已经很省事了，但它把三件事留给调用方自己串：先确认环境能不能跑、跑完
把缺口从输出里读出来、以及**别忘了自己其实没能验全部**。agent 在回路里串这三步
时最常见的失败不是漏跑，而是**把"跑了五项"读成"验了五项"**。

`verify` = `doctor` 前置门 + 全套检查 + **能力自述**。它不新增任何判定逻辑
（判定仍只在 `assemble`/各引擎里），只是把"环境限制"与"验证结论"绑成同一条
输出——这正是本仓的立身原则：**缺口是合法结论，静默不是**（FR7）。

## 三条纪律

1. **前置门挡住时不产出验证结论**。绑定缺失 → 退出码 2 + 修复指引，而**不是**
   `COVERAGE_GAP`(4)：后者说"这份 IR 我没能验全"，前者说"我根本跑不起来"。
   把二者混成一个码，就等于把环境故障伪装成被验对象的属性（FR7 明令区分）。
2. **能力自述恒在**。每条结论都附"本次实际执行了哪些检查、哪些没执行、为什么"，
   避免降级运行被当成完整验证。
3. **退出码沿用既有映射**（`verdict.verdict_exit_code`），不新造一套。

依赖方向：本模块**不得** import bishengir（D7）；一切 IR 相关操作由 `__main__`
以已降好的 VIR 传入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hivm_spec.doctor import CheckStatus, DoctorReport
from hivm_spec.run_checks import RunReport
from hivm_spec.verdict import Verdict

__all__ = ["VERIFY_VERSION", "VerifyReport"]

#: 编排层版本——与 `orchestrator_version` 同族，进入审计坐标。
VERIFY_VERSION = "0.1.0"

#: verdict → 一句话行动指向（agent 可直接转述）
_ACTION: dict[str, str] = {
    Verdict.OK.value: "验过，已覆盖范围内未发现问题",
    Verdict.OVERFLOW.value: "片上内存溢出：按诊断的贡献者列表改",
    Verdict.DEADLOCK.value: "结构性死锁：按首现 wait 位置改同步",
    Verdict.MISMATCH.value: "语义发散：修首个发散点，别修下游",
    Verdict.COVERAGE_GAP.value: "没验成：补描述/补锚点，不得当作通过",
    Verdict.UNTRUSTED_DESCRIPTION.value: "判定依据不可信：先修描述或登记 drift",
}

#: "没验成"的类属——与 `Verdict.is_gap` 同源，转述时不得与"发现问题"混用
_GAP_VERDICTS = (Verdict.COVERAGE_GAP.value, Verdict.UNTRUSTED_DESCRIPTION.value)


@dataclass(slots=True)
class VerifyReport:
    """`verify` 的一次完整产出：结论 + 能力自述。

    `run` 的结论分量原样保留（`run_report`），本类只加"这次到底跑没跑起来、
    跑了几项、哪些被降级挡在外面"。**不改写任何子结论**。
    """

    run_report: RunReport
    doctor: DoctorReport
    mode: str
    input_ir: str
    config_path: str = ""
    anchor: str | None = None

    # -- 派生视图 ---------------------------------------------------------

    @property
    def executed(self) -> list[str]:
        """实际执行了的检查（含跑了但记缺口的——执行过就要出现在账上）。"""
        return [r.tool for r in self.run_report.results]

    @property
    def gaps(self) -> list[str]:
        """没验成的检查。空列表才是"这轮验全了"。"""
        return [r.tool for r in self.run_report.results if r.verdict.is_gap]

    @property
    def problems(self) -> list[str]:
        return [r.tool for r in self.run_report.results if r.verdict.is_problem]

    @property
    def trust(self) -> str:
        """结论里出现过的信任级别（现网全部 provisional）。

        被跳过的检查其 trust 是 `n/a`（根本没跑，谈不上信任级别）——不计入，
        否则汇报里会出现 "n/a, provisional" 这种读起来像混合信任源的字符串。
        """
        levels = {r.trust for r in self.run_report.results if r.trust not in ("", "n/a")}
        return ", ".join(sorted(levels)) or "n/a（本轮无实际执行的判定）"

    @property
    def limitations(self) -> list[str]:
        """本轮的能力限制——来自 doctor 的非 OK 探测项。"""
        out: list[str] = []
        for p in self.doctor.probes:
            if p.status is CheckStatus.OK:
                continue
            out.append(f"{p.name}：{p.detail}" + (f"（影响：{p.affects}）" if p.affects else ""))
        return out

    # -- 渲染 -------------------------------------------------------------

    def summary(self) -> str:
        """一行综合结论。agent 的默认汇报应当以它开头。"""
        v = self.run_report.verdict.value
        return (
            f"{v}｜执行 {len(self.executed)} 项，其中 {len(self.gaps)} 项没验成｜信任 {self.trust}"
        )

    def render(self) -> str:
        lines = [f"[verify] {self.summary()}"]
        if self.anchor is None:
            lines.append(
                "  ⚠️ 未提供 --anchor：等价验证未执行。本轮结论不含「变换前后是否等效」这一维度"
            )
        lines.append("")
        lines.append("  逐项：")
        for r in self.run_report.results:
            action = _ACTION.get(r.verdict.value, "")
            lines.append(f"    {r.verdict.value:<22} {r.tool:<22} {action}")
            for d in r.diagnostics:
                if d.severity in ("error", "warning"):
                    lines.append(f"        {d.render()}")

        if self.gaps:
            lines.append("")
            lines.append(
                f"  没验成：{', '.join(self.gaps)} ——"
                "「没发现问题」不等于「没有问题」，汇报时不得省略"
            )
        if self.limitations:
            lines.append("")
            lines.append("  本轮能力限制（doctor 实测）：")
            for lim in self.limitations:
                lines.append(f"    ! {lim}")
        lines.append("")
        lines.append("  审计坐标见 JSON；跨版本对比前先核对 spec_hash 与 engine_version。")
        return "\n".join(lines)

    # -- 序列化 -----------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        run = self.run_report.to_json()
        run["capabilities"] = {
            "probes": [
                {
                    "name": p.name,
                    "status": p.status.value,
                    "detail": p.detail,
                    **({"affects": p.affects} if p.affects else {}),
                    **({"remedy": p.remedy} if p.remedy else {}),
                }
                for p in self.doctor.probes
            ],
            "mode": self.mode,
            "input_ir": self.input_ir,
            "anchor": self.anchor,
            "config": self.config_path,
            "executed": self.executed,
            "gaps": self.gaps,
            "limitations": self.limitations,
            "trust": self.trust,
            "self_description": (
                f"本轮执行 {len(self.executed)} 项检查，"
                f"{len(self.gaps)} 项未能完成验证；"
                f"信任级别 {self.trust}，未经对拍验证的结论不可作为合入门禁依据"
            ),
        }
        run["verify_version"] = VERIFY_VERSION
        return run

    def write_json(self, path: str) -> None:
        from json import dumps
        from pathlib import Path

        Path(path).write_text(
            dumps(self.to_json(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
