"""统一结论契约（T1.6，框架 §7.3）。

**verdict 是封闭枚举**，且 `COVERAGE_GAP` / `UNTRUSTED_DESCRIPTION` 是**合法结论**
而非错误（FR4/FR7）。这一点是本项目的立身之本：一个诚实地说"我看不懂这段 IR"
的工具，比一个蒙对的工具有用得多。

**结论的完整审计坐标** = `spec_hash` + `engine_version` + 输入 IR 指纹。
三者缺一，历史结论就不可复现（FR8）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hivm_spec.vir import Loc

__all__ = [
    "Finding",
    "ToolResult",
    "Verdict",
    "verdict_exit_code",
]


class Verdict(Enum):
    """封闭结论枚举（框架 §7.3）。

    优先级（高→低）：`UNTRUSTED_DESCRIPTION` > `COVERAGE_GAP` > 具体问题 > `OK`。
    理由：描述本身不可信时，后续一切分析都无意义；覆盖不全时，"没发现问题"
    不能被读作"没有问题"。
    """

    OK = "OK"
    OVERFLOW = "OVERFLOW"
    DEADLOCK = "DEADLOCK"
    MISMATCH = "MISMATCH"
    COVERAGE_GAP = "COVERAGE_GAP"
    UNTRUSTED_DESCRIPTION = "UNTRUSTED_DESCRIPTION"

    @property
    def is_problem(self) -> bool:
        """是否表示"发现了被验证对象的问题"。

        缺口类 verdict **不算**发现问题——它们说的是"我没能完成验证"。
        混淆二者会让 agent 把"看不懂"当成"没问题"（或反之）。
        """
        return self in (Verdict.OVERFLOW, Verdict.DEADLOCK, Verdict.MISMATCH)

    @property
    def is_gap(self) -> bool:
        return self in (Verdict.COVERAGE_GAP, Verdict.UNTRUSTED_DESCRIPTION)


#: verdict → 进程退出码。
#: 缺口与问题用**不同**退出码：脚本需要能区分"验证失败"与"没验成"。
_EXIT_CODES = {
    Verdict.OK: 0,
    Verdict.OVERFLOW: 1,
    Verdict.DEADLOCK: 1,
    Verdict.MISMATCH: 1,
    Verdict.COVERAGE_GAP: 4,
    Verdict.UNTRUSTED_DESCRIPTION: 5,
}


def verdict_exit_code(v: Verdict) -> int:
    return _EXIT_CODES[v]


@dataclass(frozen=True, slots=True)
class Finding:
    """一条诊断。

    `loc` 非可选是刻意的：FR5 要求结论可定位。一条无法定位的诊断对 agent
    几乎无用——它无法据此修改代码。
    """

    severity: str  # "error" | "warning" | "info"
    message: str
    loc: Loc | None = None
    #: 触发该诊断的规则/检查标识，便于回溯到实现
    rule: str = ""
    #: 结构化补充（如溢出的贡献者列表）
    extra: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        where = f" @ {self.loc.describe()}" if self.loc else ""
        rule = f" [{self.rule}]" if self.rule else ""
        return f"{self.severity}{rule}{where}: {self.message}"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "severity": self.severity,
            "message": self.message,
            "rule": self.rule,
        }
        if self.loc is not None:
            out["loc"] = self.loc.describe()
        if self.extra:
            out["extra"] = self.extra
        return out


@dataclass
class ToolResult:
    """工具输出（框架 §7.3 统一契约）。"""

    tool: str
    verdict: Verdict
    spec_hash: str
    engine_version: str
    trust: str
    #: 输入 IR 的 VIR 指纹——审计坐标的第三个分量
    ir_fingerprint: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Finding] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "verdict": self.verdict.value,
            "spec_hash": self.spec_hash,
            "engine_version": self.engine_version,
            "trust": self.trust,
            "ir_fingerprint": self.ir_fingerprint,
            "details": self.details,
            "diagnostics": [d.to_json() for d in self.diagnostics],
        }

    def to_json_bytes(self) -> bytes:
        """确定性 JSON 字节串（FR8）。"""
        return (
            json.dumps(self.to_json(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    @property
    def exit_code(self) -> int:
        return verdict_exit_code(self.verdict)

    def render(self) -> str:
        """人类可读摘要。

        **信任降级必须出现在结论旁**：描述为 provisional 时，一切结论都不能
        看起来权威（FR6）。这是"图画出来了就显得可信"这一认知陷阱的对策。
        """
        lines = [f"[{self.tool}] {self.verdict.value}"]
        if self.trust != "anchored":
            lines.append(
                f"  ⚠️ 信任级别：{self.trust} —— 本结论基于**未经对拍验证**的描述，不可作为最终依据"
            )
        for d in self.diagnostics:
            lines.append(f"  {d.render()}")
        lines.append(
            f"  审计坐标：spec_hash={self.spec_hash[:19]}… "
            f"engine={self.engine_version} ir={self.ir_fingerprint[:16]}…"
        )
        return "\n".join(lines)
