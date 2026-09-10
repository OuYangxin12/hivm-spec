"""环境自检（`hivm-spec doctor`）。

## 为什么值得一个子命令

本工具的运行环境有四个易错点：IR 接口层要 **cp310 ABI**（换个 Python 版本就
不行）、bindings 树的位置由 `HIVM_SPEC_BINDINGS` 指定、`ml_dtypes` 与 `z3` 是
两个**可选** extra。此前诊断这些全靠翻 README 与 docs，出错时的现象又往往是
"某些测试静默跳过"或"工具报了个看不懂的缺口"。

## 纪律：不谎报可用

每一项探测**只报实测结果**，缺就是缺。特别地：

- 缺 `ml_dtypes` **不**说"可用 f32 替代"——f32 算 bf16 会给出比真实硬件更精确
  的结果，对拍会假通过（见 `numeric.py`）；
- 缺 z3 **不**说"可退回具体档"——两档结论强度不同，不可互相替代（M4 卡 §4.1）。

这与 `COVERAGE_GAP` 是同一条原则：**缺口是合法结论，静默不是**。
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import dataclass
from enum import Enum

__all__ = ["CheckStatus", "DoctorReport", "ProbeResult", "diagnose"]


class CheckStatus(Enum):
    """一项探测的结果。"""

    OK = "ok"
    #: 缺失但只影响部分能力——工具仍可用，能力打折
    DEGRADED = "degraded"
    #: 缺失且核心能力不可用
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    name: str
    status: CheckStatus
    detail: str
    #: 缺失时怎么修（缺就必须给，否则诊断没有行动价值——同 FR5）
    remedy: str = ""
    #: 受影响的能力，用于说明"缺了它会怎样"
    affects: str = ""


@dataclass(frozen=True, slots=True)
class DoctorReport:
    probes: tuple[ProbeResult, ...]

    @property
    def usable(self) -> int:
        return sum(1 for p in self.probes if p.status is CheckStatus.OK)

    @property
    def has_blocker(self) -> bool:
        return any(p.status is CheckStatus.MISSING for p in self.probes)

    def render(self) -> str:
        marks = {CheckStatus.OK: "✓", CheckStatus.DEGRADED: "!", CheckStatus.MISSING: "✗"}
        lines = []
        for p in self.probes:
            lines.append(f"  {marks[p.status]} {p.name:<22} {p.detail}")
            if p.status is not CheckStatus.OK:
                if p.affects:
                    lines.append(f"      影响：{p.affects}")
                if p.remedy:
                    lines.append(f"      修复：{p.remedy}")

        degraded = [p for p in self.probes if p.status is CheckStatus.DEGRADED]
        missing = [p for p in self.probes if p.status is CheckStatus.MISSING]

        lines.append("")
        if missing:
            lines.append(
                f"结论：{len(missing)} 项核心能力不可用"
                f"（{', '.join(p.name for p in missing)}）——请先按上述修复"
            )
        elif degraded:
            lines.append(
                f"结论：核心可用，{len(degraded)} 项能力受限"
                f"（{', '.join(p.name for p in degraded)}）"
            )
        else:
            lines.append("结论：全部可用")
        return "\n".join(lines)


def _probe_python() -> ProbeResult:
    """核心层要 >=3.10；IR 接口层要**恰好** cp310（bindings 是 cp310 ABI）。"""
    v = sys.version_info
    ver = platform.python_version()
    if v[:2] < (3, 10):
        return ProbeResult(
            name="Python",
            status=CheckStatus.MISSING,
            detail=f"{ver}（低于最低要求 3.10）",
            affects="全部功能",
            remedy="升级到 Python 3.10+",
        )
    if v[:2] != (3, 10):
        return ProbeResult(
            name="Python",
            status=CheckStatus.DEGRADED,
            detail=f"{ver}（核心层可用；IR 接口层需 cp310）",
            affects="tool/run 命令——bindings 是 cp310 ABI，本解释器加载不了",
            remedy="用 3.10 解释器跑 tool/run：uv venv .venv310 --python 3.10",
        )
    return ProbeResult(
        name="Python", status=CheckStatus.OK, detail=f"{ver}（cp310，IR 接口层可用）"
    )


def _probe_bindings() -> ProbeResult:
    """委托给 `hivm_spec.bindings`，不自己再写一份探测逻辑。

    两处独立的探测必然漂移（`tests/conftest.py` 里有同样的注释）。
    """
    env = os.environ.get("HIVM_SPEC_BINDINGS", "")
    hint = f"HIVM_SPEC_BINDINGS={env}" if env else "HIVM_SPEC_BINDINGS 未设"
    try:
        from hivm_spec.bindings import bindings_available

        ok = bindings_available()
    except Exception as exc:  # pragma: no cover - 防御性
        return ProbeResult(
            name="bishengir bindings",
            status=CheckStatus.MISSING,
            detail=f"探测失败：{exc}",
            affects="全部 tool/run 命令（无法解析 MLIR）",
            remedy="bash scripts/setup_bindings.sh",
        )
    if not ok:
        return ProbeResult(
            name="bishengir bindings",
            status=CheckStatus.MISSING,
            detail=f"不可用（{hint}）",
            affects="全部 tool/run 命令——无法把 MLIR 降级为 VIR",
            remedy="bash scripts/setup_bindings.sh && export HIVM_SPEC_BINDINGS=$PWD/.bindings",
        )
    return ProbeResult(name="bishengir bindings", status=CheckStatus.OK, detail=f"可用（{hint}）")


def _probe_dist(
    module: str,
    label: str,
    *,
    required: bool,
    affects: str,
    remedy: str,
    dist: str = "",
) -> ProbeResult:
    """`dist` 是 PyPI 包名——**未必等于 import 名**（z3 的包名是 z3-solver）。"""
    if importlib.util.find_spec(module) is None:
        return ProbeResult(
            name=label,
            status=CheckStatus.MISSING if required else CheckStatus.DEGRADED,
            detail="未安装",
            affects=affects,
            remedy=remedy,
        )
    try:
        from importlib.metadata import version

        ver = version(dist or module)
    except Exception:  # pragma: no cover - 版本读不到不影响可用性判断
        ver = "版本未知"
    return ProbeResult(name=label, status=CheckStatus.OK, detail=ver)


def diagnose() -> DoctorReport:
    """跑全部探测。"""
    return DoctorReport(
        probes=(
            _probe_python(),
            _probe_bindings(),
            _probe_dist(
                "numpy",
                "numpy",
                required=True,
                affects="具体档执行（等价验证、未初始化读）",
                remedy="pip install -e .",
            ),
            _probe_dist(
                "ml_dtypes",
                "ml_dtypes",
                required=False,
                # 不说"可用 f32 替代"：f32 算 bf16 比真实硬件更精确，对拍会假通过
                affects="bf16/fp8 等非标准 dtype——缺失时相关用例报缺口，不会退化为 f32 静默通过",
                remedy="pip install -e '.[numeric]'",
            ),
            _probe_dist(
                "z3",
                "z3",
                dist="z3-solver",
                required=False,
                # 不说"可退回具体档"：两档结论强度不同，不可互相替代
                affects=(
                    "符号档（equivalence --mode symbolic）；具体档不受影响，但二者结论不可互相替代"
                ),
                remedy="pip install -e '.[symbolic]'",
            ),
        )
    )
