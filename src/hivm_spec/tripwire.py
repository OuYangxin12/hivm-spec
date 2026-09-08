"""账本绊线（T0.8）：已注册 op vs 描述账本的覆盖报告。

**为什么叫"绊线"**：它不是为了通过，而是为了在覆盖率退步或主仓新增 op 时**绊一下**，
让缺口浮出水面。绊线的输出是**报告**而非判定——覆盖率低不是错误（M0 只有 9 个 op
建模是预期的），但"覆盖率无人知晓"是错误。

对应 FR4：未建模的 op 必须可见。这里是描述侧的可见性（编译期），
`vir.Coverage` 是运行侧的可见性（分析期）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CoverageReport",
    "OpRegistry",
    "load_registry",
    "report_coverage",
]

REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "specs" / "registry" / "hivm_ops.json"
)

#: 分类前缀 → 人类可读的分组名，便于按子系统排优先级。
GROUP_PREFIXES = {
    "hivm.hir.": "hir（高层 op）",
    "hivm.lir.": "lir（低层 op）",
    "hivm.": "其他",
}


@dataclass(frozen=True, slots=True)
class OpRegistry:
    """主仓已注册 op 的快照。"""

    ops: tuple[str, ...]
    source_commit: str = ""
    extraction: str = ""

    @property
    def count(self) -> int:
        return len(self.ops)

    def group_of(self, op: str) -> str:
        for prefix, label in GROUP_PREFIXES.items():
            if op.startswith(prefix):
                return label
        return "未分类"


def load_registry(path: Path | None = None) -> OpRegistry:
    p = path or REGISTRY_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"op 注册表快照缺失：{p}——绊线无基准可比。重新抓取方式见该文件的 extraction 字段。"
        )
    with p.open(encoding="utf-8") as fh:
        doc: dict[str, Any] = json.load(fh)
    return OpRegistry(
        ops=tuple(doc["ops"]),
        source_commit=doc.get("source_commit", ""),
        extraction=doc.get("extraction", ""),
    )


@dataclass
class CoverageReport:
    """描述覆盖报告。"""

    total: int
    modeled: tuple[str, ...] = ()
    unmodeled: tuple[str, ...] = ()
    #: 描述里声明了但主仓注册表中不存在的 op —— 强信号：拼写错误或主仓已删除
    unknown_to_upstream: tuple[str, ...] = ()
    by_group: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def coverage_ratio(self) -> float:
        return len(self.modeled) / self.total if self.total else 0.0

    def render(self) -> str:
        lines = [
            f"账本绊线：主仓已注册 {self.total} 个 op，"
            f"已建模 {len(self.modeled)}（{self.coverage_ratio:.1%}）"
        ]
        for group, (modeled, total) in sorted(self.by_group.items()):
            lines.append(f"  {group}：{modeled}/{total}")
        if self.unknown_to_upstream:
            lines.append(
                f"  ⚠️ 描述中存在但主仓未注册的 op：{list(self.unknown_to_upstream)}"
                "——可能是拼写错误，或主仓已删除该 op（后者须按 D9 登记漂移）"
            )
        if self.unmodeled:
            lines.append(f"  未建模 {len(self.unmodeled)} 个（前 10）：{list(self.unmodeled[:10])}")
        return "\n".join(lines)


def report_coverage(
    modeled_ops: set[str] | frozenset[str], registry: OpRegistry | None = None
) -> CoverageReport:
    reg = registry or load_registry()
    registered = set(reg.ops)

    modeled = tuple(sorted(modeled_ops & registered))
    unmodeled = tuple(sorted(registered - set(modeled_ops)))
    unknown = tuple(sorted(set(modeled_ops) - registered))

    by_group: dict[str, tuple[int, int]] = {}
    for op in reg.ops:
        g = reg.group_of(op)
        m, t = by_group.get(g, (0, 0))
        by_group[g] = (m + (1 if op in modeled_ops else 0), t + 1)

    return CoverageReport(
        total=reg.count,
        modeled=modeled,
        unmodeled=unmodeled,
        unknown_to_upstream=unknown,
        by_group=by_group,
    )
