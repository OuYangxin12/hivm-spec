"""占用图文本渲染器（T1.5）：让结论**可被人类一眼读懂**。

JSON 是 agent 的消费通道；本模块服务另一侧——终端里的人。设计约束：

1. **确定性**：同一结果两次渲染逐字节一致（FR8 延伸到视图层）；
2. **定宽 ≤80 列**：终端不折行，agent 抓取 stdout 也不会被撑爆；
3. **溢出视觉不可隐藏**：超容量时必须出现 `!` 标记与超限数字——渲染层
   不得把坏消息画得不显眼（FR6 的视图侧延伸）；
4. 峰值/容量比例条 + sparkline 时间线 + 贡献者排行，三层信息各有分工。
"""

from __future__ import annotations

from typing import Any

__all__ = ["render_occupancy_chart"]

_WIDTH = 52
#: 8 级 sparkline 块字符（U+2581..U+2588），空格表示零值
_BLOCKS = " ▁▂▃▄▅▆▇█"


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    if n >= 1024 * 1024:
        v = n / (1024 * 1024)
        return f"{v:.1f}MB" if v >= 10 else f"{v:.2f}MB"
    if n >= 1024:
        v = n / 1024
        return f"{v:.1f}KB" if v >= 10 else f"{v:.2f}KB"
    return f"{n}B"


def _ratio_bar(peak: int, capacity: int | None, width: int = _WIDTH) -> str:
    """峰值 vs 容量的单条比例条。溢出时满格加 `!`。"""
    if capacity is None or capacity <= 0:
        return "[" + "·" * width + "]（容量未知，无法画容量线）"
    frac = min(peak / capacity, 1.0)
    filled = round(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    if peak > capacity:
        bar = bar[:-1] + "!"
    return f"[{bar}]"


def _sparkline(curve: list[tuple[int, int]], width: int = _WIDTH) -> str:
    """占用曲线 → 定宽 sparkline。节点数超宽时按桶取峰值（不失真地保守）。"""
    if not curve:
        return "（无时间线：无节点）"
    n = len(curve)
    if n <= width:
        values = [v for _, v in curve]
        scale = max(values) or 1
        return "".join(_BLOCKS[min(int(v / scale * 8), 8)] for v in values)
    # 分桶取**桶内峰值**：取均值会把真实的瞬时高峰抹平
    bucket = n / width
    values = []
    for c in range(width):
        lo, hi = int(c * bucket), max(int((c + 1) * bucket), int(c * bucket) + 1)
        values.append(max(v for _, v in curve[lo:hi]))
    scale = max(values) or 1
    return "".join(_BLOCKS[min(int(v / scale * 8), 8)] for v in values)


def render_occupancy_chart(details: dict[str, Any]) -> str:
    """从 ToolResult.details 渲染文本占用图。

    仅依赖 details 的已序列化字段（而非内部对象），渲染与结论契约解耦。
    """
    spaces: dict[str, Any] = details.get("spaces", {})
    if not spaces:
        return ""

    lines: list[str] = ["", "占用图："]
    for name in sorted(spaces):
        sp = spaces[name]
        cap, peak = sp.get("capacity"), sp.get("peak_bytes", 0)
        util = sp.get("utilization")
        lines.append(f"  {name}（容量 {_fmt_bytes(cap)}）")
        lines.append(
            f"    峰值 {_fmt_bytes(peak):>9s} {_ratio_bar(peak, cap)}"
            f" {f'{util:.0%}' if util is not None else ''}".rstrip()
        )
        curve = sp.get("curve", [])
        if curve:
            n_pts = len(curve)
            lines.append(f"    时间线（节点 0→{n_pts - 1}，{n_pts} 点）")
            lines.append(f"    {_sparkline(curve)}")
        contribs = sp.get("contributors", [])
        if contribs:
            lines.append("    贡献者（按尺寸降序，区间=节点序号）：")
            for c in contribs[:5]:
                iv = c.get("interval", [0, 0])
                mb = c.get("multi_buffer", 1)
                mb_note = f" ×multi_buffer{mb}" if mb > 1 else ""
                lines.append(
                    f"      {c['name']}({_fmt_bytes(c.get('nbytes'))}{mb_note})"
                    f" 活跃 [{iv[0]}, {iv[1]}]"
                )
        unsized = sp.get("unsized_buffers", 0)
        if unsized:
            lines.append(f"    ⚠ {unsized} 个 buffer 尺寸未知，未计入上述峰值（峰值为下界）")
    return "\n".join(lines)
