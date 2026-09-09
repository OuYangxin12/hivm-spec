"""占用图文本渲染器（T1.5）：让结论**可被人类一眼读懂**。

JSON 是 agent 的消费通道；本模块服务另一侧——终端里的人。设计约束：

1. **确定性**：同一结果两次渲染逐字节一致（FR8 延伸到视图层）；
2. **定宽 ≤80 列**：终端不折行，agent 抓取 stdout 也不会被撑爆；
3. **溢出视觉不可隐藏**：超容量时必须出现 `!` 标记与超限数字——渲染层
   不得把坏消息画得不显眼（FR6 的视图侧延伸）；
4. 峰值/容量比例条 + sparkline 时间线 + 贡献者排行，三层信息各有分工。
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["chrome_trace_json", "render_occupancy_chart", "render_timeline_chart"]

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


# ---------------------------------------------------------------------------
# 时序图渲染（T2.4）：文本甘特 + Chrome Trace Event Format
# ---------------------------------------------------------------------------

#: 甘特图列宽（≤80 列约束下留出行头空间）
_GANTT_WIDTH = 56

#: 步骤类别 → 甘特字符。受阻的 wait 用 `×`——坏消息在视图层同样不可隐藏。
_KIND_MARK = {
    "op": "·",
    "set_flag": "▲",
    "sync_block_set": "▲",
    "wait_flag": "▽",
    "sync_block_wait": "▽",
    "pipe_barrier": "≡",
}
_BLOCKED_MARK = "×"
#: 文本行硬上限（设计约束 2：终端不折行）
_MAX_LINE = 80


def _clip(line: str, limit: int = _MAX_LINE) -> str:
    """把摘要行收进行宽，超长时以 `…` 显式提示有下文（截断不得静默）。"""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def render_timeline_chart(details: dict[str, Any]) -> str:
    """从 ToolResult.details 渲染文本甘特图（主策略 = sequential）。

    行 = 泳道（pipe，按 vm.pipes 声明序，未声明的排后），列 = 执行步
    （事件步单位，超宽按桶收拢——桶内**受阻 wait 优先保留**，坏消息不丢）。
    仅依赖 details 已序列化字段，与结论契约解耦；确定性输出（FR8）。
    """
    timelines: dict[str, Any] = details.get("timelines", {})
    if not timelines:
        return ""
    primary = timelines.get("sequential") or next(iter(timelines.values()))
    steps: list[dict[str, Any]] = primary.get("steps", [])
    if not steps:
        return ""

    lane_order: list[str] = list(details.get("pipe_order", []))
    for st in steps:
        if st["lane"] not in lane_order:
            lane_order.append(st["lane"])

    blocked_seqs = {int(s) for s in primary.get("blocked", [])}
    total = max(int(st["pos"]) for st in steps) + 1

    def _rank(st: dict[str, Any]) -> int:
        """桶内代表步骤的优先级：受阻 wait > 同步 > barrier > 普通 op。"""
        if int(st["seq"]) in blocked_seqs:
            return 3
        if st["kind"] in ("set_flag", "sync_block_set", "wait_flag", "sync_block_wait"):
            return 2
        if st["kind"] == "pipe_barrier":
            return 1
        return 0

    # 列 = 执行位次（与 Chrome Trace 的 ts 同口径）；每列每泳道取一个代表步骤，
    # 桶内受阻 wait 优先保留（坏消息不丢）
    grid: dict[int, dict[str, dict[str, Any]]] = {}
    if total <= _GANTT_WIDTH:
        for st in steps:
            grid.setdefault(int(st["pos"]), {})[st["lane"]] = st
    else:
        for st in steps:
            col = int(st["pos"]) * _GANTT_WIDTH // total
            lane_cells = grid.setdefault(col, {})
            cur = lane_cells.get(st["lane"])
            if cur is None or _rank(st) > _rank(cur):
                lane_cells[st["lane"]] = st

    lines: list[str] = ["", f"时序图（{primary.get('strategy', 'sequential')}，事件步）："]
    width = _GANTT_WIDTH if total > _GANTT_WIDTH else total
    for lane in lane_order:
        cells = ""
        for col in range(width):
            cell = grid.get(col, {}).get(lane)
            if cell is None:
                cells += " "
            elif int(cell["seq"]) in blocked_seqs:
                cells += _BLOCKED_MARK
            else:
                cells += _KIND_MARK.get(cell["kind"], "·")
        lines.append(f"  {lane:<12s} {cells}")
    lines.append("  图例：· op　▲ set　▽ wait　≡ barrier　× 受阻 wait")
    for note in details.get("truncation_notes", [])[:2]:
        lines.append(_clip(f"  ⚠ {note}"))
    deadlocks = details.get("deadlocks", [])
    if deadlocks:
        d0 = deadlocks[0]
        # 摘要行必须收进行宽（设计约束 2）：完整 message 在 diagnostics/JSON 里，
        # 视图层只负责"看得见"，不负责"看得全"——但省略号必须显式提示有下文。
        lines.append(_clip(f"  ✗ 死锁：{d0['message']} @ {d0['loc']}"))
    return "\n".join(lines)


def chrome_trace_json(details: dict[str, Any], *, max_events: int = 20000) -> str:
    """把各策略时间线转为 Chrome Trace Event Format（perfetto 可导入）。

    时间轴用"事件步"而非时钟（M2 无耗时模型）：ts = 执行位次，dur = 1。
    pid = 策略名，tid = 泳道序号。确定性序列化（FR8）。
    """
    timelines: dict[str, Any] = details.get("timelines", {})
    pipe_order: list[str] = list(details.get("pipe_order", []))
    for tl in timelines.values():
        for st in tl.get("steps", []):
            if st["lane"] not in pipe_order:
                pipe_order.append(st["lane"])
    tid_of = {lane: i for i, lane in enumerate(pipe_order)}

    events: list[dict[str, Any]] = []
    for name in sorted(timelines):
        tl = timelines[name]
        blocked = {int(s) for s in tl.get("blocked", [])}
        events.append({"ph": "M", "name": "process_name", "pid": name, "args": {"name": name}})
        for st in tl.get("steps", []):
            if len(events) >= max_events:
                break
            ev = st.get("event")
            events.append(
                {
                    "ph": "X",
                    "name": st["op"],
                    "cat": st["kind"],
                    "pid": name,
                    "tid": tid_of.get(st["lane"], 0),
                    "ts": int(st["pos"]),
                    "dur": 1,
                    "args": {
                        "seq": st["seq"],
                        "iter": st.get("iter", ""),
                        "event": ev,
                        "blocked": int(st["seq"]) in blocked,
                        "loc": st.get("loc", ""),
                    },
                }
            )
    return json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}, ensure_ascii=False)
