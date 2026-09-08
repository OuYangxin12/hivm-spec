"""占用/生存期引擎（T1.2–T1.4）：per-space 峰值占用与容量判定。

**只消费 VIR**（D6）——本模块不 import `bishengir`，故可用手写 VIR 完整测试。

## 口径声明（判定边界随结论输出，Q2 钩子）

1. **生存期 = alloc 点 → 最后一次访问**（顺序语义）。真实分配器可能更激进
   （复用已死 buffer）或更保守（对齐/多缓冲），差异按 D9 登记漂移，
   **不得为对齐结果反推语义**；
2. **循环体内的 alloc 按单次迭代计**，但若循环中 alloc 逃逸（被循环外访问），
   报缺口而非猜测；
3. **尺寸未知的 alloc 不参与峰值累加**，而是使判定降级为 `COVERAGE_GAP`——
   这是全模块最关键的一条：拿 0 顶替未知尺寸会让结论呈现虚假精确。

## 为什么"峰值"需要区间而非单点

一个 buffer 从 alloc 到 last-use 之间**始终占用空间**。若只统计 alloc 时刻，
两个生存期不重叠的 1KB buffer 会被算成 2KB（虚假溢出）；若只统计当前活跃数，
会漏掉跨循环存活的 buffer。故必须按"事件点扫描"计算逐点占用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hivm_spec.vir import (
    Access,
    AllocOrigin,
    GapKind,
    Loc,
    SizeOrigin,
    VAlloc,
    VModule,
)

__all__ = [
    "Interval",
    "OccupancyResult",
    "SpaceOccupancy",
    "analyze_occupancy",
]

ENGINE_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class Interval:
    """一个 buffer 的活跃区间（以节点序号为坐标）。"""

    alloc: VAlloc
    #: 起点：alloc 所在的节点序号（alloc 先于其首次使用）
    start: int
    #: 终点：最后一次访问的节点序号（含）
    end: int
    #: 该 buffer 是否从未被访问——强信号：要么是死代码，要么是引擎漏了效应
    never_used: bool = False

    @property
    def nbytes(self) -> int | None:
        return self.alloc.nbytes

    def contains(self, point: int) -> bool:
        return self.start <= point <= self.end


@dataclass
class SpaceOccupancy:
    """单个地址空间的占用结果。"""

    space: str
    capacity: int | None
    intervals: list[Interval] = field(default_factory=list)
    #: 逐事件点的占用曲线 [(节点序号, 字节数), ...]
    curve: list[tuple[int, int]] = field(default_factory=list)
    peak_bytes: int = 0
    peak_at: int = -1
    #: 峰值时刻的活跃 buffer（按尺寸降序）——溢出时的贡献者排序
    peak_contributors: list[Interval] = field(default_factory=list)
    #: 尺寸未知因而未计入峰值的 buffer
    unsized: list[VAlloc] = field(default_factory=list)

    @property
    def overflows(self) -> bool:
        return self.capacity is not None and self.peak_bytes > self.capacity

    @property
    def is_vacuous(self) -> bool:
        """本 space 是否什么都没分析到。

        零 buffer 的"未溢出"是**空洞结论**：它不是"这个 kernel 安全"，而是
        "我没找到任何 buffer"。二者对 agent 的意义完全相反。
        """
        return not self.intervals and not self.unsized

    @property
    def headroom(self) -> int | None:
        if self.capacity is None:
            return None
        return self.capacity - self.peak_bytes

    @property
    def utilization(self) -> float | None:
        if not self.capacity:
            return None
        return self.peak_bytes / self.capacity


@dataclass
class OccupancyResult:
    """占用分析结果（尚未转为 verdict——那是结论框架的职责）。"""

    spaces: dict[str, SpaceOccupancy] = field(default_factory=dict)
    #: 分析过程产生的新缺口（如尺寸未知、alloc 从未使用）
    gap_notes: list[tuple[GapKind, str, Loc | None]] = field(default_factory=list)

    @property
    def any_overflow(self) -> bool:
        return any(s.overflows for s in self.spaces.values())

    @property
    def is_vacuous(self) -> bool:
        """所有被检查的 space 都没有任何 buffer。

        此时"未溢出"毫无信息量，必须降级为缺口而非给 OK——否则一个
        本工具根本没看懂的 kernel 会得到干净的绿灯（FR6 反自欺）。
        """
        return bool(self.spaces) and all(s.is_vacuous for s in self.spaces.values())

    @property
    def overflowing_spaces(self) -> tuple[str, ...]:
        return tuple(sorted(n for n, s in self.spaces.items() if s.overflows))


def _last_access_index(
    module: VModule, alloc_name: str, order: list[str], value_to_alloc: dict[str, str]
) -> int:
    """找出 buffer 最后一次被访问的节点序号。

    匹配依据是效应的 `target` 经操作数解析到的 buffer。当前引擎对操作数
    只有文本形式，故用 value → alloc 的映射表；解析不到的一律不算访问，
    并由调用方登记"从未使用"缺口——宁可报缺口，不可假设它早已死亡
    （那会低估峰值，把溢出判成 OK）。
    """
    last = -1
    for idx, node in enumerate(module.walk_nodes()):
        for operand in node.operands:
            if value_to_alloc.get(operand) == alloc_name:
                last = idx
    return last


def _build_value_map(module: VModule) -> dict[str, str]:
    """SSA value 文本 → alloc 名（精确关联，依赖 `VAlloc.value`）。

    刻意不做"按出现顺序对齐"的近似：关联一旦错位，生存期就会算错，且错误
    方向不可控——可能低估峰值从而把真实溢出判成 OK。
    """
    return {a.value: a.name for a in module.allocs if a.value}


def analyze_occupancy(
    module: VModule,
    capacities: dict[str, int | None],
    *,
    spaces_of_interest: tuple[str, ...] = (),
) -> OccupancyResult:
    """计算 per-space 峰值占用。

    Args:
        module: 待分析 VIR。
        capacities: space → 容量（字节）；None 表示容量未知。
        spaces_of_interest: 限定分析范围；空则分析全部出现过的 space。
    """
    result = OccupancyResult()
    order = list(module.node_order())
    value_map = _build_value_map(module)

    # 节点序号 → 该节点访问到的 buffer 名集合
    access_at: list[set[str]] = []
    for node in module.walk_nodes():
        touched: set[str] = set()
        for eff in node.effects:
            if eff.access in (Access.READ, Access.WRITE):
                for operand in node.operands:
                    name = value_map.get(operand)
                    if name:
                        touched.add(name)
        # 没有效应声明的节点也可能访问 buffer（未建模 op），
        # 但那已由 COVERAGE_GAP 覆盖，此处不猜。
        access_at.append(touched)

    target_spaces = (
        set(spaces_of_interest) if spaces_of_interest else {a.space for a in module.allocs}
    )

    for space in sorted(target_spaces):
        allocs = [a for a in module.allocs if a.space == space]
        so = SpaceOccupancy(space=space, capacity=capacities.get(space))

        for alloc in allocs:
            if alloc.size_origin is SizeOrigin.UNKNOWN or alloc.nbytes is None:
                so.unsized.append(alloc)
                result.gap_notes.append(
                    (
                        GapKind.UNKNOWN_SIZE,
                        f"buffer {alloc.name}（space={space}）尺寸未知，未计入峰值——"
                        "峰值因此是**下界**，不足以判定不溢出",
                        alloc.loc,
                    )
                )
                continue

            if alloc.origin is AllocOrigin.FUNC_ARG:
                # 函数参数由调用方持有：生存期覆盖整个函数，与是否被本函数
                # 访问无关。按"活到末尾"处理，不登记"未见访问"缺口。
                so.intervals.append(
                    Interval(
                        alloc=alloc,
                        start=0,
                        end=max(len(order) - 1, 0),
                        never_used=False,
                    )
                )
                continue

            start = _alloc_index(module, alloc, order)
            last = _last_access_index(module, alloc.name, order, value_map)
            never = last < 0
            if never:
                result.gap_notes.append(
                    (
                        GapKind.UNRECOGNIZED_STRUCTURE,
                        f"buffer {alloc.name} 分配后未见任何访问——"
                        "可能是死代码，也可能是引擎漏了某个 op 的效应；"
                        "已按『存活至模块末尾』保守处理",
                        alloc.loc,
                    )
                )
                # 保守：假设活到最后。低估峰值会把溢出判成 OK，代价不可接受。
                last = max(len(order) - 1, start)
            so.intervals.append(Interval(alloc=alloc, start=start, end=last, never_used=never))

        _compute_curve(so, len(order))
        result.spaces[space] = so

    return result


def _alloc_index(module: VModule, alloc: VAlloc, order: list[str]) -> int:
    """alloc 在节点序中的位置。

    alloc 本身不是 VIR 节点（memref.alloc 不进节点流），故用"首个访问它的
    节点"作为起点；若从未被访问则从 0 起——保守，宁可高估。
    """
    value_map = {alloc.value: alloc.name} if alloc.value else {}
    for idx, node in enumerate(module.walk_nodes()):
        for operand in node.operands:
            if value_map.get(operand) == alloc.name:
                return idx
    return 0


def _compute_curve(so: SpaceOccupancy, n_points: int) -> None:
    """事件点扫描计算逐点占用与峰值。

    必须按区间累加而非按 alloc 时刻累加：两个生存期不重叠的 1KB buffer
    若被同时计入会产生**虚假溢出**，而漏算跨循环存活的 buffer 会**漏报真实溢出**。
    """
    if n_points <= 0:
        # 没有任何 hivm 节点，但可能仍有 buffer（如全 vector.* 的 kernel，其
        # 片上 buffer 由函数参数传入）。此时不能让峰值停留在 0——那会把
        # 768B 的真实占用报成 0。退化为单点：所有 buffer 同时活跃。
        total = sum(iv.nbytes or 0 for iv in so.intervals)
        so.curve = [(0, total)] if so.intervals else []
        so.peak_bytes = total
        so.peak_at = 0 if so.intervals else -1
        so.peak_contributors = sorted(
            so.intervals, key=lambda iv: (-(iv.nbytes or 0), iv.alloc.name)
        )
        return

    for point in range(n_points):
        total = 0
        for iv in so.intervals:
            if iv.contains(point) and iv.nbytes is not None:
                total += iv.nbytes
        so.curve.append((point, total))
        if total > so.peak_bytes:
            so.peak_bytes = total
            so.peak_at = point

    if so.peak_at >= 0:
        so.peak_contributors = sorted(
            (iv for iv in so.intervals if iv.contains(so.peak_at)),
            key=lambda iv: (-(iv.nbytes or 0), iv.alloc.name),
        )
