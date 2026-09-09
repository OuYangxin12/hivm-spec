"""时序引擎（T2.1–T2.3/T2.5）：pipe/event 状态机 + 结构性死锁判定。

**只消费 VIR**（D6）——与占用引擎平级，不接触 MLIR，不另建遍历核。

## 模型口径（M2 已声明的简化，随结论输出）

1. **事件步语义**：时间轴是"执行步"而非时钟（耗时模型属 M3+，见 M2 卡 §5）。
2. **泳道 = pipe**：每个 op 的泳道取其 pipe 归属（VNode.pipe / VSync.pipe）。
   同一泳道内严格保持**程序序**；跨泳道并发——这正是 HIVM 流水线的本质，
   也是"计数配平 ≠ 顺序可行"的物理来源：set/wait 总数配平的 IR，
   只要 wait 与它的 set 落在同一泳道且 wait 在前，照样结构性死锁。
3. **事件计数语义（re-arm 模型）**：flag 初态建模为**已装载**（计数 =
   `INITIAL_ARM`；D9 候选：对拍主仓 C++ 链前按保守方向取值——初态已装载
   使等待更容易放行，压制假阳性死锁）；`set` 使事件计数 +1（永不阻塞，
   即"re-arm"），`wait` 在计数 >0 时消费 1 并放行，否则阻塞。set/wait 的
   **配对判定只用 event id**——IR 括号中 set_pipe/wait_pipe 的方向约定尚未
   与主仓 C++ 链对拍定稿，判定刻意不依赖方向（方向只影响展示）；若后续
   对拍发现语义偏差，按 D9 登记漂移。
4. **barrier = 函数级汇合（保守）**：`pipe_barrier` 在其所属函数内**先于它的
   全部步骤**完成前不放行。真实 barrier 语义若更弱，按 D9 登记。
5. **循环展开**：静态 trip 全量展开（不截断，安全上限内）；动态/未知 trip
   展开至 `bound` 并标记截断——**结论必须携带截断限定**（T2.5 诚实口径）。
6. **事件 id 缺失的 set/wait 不参与死锁判定**（无法配对），只作为分析缺口
   驱动 verdict 降级——不猜、不静默（FR4）。

## 两层判定（OD2/D4 的落地）

- **确定性层（`structural_fixpoint`）**：两种对立贪心序（程序序升序 /
  set 优先）求可执行闭包，wait 受阻的判定规则：
  - **规则 A（unpaired-wait）**：事件在整个模块（全量 syncs，与展开界无关）
    中不存在任何 set → 结构性死锁；
  - **规则 B（wait-cycle）**：事件存在 set，但展开域内的全部 set 在两种
    贪心序下都被前置阻塞 wait 挡住（wait-for 环），且**无截断** → 结构性
    死锁；有截断时降级为探索层观察（截断可能人为切断 set 可达性）。
  只有规则 A/B 才产生 `DEADLOCK` verdict——确定性结论不依赖策略探索。
- **探索层（`simulate`）**：交错策略集（顺序/轮转/pipe 优先/K 随机种子）
  产出时间线视图。策略下的 wait 受阻是 **schedule 依赖的观察**，只进
  diagnostics（NFR2：误判优先压制），不产生 verdict；探索层负向结论恒为
  "在{策略集}×{展开界}内未发现"（T2.5 措辞，禁止无条件"无死锁"）。
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from hivm_spec.vir import Loc, SyncKind, VModule, VNode, VSync

__all__ = [
    "DEFAULT_BOUND",
    "MAX_STATIC_TRIP",
    "RANDOM_SEEDS",
    "STRATEGIES",
    "TIMELINE_ENGINE_VERSION",
    "DeadlockFinding",
    "Expansion",
    "SimResult",
    "TStep",
    "event_of",
    "events_with_any_set",
    "expand_steps",
    "simulate",
    "structural_fixpoint",
]

TIMELINE_ENGINE_VERSION = "0.1.0"

#: 静态 trip 的全量展开安全上限：超过则按界截断（防步数爆炸）。
MAX_STATIC_TRIP = 256
#: 全模块展开步数硬上限（超出即截断并如实标注）。
MAX_TOTAL_STEPS = 4096
#: 展开界缺省值（check options 未声明 unroll_bound 时）。
DEFAULT_BOUND = 16
# flag 初态：按 re-arm 语义建模为已装载（D9 候选，见模块口径 3）
INITIAL_ARM = 1

#: 确定性/探索层共用的同步 kind 分组
_SET_KINDS = frozenset({SyncKind.SET_FLAG, SyncKind.SYNC_BLOCK_SET})
_WAIT_KINDS = frozenset({SyncKind.WAIT_FLAG, SyncKind.SYNC_BLOCK_WAIT})

#: 探索策略集（T2.2）。random 策略按固定种子展开 → 同种子可复现（FR8）。
STRATEGIES: tuple[str, ...] = ("sequential", "round_robin", "pipe_priority")
RANDOM_SEEDS: tuple[int, ...] = (0, 1, 2, 3)


def event_of(step: TStep) -> int | None:
    """set/wait 步骤的事件 id；barrier 与普通 op 为 None。"""
    if step.sync is None:
        return None
    if step.sync.kind in _SET_KINDS | _WAIT_KINDS:
        return step.sync.event_id
    return None


def events_with_any_set(module: VModule) -> set[int]:
    """模块全量 syncs 中出现过 set 的事件 id 集合（规则 A 的存在性基准）。

    刻意取**全量**而非展开域：截断只影响"set 是否被执行"，
    不影响"set 是否存在"——规则 A 必须与展开界无关。
    """
    return {s.event_id for s in module.syncs if s.kind in _SET_KINDS and s.event_id is not None}


# ---------------------------------------------------------------------------
# 展开：程序序 → 线性步骤序列（T2.1 的 VIR 消费面）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TStep:
    """展开后的一个执行步（VNode + 迭代路径 + 泳道）。"""

    seq: int
    node: VNode
    sync: VSync | None
    #: 迭代下标路径（嵌套循环逐层下标；空 = 循环外）
    loop_path: tuple[int, ...]
    #: 所属顶层区域（func）序号
    func: int
    #: 泳道（pipe）；空 pipe 归入 "@unassigned"
    lane: str


@dataclass(frozen=True, slots=True)
class Expansion:
    """展开产物：步骤序列 + 诚实标注。"""

    steps: tuple[TStep, ...]
    #: 是否发生了截断（未知 trip 按界 / 静态 trip 超上限 / 步数超硬上限）
    truncated: bool
    #: 截断原因清单（人可读，随结论输出）
    truncation_notes: tuple[str, ...]
    #: 静态 trip 全量展开的循环数（"未截断"的证据面）
    static_full_loops: int


def expand_steps(module: VModule, bound: int) -> Expansion:
    """把 VIR 程序序展开为线性步骤序列（循环按口径迭代）。"""
    steps: list[TStep] = []
    notes: list[str] = []
    truncated = False
    static_full = 0

    sync_by_node = {s.node_id: s for s in module.syncs}

    def emit_region(items: tuple[Any, ...], path: tuple[int, ...], func: int) -> bool:
        """按程序序展开一个区域；返回是否因上限截断。"""
        nonlocal truncated, static_full
        for item in items:
            if isinstance(item, VNode):
                if len(steps) >= MAX_TOTAL_STEPS:
                    notes.append(f"步数达到硬上限 {MAX_TOTAL_STEPS}，其后步骤未展开")
                    truncated = True
                    return True
                sync = sync_by_node.get(item.id)
                lane = (sync.pipe if sync is not None else item.pipe) or "@unassigned"
                steps.append(
                    TStep(
                        seq=len(steps), node=item, sync=sync, loop_path=path, func=func, lane=lane
                    )
                )
                continue
            loop = item.loop
            if loop is None:
                if emit_region(item.items, (*path, 0), func):
                    return True
                continue
            trip = loop.trip_count
            if trip is None:
                iters, why = bound, f"trip 未知（{item.id}），按展开界 {bound} 截断"
            elif trip > MAX_STATIC_TRIP:
                iters, why = (
                    bound,
                    (
                        f"静态 trip={trip} 超出安全上限 {MAX_STATIC_TRIP}（{item.id}），"
                        f"按展开界 {bound} 截断"
                    ),
                )
            else:
                iters, why = trip, ""
                static_full += 1
            for it in range(iters):
                if emit_region(item.items, (*path, it), func):
                    return True
            if why:
                notes.append(why)
                truncated = True
        return False

    for fi, top in enumerate(module.items):
        if emit_region(top.items, (), fi):
            break

    return Expansion(
        steps=tuple(steps),
        truncated=truncated,
        truncation_notes=tuple(notes),
        static_full_loops=static_full,
    )


# ---------------------------------------------------------------------------
# 可执行性谓词（两层判定共用——口径必须唯一）
# ---------------------------------------------------------------------------


@dataclass
class _State:
    """模拟状态：已执行集合 + 事件计数 + 泳道/函数进度。"""

    executed: set[int]
    counters: dict[int, int]
    lane_done: dict[str, int]
    func_done: dict[int, int]

    @classmethod
    def fresh(cls, event_ids: Iterable[int] = ()) -> _State:
        # 初始装载必须**真实写入**而非 get 缺省——否则首个 set 落盘后幻影装载
        # 即消失，计数轨迹错乱（M2 实测 bug，见 re-arm 口径 3）
        return cls(
            executed=set(),
            counters={ev: INITIAL_ARM for ev in event_ids},
            lane_done={},
            func_done={},
        )


class _Semantics:
    """可执行性判定与状态转移（确定性层与策略层共享同一实例口径）。

    **泳道前沿调度**：泳道内程序序门控意味着只有各泳道的**前沿步骤**可能
    可执行——候选集大小为 O(泳道数)，drain 总代价 O(n×泳道数)，而非 O(n²)。
    这是 19 份 L2 kernel 在 30s 预算内跑 7 策略的前提（D10）。
    """

    def __init__(self, steps: tuple[TStep, ...]) -> None:
        self.steps = steps
        self.lane_index: dict[int, int] = {}
        lane_totals: dict[str, int] = {}
        for i, st in enumerate(steps):
            self.lane_index[i] = lane_totals.get(st.lane, 0)
            lane_totals[st.lane] = lane_totals.get(st.lane, 0) + 1
        self.lane_totals = lane_totals
        # 每泳道的步骤序列（前沿指针的基准）
        self.lane_steps: dict[str, list[int]] = {lane: [] for lane in lane_totals}
        for i, st in enumerate(steps):
            self.lane_steps[st.lane].append(i)
        self.event_ids: tuple[int, ...] = tuple(
            sorted({ev for st in steps if (ev := event_of(st)) is not None})
        )
        # barrier 门控：步骤 i 之前（同函数）的全部 barrier——它们不执行完，
        # i 不可执行（barrier = 函数级汇合，跨泳道成立）
        self.barrier_gates: dict[int, tuple[int, ...]] = {}
        self.func_index: dict[int, int] = {}
        barriers: list[int] = []
        cur_func = -1
        pos_in_func = 0
        for i, st in enumerate(steps):
            if st.func != cur_func:
                cur_func = st.func
                barriers = []
                pos_in_func = 0
            self.func_index[i] = pos_in_func
            pos_in_func += 1
            if st.sync is not None and st.sync.kind is SyncKind.PIPE_BARRIER:
                self.barrier_gates[i] = tuple(barriers)
                barriers.append(i)
            else:
                self.barrier_gates[i] = tuple(barriers)

    def fronts(self, state: _State) -> list[int]:
        """各泳道前沿中尚未执行的步骤（唯一可能可执行的候选集）。"""
        cand: list[int] = []
        for lane, seqs in self.lane_steps.items():
            done = state.lane_done.get(lane, 0)
            if done < len(seqs):
                cand.append(seqs[done])
        return sorted(cand)

    def executable(self, i: int, state: _State) -> bool:
        st = self.steps[i]
        if i in state.executed:
            return False
        # 泳道内程序序：必须是本泳道前沿
        if state.lane_done.get(st.lane, 0) != self.lane_index[i]:
            return False
        # barrier 汇合：先于本步的同函数 barrier 必须已全部执行
        for b in self.barrier_gates[i]:
            if b not in state.executed:
                return False
        if st.sync is None:
            return True
        kind = st.sync.kind
        if kind in _SET_KINDS:
            return True
        if kind in _WAIT_KINDS:
            ev = event_of(st)
            if ev is None:
                # 事件 id 不可解析：无法判定配对——放行但记为分析缺口（不阻塞闭包）
                return True
            return state.counters.get(ev, 0) > 0
        # PIPE_BARRIER 自身：函数内先于它的步骤全部完成（保守汇合）
        return state.func_done.get(st.func, 0) == self.func_index[i]

    def apply(self, i: int, state: _State) -> None:
        st = self.steps[i]
        state.executed.add(i)
        state.lane_done[st.lane] = state.lane_done.get(st.lane, 0) + 1
        state.func_done[st.func] = state.func_done.get(st.func, 0) + 1
        ev = event_of(st)
        if ev is None or st.sync is None:
            return
        if st.sync.kind in _SET_KINDS:
            state.counters[ev] = state.counters.get(ev, 0) + 1
        elif st.sync.kind in _WAIT_KINDS:
            state.counters[ev] = state.counters.get(ev, 0) - 1

    def drain(self, state: _State, pick: Callable[[list[int], _State], int]) -> None:
        """反复执行可执行的前沿步骤直到不动点。"""
        while True:
            cand = [i for i in self.fronts(state) if self.executable(i, state)]
            if not cand:
                return
            self.apply(pick(cand, state), state)

    def blocked_waits(self, state: _State) -> tuple[int, ...]:
        out: list[int] = []
        for i, st in enumerate(self.steps):
            if i in state.executed or st.sync is None:
                continue
            if st.sync.kind in _WAIT_KINDS:
                out.append(i)
        return tuple(out)


# ---------------------------------------------------------------------------
# 确定性层（T2.3）：结构性死锁
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeadlockFinding:
    """一条结构性死锁（确定性结论）。"""

    rule: str  # "unpaired-wait" | "wait-cycle"
    event_id: int | None
    step_seq: int
    loc: Loc
    message: str


def structural_fixpoint(
    steps: tuple[TStep, ...],
    truncated: bool,
    module_set_events: set[int],
) -> tuple[DeadlockFinding, ...]:
    """确定性死锁判定（规则 A/B，见模块 docstring）。

    两种对立贪心序都受阻才判 wait-cycle——单一顺序下的受阻可能只是
    调度巧合（NFR2：误判优先压制）。规则 A 的存在性基准是模块全量
    syncs，与展开界无关。
    """
    sem = _Semantics(steps)
    expanded_sets: dict[int, list[int]] = {}
    for i, st in enumerate(steps):
        if st.sync is not None and st.sync.kind in _SET_KINDS:
            ev = event_of(st)
            if ev is not None:
                expanded_sets.setdefault(ev, []).append(i)

    # 贪心序一：程序序升序
    s1 = _State.fresh(sem.event_ids)
    sem.drain(s1, lambda cand, _s: cand[0])
    # 贪心序二：set 优先（最大化放行——最宽松口径）
    s2 = _State.fresh(sem.event_ids)

    def _pick_sets_first(cand: list[int], _s: _State) -> int:
        for i in cand:
            st = steps[i]
            if st.sync is not None and st.sync.kind in _SET_KINDS:
                return i
        return cand[0]

    sem.drain(s2, _pick_sets_first)

    # 去重口径：同一 (rule, event) 只报**首个受阻 wait**的 loc，并附受阻总数——
    # 展开循环里同一缺陷会复制成 N 个受阻步，逐条输出既淹没摘要也不影响判定
    seen: dict[tuple[str, int], DeadlockFinding] = {}
    counts: dict[tuple[str, int], int] = {}
    for i, st in enumerate(steps):
        if st.sync is None or st.sync.kind not in _WAIT_KINDS:
            continue
        blocked_in_both = i not in s1.executed and i not in s2.executed
        if not blocked_in_both:
            continue
        ev = event_of(st)
        if ev is None:
            continue  # 事件 id 缺失属分析缺口（verdict 层降级），不作死锁判定
        if ev not in module_set_events:
            key = ("unpaired-wait", ev)
            counts[key] = counts.get(key, 0) + 1
            if key not in seen:
                seen[key] = DeadlockFinding(
                    rule="unpaired-wait",
                    event_id=ev,
                    step_seq=i,
                    loc=st.node.loc,
                    message=(
                        f"wait 等待事件 {ev}：模块中不存在任何对它的 set，"
                        "连初始装载也不足以放行（与展开界无关，确定性结论）"
                    ),
                )
            continue
        # 规则 B：展开域内的 set 全部不可达 + 无截断
        in_expansion = expanded_sets.get(ev, [])
        reachable = any(s in s1.executed or s in s2.executed for s in in_expansion)
        if in_expansion and not reachable and not truncated:
            key = ("wait-cycle", ev)
            counts[key] = counts.get(key, 0) + 1
            if key not in seen:
                seen[key] = DeadlockFinding(
                    rule="wait-cycle",
                    event_id=ev,
                    step_seq=i,
                    loc=st.node.loc,
                    message=(
                        f"wait 等待事件 {ev}：其展开域内全部 set（{len(in_expansion)} 处）在两种"
                        "贪心序下均被前置阻塞 wait 挡住——wait-for 环，结构性死锁"
                    ),
                )
    findings: list[DeadlockFinding] = []
    for (rule, ev), f in seen.items():
        n = counts[(rule, ev)]
        message = f.message if n == 1 else f"{f.message}（共 {n} 个受阻 wait 步，此处为首现位置）"
        findings.append(
            DeadlockFinding(rule=rule, event_id=ev, step_seq=f.step_seq, loc=f.loc, message=message)
        )
    return tuple(findings)


# ---------------------------------------------------------------------------
# 探索层（T2.2）：交错策略集 → 时间线
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimResult:
    """单策略模拟产物。"""

    strategy: str
    #: 执行序（步骤 seq 按执行先后排列）
    order: tuple[int, ...]
    #: 该策略下受阻的 wait 步骤 seq
    blocked: tuple[int, ...]


def simulate(steps: tuple[TStep, ...], strategy: str, pipe_priority: dict[str, int]) -> SimResult:
    """按指定策略模拟一次，产出执行序与受阻清单。

    策略只决定**执行顺序的选择**，不改变可执行性口径（`_Semantics` 唯一）。
    执行序经 `drain` 的泳道前沿调度取得（O(n×泳道数)）；随机策略以固定种子
    实例化 → 同输入同结论（FR8）。
    """
    sem = _Semantics(steps)
    state = _State.fresh(sem.event_ids)
    order: list[int] = []

    if strategy == "sequential":

        def pick(cand: list[int], _s: _State) -> int:
            return cand[0]

    elif strategy == "round_robin":
        lanes = sorted(sem.lane_totals)
        cursor = {"i": 0}

        def pick(cand: list[int], _s: _State) -> int:
            by_lane: dict[str, int] = {}
            for i in cand:
                by_lane.setdefault(steps[i].lane, i)
            for _ in range(len(lanes)):
                lane = lanes[cursor["i"] % len(lanes)]
                cursor["i"] += 1
                if lane in by_lane:
                    return by_lane[lane]
            return cand[0]

    elif strategy == "pipe_priority":

        def pick(cand: list[int], _s: _State) -> int:
            return min(cand, key=lambda i: (pipe_priority.get(steps[i].lane, 1 << 30), i))

    elif strategy.startswith("random"):
        # 名称形如 "random(seed=3)"（assemble/CLI 构造），解析失败即契约 bug，须炸出
        seed = int(strategy.removeprefix("random(seed=").removesuffix(")"))
        rng = random.Random(seed)

        def pick(cand: list[int], _s: _State) -> int:
            return rng.choice(cand)

    else:  # pragma: no cover — 由 STRATEGIES/RANDOM_SEEDS 常量约束
        raise ValueError(f"未知策略：{strategy}")

    def pick_and_record(cand: list[int], s: _State) -> int:
        i = pick(cand, s)
        order.append(i)
        return i

    sem.drain(state, pick_and_record)
    return SimResult(strategy=strategy, order=tuple(order), blocked=sem.blocked_waits(state))
