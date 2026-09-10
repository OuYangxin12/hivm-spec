"""同步静态配对检查（T4.4）——账本级快速检查。

## 与 M2 timeline 工具的分工

二者**互补，不是替代**。timeline 做交错探索（重：展开循环 × 多策略 × 定点
迭代），本模块做静态配对盘点（轻：一遍扫描全量 `VSync`）。

各自能看见对方看不见的东西：

| | timeline | 本模块 |
|---|---|---|
| 判定依据 | 展开域内的可调度性 | 全模块 sync 账目 |
| 受展开界影响 | **是**（截断时规则 B/C 失效） | **否**（不展开，只计数） |
| set 有而无人 wait | 不查（不影响可调度性） | **查**（`orphan-set`） |
| 顺序不可行导致的死锁 | **查**（wait-cycle） | 不查（计数看不出顺序） |

`VSync` 的 docstring 记着真实死锁案例（CreatePreload stage-major）的教训：
**不能只看 set/wait 总数配平**——顺序不可行时计数照样平衡。所以本模块
**不声称能判死锁**，它报告的是"账目异常"，是线索而非结论。

## 为什么 orphan-set 值得单独报

一个 set 从来没有人等，通常意味着两件事之一：① 对应的 wait 在变换中被删了
（真缺陷）；② 该事件本就不需要同步（冗余 set，性能问题而非正确性问题）。

二者都值得知道，但**严重性不同**，故本模块报 `warning` 而非 `error`——它不
构成"验证失败"。把它报成 error 会让大量合法 IR 变红，淹没真问题（NFR2）。

## 结论口径

本模块**不产出 DEADLOCK**。账目异常在语义上是"可疑"，不是"必然死锁"；
真正的死锁判定归 timeline。二者结论不一致时以 timeline 为准，且不一致本身
必须暴露（M4 卡 §4.6）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from hivm_spec.vir import Loc, SyncKind, VModule, VSync

__all__ = [
    "PAIRING_ENGINE_VERSION",
    "EventLedger",
    "PairingFinding",
    "PairingReport",
    "analyze_pairing",
]

#: 引擎版本——结论里回显，便于把历史结论对应到当时的判定口径。
PAIRING_ENGINE_VERSION = "0.1.0"

_SET_KINDS = frozenset({SyncKind.SET_FLAG, SyncKind.SYNC_BLOCK_SET})
_WAIT_KINDS = frozenset({SyncKind.WAIT_FLAG, SyncKind.SYNC_BLOCK_WAIT})


@dataclass(frozen=True, slots=True)
class EventLedger:
    """单个事件 id 的供需账目。"""

    event_id: int
    sets: int = 0
    waits: int = 0
    #: 首个 set / wait 的源位置，供诊断定位
    first_set: Loc | None = None
    first_wait: Loc | None = None
    #: 参与该事件的 pipe 集合（跨 pipe 同步的线索）
    pipes: tuple[str, ...] = ()
    #: 是否含隐式（macro 内部）同步——隐式项不可见，判定须保守
    has_implicit: bool = False

    @property
    def balanced(self) -> bool:
        return self.sets == self.waits


@dataclass(frozen=True, slots=True)
class PairingFinding:
    """一条配对异常。

    `severity` 刻意区分：
    - `error`：wait 无任何 set —— 该 wait 必然等不到（与 timeline 规则 A 同源）
    - `warning`：set 无任何 wait、供需不平 —— 可疑但不必然错
    """

    rule: str
    severity: str
    event_id: int | None
    message: str
    loc: Loc | None = None


@dataclass(frozen=True, slots=True)
class PairingReport:
    ledgers: tuple[EventLedger, ...] = ()
    findings: tuple[PairingFinding, ...] = ()
    #: 无 event_id 因而无法参与配对的 sync 数量（分析缺口，不是"通过"）
    unattributable: int = 0
    #: 含隐式同步的事件数——这些事件的账目不完整
    implicit_events: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_error(self) -> bool:
        return any(f.severity == "error" for f in self.findings)


def _loc_of(sync: VSync) -> Loc | None:
    return sync.loc


def analyze_pairing(module: VModule) -> PairingReport:
    """全模块 sync 账目盘点。

    **不展开循环**：这是与 timeline 的关键区别，也是它不受截断影响的原因。
    代价是看不出顺序问题——这个代价是自觉的，见模块 docstring。
    """
    sets: Counter[int] = Counter()
    waits: Counter[int] = Counter()
    first_set: dict[int, Loc] = {}
    first_wait: dict[int, Loc] = {}
    pipes: dict[int, set[str]] = {}
    implicit: set[int] = set()
    unattributable = 0

    for sync in module.syncs:
        if sync.kind is SyncKind.PIPE_BARRIER:
            # barrier 不按 event id 配对，归 timeline 的语义处理
            continue
        if sync.event_id is None:
            # 无 id 则无法配对。**计入缺口而非忽略**：静默丢弃会让"没检查"
            # 伪装成"检查通过"（FR6）。
            unattributable += 1
            continue

        ev = sync.event_id
        if getattr(sync, "implicit", False):
            implicit.add(ev)
        if sync.pipe:
            pipes.setdefault(ev, set()).add(sync.pipe)

        if sync.kind in _SET_KINDS:
            sets[ev] += 1
            first_set.setdefault(ev, _loc_of(sync))  # type: ignore[arg-type]
        elif sync.kind in _WAIT_KINDS:
            waits[ev] += 1
            first_wait.setdefault(ev, _loc_of(sync))  # type: ignore[arg-type]

    ledgers = tuple(
        EventLedger(
            event_id=ev,
            sets=sets.get(ev, 0),
            waits=waits.get(ev, 0),
            first_set=first_set.get(ev),
            first_wait=first_wait.get(ev),
            pipes=tuple(sorted(pipes.get(ev, ()))),
            has_implicit=ev in implicit,
        )
        for ev in sorted(set(sets) | set(waits))
    )

    findings: list[PairingFinding] = []
    for led in ledgers:
        if led.has_implicit:
            # 隐式同步发生在 macro 内部，IR 不可见 → 账目不完整 → 不判异常。
            # 报"可能不完整"而非"没问题"：前者诚实，后者是自欺。
            continue

        if led.waits > 0 and led.sets == 0:
            findings.append(
                PairingFinding(
                    rule="pairing/unpaired-wait",
                    severity="error",
                    event_id=led.event_id,
                    message=(
                        f"事件 {led.event_id} 有 {led.waits} 个 wait 但全模块无任何 set"
                        "——该 wait 必然等不到"
                    ),
                    loc=led.first_wait,
                )
            )
        elif led.sets > 0 and led.waits == 0:
            findings.append(
                PairingFinding(
                    rule="pairing/orphan-set",
                    severity="warning",
                    event_id=led.event_id,
                    message=(
                        f"事件 {led.event_id} 有 {led.sets} 个 set 但无人 wait"
                        "——可能是配对的 wait 被误删，也可能是冗余同步"
                    ),
                    loc=led.first_set,
                )
            )
        elif not led.balanced:
            findings.append(
                PairingFinding(
                    rule="pairing/count-imbalance",
                    severity="warning",
                    event_id=led.event_id,
                    message=(
                        f"事件 {led.event_id} 供需不平：{led.sets} set / {led.waits} wait。"
                        "注意计数平衡**不等于**无死锁——顺序不可行时计数照样平衡，"
                        "死锁判定以 timeline 为准"
                    ),
                    loc=led.first_wait or led.first_set,
                )
            )

    notes: list[str] = []
    if unattributable:
        notes.append(f"{unattributable} 个 sync 无 event id，无法参与配对分析——本报告未覆盖它们")
    if implicit:
        notes.append(f"{len(implicit)} 个事件含隐式（macro 内部）同步，账目不完整，已跳过判定")
    notes.append("本检查为账目级：计数平衡不蕴含无死锁，顺序问题请看 timeline 工具")

    return PairingReport(
        ledgers=ledgers,
        findings=tuple(findings),
        unattributable=unattributable,
        implicit_events=len(implicit),
        notes=tuple(notes),
    )
