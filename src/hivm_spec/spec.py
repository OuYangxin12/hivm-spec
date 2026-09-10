"""Spec DSL：HIVM 语义域的描述 API（T0.1）。

设计依据 T0.0b spike（`specs/spike_t0_0b.py`）导出的 API 需求：

- `In[]`/`Out[]`/`Attr[]`/`Variadic[]` 参数标注 → 自动推导效应，减少描述样板；
- **独立的 `sync_*` 效应原语** —— spike 的关键发现：同步是对"同步状态"的读写，
  只有 `rd/wr(space)` 一种原语时事件对根本无法表达；
- **`cond_wr` 条件效应** —— 否则 `mmadL1.init_condition` 的条件语义静默丢失；
- **`host_fn` 逃生舱**（OD8）—— 布局代数不进声明式描述，且 trust 自动封顶
  `provisional`（不可升级，防止逃生舱条目伪装成高可信）。

本模块是**纯 Python**（D7），不 import `bishengir`。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "UNBOUND",
    "Attr",
    "CheckSpec",
    "EffectDecl",
    "EventDecl",
    "In",
    "OpSpec",
    "Out",
    "ParamKind",
    "PipeDecl",
    "SemanticAssumption",
    "SpaceDecl",
    "Spec",
    "SpecError",
    "Variadic",
    "cond_wr",
    "host_fn",
    "rd",
    "sync_barrier",
    "sync_set",
    "sync_wait",
    "wr",
]


class SpecError(ValueError):
    """描述层错误。

    与 `VIRError` 区分：这是**描述作者**的错误，报错必须带描述内定位
    （FR7：区分"描述问题"与"被验证 IR 的问题"）。
    """


class Trust(Enum):
    """信任级别（D4/OD4）。逃生舱条目封顶 PROVISIONAL。"""

    PROVISIONAL = "provisional"
    CROSS_VALIDATED = "cross-validated"
    ANCHORED = "anchored"


TRUST_ORDER = {Trust.PROVISIONAL: 0, Trust.CROSS_VALIDATED: 1, Trust.ANCHORED: 2}


# ---------------------------------------------------------------------------
# 参数标注（spike 需求 ①③）
# ---------------------------------------------------------------------------


class ParamKind(Enum):
    IN = "in"
    OUT = "out"
    #: 来自 MLIR 属性而非操作数（如 [<PIPE_V>, <EVENT_ID0>]）
    ATTR = "attr"


#: 未绑定参数名的哨兵：`In()`/`Out()` 构造时还不知道参数名，
#: 名字在 `Spec.op(params={...})` 处由 key 绑定。用显式哨兵而非空串，
#: 使"忘记绑定"与"名字为空"两种错误可区分。
UNBOUND = "<unbound>"


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    kind: ParamKind
    #: 声明的语义类型（"buf"/"scalar"/"index"/"pipe"/"event"）

    type_name: str = "buf"
    variadic: bool = False
    #: variadic 的数量约束（None 表示不限）；ODS 常要求 empty 或特定 size
    arity: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise SpecError("参数名不得为空")

    @property
    def is_bound(self) -> bool:
        return self.name != UNBOUND


def In(type_name: str = "buf") -> Param:
    return Param(name=UNBOUND, kind=ParamKind.IN, type_name=type_name)


def Out(type_name: str = "buf") -> Param:
    return Param(name=UNBOUND, kind=ParamKind.OUT, type_name=type_name)


def Attr(type_name: str) -> Param:
    return Param(name=UNBOUND, kind=ParamKind.ATTR, type_name=type_name)


def Variadic(inner: Param, arity: tuple[int, ...] | None = None) -> Param:
    return Param(
        name=UNBOUND, kind=inner.kind, type_name=inner.type_name, variadic=True, arity=arity
    )


# ---------------------------------------------------------------------------
# 效应原语（spike 需求 ②③ —— 本项目表达力的核心）
# ---------------------------------------------------------------------------


class EffectKind(Enum):
    READ = "read"
    WRITE = "write"
    #: 条件写：spike ③ 的 init_condition
    COND_WRITE = "cond_write"
    #: 同步效应：spike ② 的关键发现，不可用 rd/wr 表达
    SYNC_SET = "sync_set"
    SYNC_WAIT = "sync_wait"
    SYNC_BARRIER = "sync_barrier"


@dataclass(frozen=True, slots=True)
class EffectDecl:
    """描述层声明的一条效应（区别于 `vir.Effect`：后者是解析后的实例）。"""

    kind: EffectKind
    #: 内存效应：目标参数名；同步效应：留空
    target: str = ""
    #: 地址空间；可为 "@from_type" 表示从 memref 类型的 address_space 提取
    space: str = ""
    #: 条件效应的条件参数名
    when: str = ""
    #: 同步效应：event/flag 参数名
    event: str = ""
    from_pipe: str = ""
    to_pipe: str = ""

    def __post_init__(self) -> None:
        mem = (EffectKind.READ, EffectKind.WRITE, EffectKind.COND_WRITE)
        if self.kind in mem and not self.target:
            raise SpecError(f"内存效应 {self.kind.value} 必须指明 target 参数")
        if self.kind is EffectKind.COND_WRITE and not self.when:
            raise SpecError("cond_wr 必须指明 when（条件参数名）")
        if self.kind in (EffectKind.SYNC_SET, EffectKind.SYNC_WAIT) and not self.event:
            raise SpecError(f"{self.kind.value} 必须指明 event 参数")

    @property
    def is_sync(self) -> bool:
        return self.kind in (
            EffectKind.SYNC_SET,
            EffectKind.SYNC_WAIT,
            EffectKind.SYNC_BARRIER,
        )


def rd(target: str, space: str = "@from_type") -> EffectDecl:
    return EffectDecl(kind=EffectKind.READ, target=target, space=space)


def wr(target: str, space: str = "@from_type") -> EffectDecl:
    return EffectDecl(kind=EffectKind.WRITE, target=target, space=space)


def cond_wr(target: str, when: str, space: str = "@from_type") -> EffectDecl:
    """条件写（spike ③）：`init_condition` 为真时才清零/写入。"""
    return EffectDecl(kind=EffectKind.COND_WRITE, target=target, space=space, when=when)


def sync_set(event: str, from_pipe: str = "", to_pipe: str = "") -> EffectDecl:
    return EffectDecl(kind=EffectKind.SYNC_SET, event=event, from_pipe=from_pipe, to_pipe=to_pipe)


def sync_wait(event: str, from_pipe: str = "", to_pipe: str = "") -> EffectDecl:
    return EffectDecl(kind=EffectKind.SYNC_WAIT, event=event, from_pipe=from_pipe, to_pipe=to_pipe)


def sync_barrier(pipe: str = "") -> EffectDecl:
    return EffectDecl(kind=EffectKind.SYNC_BARRIER, from_pipe=pipe)


# ---------------------------------------------------------------------------
# 逃生舱（OD8）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HostFn:
    """值语义由 host Python 函数提供（OD8 逃生舱）。

    trust 强制封顶 `provisional`：逃生舱内容不可被静态检查，也无法与
    C++ 链结构化对拍，故不允许伪装成高可信条目（spec-gate R2 同此约束）。
    """

    name: str
    fn: Callable[..., Any] | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise SpecError("host_fn 必须命名")
        if not self.reason:
            raise SpecError(
                f"host_fn({self.name}) 必须说明 reason——逃生舱是覆盖缺口的一种，理由须可审计"
            )


def host_fn(name: str, reason: str, fn: Callable[..., Any] | None = None) -> HostFn:
    return HostFn(name=name, fn=fn, reason=reason)


# ---------------------------------------------------------------------------
# 声明单元
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpSpec:
    op: str
    params: tuple[Param, ...]
    effects: tuple[EffectDecl, ...]
    pipe: str = ""
    value: HostFn | str = ""
    trust: Trust = Trust.PROVISIONAL
    doc: str = ""

    @property
    def effective_trust(self) -> Trust:
        """逃生舱条目封顶 provisional（OD8）。"""
        if isinstance(self.value, HostFn):
            return Trust.PROVISIONAL
        return self.trust

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)


@dataclass(frozen=True, slots=True)
class SpaceDecl:
    name: str
    capacity: int | None = None
    align: int | None = None


@dataclass(frozen=True, slots=True)
class PipeDecl:
    name: str


@dataclass(frozen=True, slots=True)
class EventDecl:
    name: str


@dataclass(frozen=True, slots=True)
class CheckSpec:
    name: str
    options: Mapping[str, Any] = field(default_factory=dict)


#: 已知 check 名（封闭集合：未知 check 应在静态检查阶段被拒绝，
#: 而不是生成一个什么都不做的工具）
KNOWN_CHECKS = frozenset({"ub_occupancy", "timeline", "equivalence", "sync_pairing"})


# ---------------------------------------------------------------------------
# Spec：描述的根容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticAssumption:
    """未经对拍确认的语义假设（D9 前置登记）。

    **为何需要它**：D9 要求"发现漂移一律先登记 drift ledger"，但漂移的**前身**
    是"我按某个方向猜了语义、还没和权威链对拍"。这类假设此前只能写在任务卡
    散文里（M2 的 flag 初态假设即如此），机器读不到、也无法在结论中声明——
    等于把最敏感的语义决策留在了账本之外（M2 审查发现 3）。

    登记后：进入账本 `assumptions` 段、随结论输出 provenance，并在对拍完成前
    冻结相关 op/引擎口径的 trust 升级（与 DriftEntry 同一原则）。
    """

    #: 假设的主体：op 名，或引擎口径（如 "timeline/flag-initial-state"）
    subject: str
    #: 假设内容（当前实现按什么语义执行）
    assumed: str
    #: 判定该假设真伪的权威（"upstream-cpp" | "hardware-golden" | "doc"）
    authority: str = "upstream-cpp"
    #: 选择该方向的理由（尤其：为何这个方向更保守）
    rationale: str = ""
    #: 若假设为假，会朝哪个方向错（"false-positive" | "false-negative" | "both"）
    risk_direction: str = "both"
    #: 计划的对拍时点（里程碑号，如 "M3"）
    resolve_by: str = ""


class Spec:
    """一份 HIVM 语义描述（§4.2 的 ops / vm / checks 三段）。"""

    def __init__(self, name: str, arch: str = "a3") -> None:
        if not name:
            raise SpecError("Spec 必须命名")
        self.name = name
        self.arch = arch
        self._ops: dict[str, OpSpec] = {}
        self._spaces: dict[str, SpaceDecl] = {}
        self._pipes: list[PipeDecl] = []
        self._events: list[EventDecl] = []
        self._checks: list[CheckSpec] = []
        self._assumptions: list[SemanticAssumption] = []

    # -- ops 段 -----------------------------------------------------------

    def op(
        self,
        name: str,
        *,
        params: Mapping[str, Param] | None = None,
        effects: Sequence[EffectDecl] = (),
        pipe: str = "",
        value: HostFn | str = "",
        trust: Trust = Trust.PROVISIONAL,
        doc: str = "",
    ) -> OpSpec:
        """声明一个 op 的语义。

        `params` 为 `{参数名: In()/Out()/Attr()/Variadic(...)}`；参数名在此绑定，
        使 `rd("a")` 等效应声明可被静态检查解析。
        """
        if not name:
            raise SpecError("op 名不得为空")
        if name in self._ops:
            raise SpecError(f"op 重复声明：{name}")
        bound = tuple(
            Param(
                name=k,
                kind=v.kind,
                type_name=v.type_name,
                variadic=v.variadic,
                arity=v.arity,
            )
            for k, v in (params or {}).items()
        )
        spec = OpSpec(
            op=name,
            params=bound,
            effects=tuple(effects),
            pipe=pipe,
            value=value,
            trust=trust,
            doc=doc,
        )
        self._ops[name] = spec
        return spec

    def merge(self, other: Spec) -> None:
        """并入另一份描述（真实 kernel 需要多个描述文件的**并集**）。

        重名一律报错——静默覆盖会让"两份描述对同一 op 给出不同语义"这一
        最危险的分歧被掩盖；冲突必须由描述作者显式解决。
        """
        if other.arch != self.arch:
            raise SpecError(
                f"arch 不一致，无法合并：{self.name}={self.arch}，{other.name}={other.arch}"
            )
        for name, op in other._ops.items():
            if name in self._ops:
                raise SpecError(f"op 重复声明：{name}（来自 {self.name} 与 {other.name}）")
            self._ops[name] = op
        for name, sp in other._spaces.items():
            if name in self._spaces:
                if self._spaces[name] != sp:
                    raise SpecError(f"space 冲突：{name}（{self._spaces[name]} vs {sp}）")
                continue
            self._spaces[name] = sp
        known_pipes = {p.name for p in self._pipes}
        known_events = {e.name for e in self._events}
        for p in other._pipes:
            if p.name not in known_pipes:
                self._pipes.append(p)
        for e in other._events:
            if e.name not in known_events:
                self._events.append(e)
        known_checks = {c.name for c in self._checks}
        for c in other._checks:
            if c.name in known_checks:
                continue
            self._checks.append(c)

    # -- vm 段 ------------------------------------------------------------

    def space(self, name: str, capacity: int | None = None, align: int | None = None) -> None:
        if name in self._spaces:
            raise SpecError(f"space 重复声明：{name}")
        self._spaces[name] = SpaceDecl(name=name, capacity=capacity, align=align)

    def pipe(self, *names: str) -> None:
        for n in names:
            if any(p.name == n for p in self._pipes):
                raise SpecError(f"pipe 重复声明：{n}")
            self._pipes.append(PipeDecl(name=n))

    def event(self, *names: str) -> None:
        for n in names:
            if any(e.name == n for e in self._events):
                raise SpecError(f"event 重复声明：{n}")
            self._events.append(EventDecl(name=n))

    # -- checks 段 --------------------------------------------------------

    def check(self, name: str, **options: Any) -> None:
        if any(c.name == name for c in self._checks):
            raise SpecError(f"check 重复声明：{name}")
        self._checks.append(CheckSpec(name=name, options=dict(options)))

    def assume(
        self,
        subject: str,
        assumed: str,
        *,
        authority: str = "upstream-cpp",
        rationale: str = "",
        risk_direction: str = "both",
        resolve_by: str = "",
    ) -> None:
        """登记一条未经对拍的语义假设（D9 前置，见 `SemanticAssumption`）。

        重复登记同一 subject 视为描述错误——一个主体只应有一个当前假设；
        假设变更应改写原条目，以免账本里堆叠互相矛盾的历史猜测。
        """
        if not subject or not assumed:
            raise SpecError("语义假设必须同时给出 subject 与 assumed")
        if any(a.subject == subject for a in self._assumptions):
            raise SpecError(f"语义假设重复登记：{subject}")
        if risk_direction not in ("false-positive", "false-negative", "both"):
            raise SpecError(
                f"risk_direction 取值非法：{risk_direction}"
                "（须为 false-positive / false-negative / both）"
            )
        self._assumptions.append(
            SemanticAssumption(
                subject=subject,
                assumed=assumed,
                authority=authority,
                rationale=rationale,
                risk_direction=risk_direction,
                resolve_by=resolve_by,
            )
        )

    # -- 只读访问 ---------------------------------------------------------

    @property
    def ops(self) -> tuple[OpSpec, ...]:
        return tuple(self._ops[k] for k in sorted(self._ops))

    @property
    def spaces(self) -> tuple[SpaceDecl, ...]:
        return tuple(self._spaces[k] for k in sorted(self._spaces))

    @property
    def pipes(self) -> tuple[PipeDecl, ...]:
        return tuple(sorted(self._pipes, key=lambda p: p.name))

    @property
    def events(self) -> tuple[EventDecl, ...]:
        return tuple(sorted(self._events, key=lambda e: e.name))

    @property
    def assumptions(self) -> tuple[SemanticAssumption, ...]:
        return tuple(sorted(self._assumptions, key=lambda a: a.subject))

    @property
    def checks(self) -> tuple[CheckSpec, ...]:
        return tuple(sorted(self._checks, key=lambda c: c.name))

    # -- 归一化与指纹（T0.3/T0.4） ----------------------------------------

    def normalize(self) -> dict[str, Any]:
        """归一化为稳定排序的配置文档（FR8：两次生成逐字节一致）。

        排序是确定性的唯一来源：dict 插入顺序、集合迭代顺序都不可依赖。
        """
        return {
            "schema_version": 1,
            "spec_name": self.name,
            "arch": self.arch,
            "ops": [
                {
                    "op": o.op,
                    "pipe": o.pipe,
                    "trust": o.effective_trust.value,
                    "escape_hatch": isinstance(o.value, HostFn),
                    "value": (
                        {"kind": "host_fn", "name": o.value.name, "reason": o.value.reason}
                        if isinstance(o.value, HostFn)
                        else {"kind": "declarative", "expr": o.value}
                    ),
                    "params": [
                        {
                            "name": p.name,
                            "kind": p.kind.value,
                            "type": p.type_name,
                            "variadic": p.variadic,
                            "arity": list(p.arity) if p.arity else None,
                        }
                        for p in o.params
                    ],
                    "effects": [
                        {
                            "kind": e.kind.value,
                            "target": e.target,
                            "space": e.space,
                            "when": e.when,
                            "event": e.event,
                            "from_pipe": e.from_pipe,
                            "to_pipe": e.to_pipe,
                        }
                        for e in o.effects
                    ],
                }
                for o in self.ops
            ],
            "vm": {
                "spaces": [
                    {"name": s.name, "capacity": s.capacity, "align": s.align} for s in self.spaces
                ],
                "pipes": [p.name for p in self.pipes],
                "events": [e.name for e in self.events],
            },
            "checks": [{"name": c.name, "options": _canonical(c.options)} for c in self.checks],
            #: 未经对拍的语义假设（D9 前置）。进 spec_hash：假设变更即语义口径
            #: 变更，必须产生新 hash，否则历史结论无法与其假设前提对应。
            "assumptions": [
                {
                    "subject": a.subject,
                    "assumed": a.assumed,
                    "authority": a.authority,
                    "rationale": a.rationale,
                    "risk_direction": a.risk_direction,
                    "resolve_by": a.resolve_by,
                }
                for a in self.assumptions
            ],
        }

    def canonical_bytes(self) -> bytes:
        """归一化配置文档的规范字节串（`spec_hash` 的输入，框架 §7.3）。"""
        return json.dumps(
            self.normalize(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()

    def spec_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()


def _canonical(value: Any) -> Any:
    """把 options 里的任意值转成可稳定序列化的形式。"""
    if isinstance(value, Mapping):
        return {k: _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
