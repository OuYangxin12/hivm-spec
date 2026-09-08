"""性质测试骨架（T0.7）：为 op 语义声明代数性质，用 Hypothesis 随机搜索反例。

**定位**：这是 D3 分层验证里"值语义层"的入口，但**不等于**等价验证（M3）。
两者分工：

| | 性质测试（T0.7） | 等价验证（M3） |
|---|---|---|
| 问的问题 | 描述**自身**是否自相矛盾 | 变换前后是否**语义等效** |
| 需要锚点 IR | ❌ | ✅（待验 + 锚点） |
| 发现的问题 | 描述写错了（如把 vsub 写成交换的） | pass 改错了 |

**为什么性质测试对本项目特别有价值**：描述由 AI agent 撰写，最典型的错误是
"看起来合理但代数性质不成立"（如误认为 `vdiv` 可交换）。这类错误无法被静态检查
发现——签名和效应都正确，只有值语义是错的。性质测试是唯一能自动发现它的手段。

注册表设计：性质与 op 名绑定但**不要求该 op 已被建模**——未建模 op 的性质声明
会被报告为"待建模"，而不是静默消失（FR4 的同一原则）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "REGISTRY",
    "Property",
    "PropertyKind",
    "PropertyRegistry",
    "commutative",
    "idempotent",
    "identity_element",
    "property_for",
]


class PropertyKind(Enum):
    """代数性质类别。"""

    COMMUTATIVE = "commutative"
    ASSOCIATIVE = "associative"
    IDEMPOTENT = "idempotent"
    IDENTITY = "identity"
    #: 自定义谓词
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class Property:
    """一条可被随机搜索反例的性质声明。"""

    op: str
    kind: PropertyKind
    #: 判定函数：接受该性质所需的参数，返回 bool
    predicate: Callable[..., bool]
    #: 性质的自然语言描述（反例报告里要能读懂违反了什么）
    description: str
    #: 已知不成立的场景（如浮点结合律）——显式记录而非假装成立
    caveats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.op:
            raise ValueError("Property.op 不得为空")
        if not self.description:
            raise ValueError(f"Property({self.op}/{self.kind.value}) 必须有描述")


class PropertyRegistry:
    """性质注册表。

    按 op 聚合，便于 ① 对单 op 跑全部性质；② 报告"有性质声明但未建模"的 op。
    """

    def __init__(self) -> None:
        self._props: list[Property] = []

    def add(self, prop: Property) -> Property:
        self._props.append(prop)
        return prop

    def all(self) -> tuple[Property, ...]:
        return tuple(sorted(self._props, key=lambda p: (p.op, p.kind.value)))

    def for_op(self, op: str) -> tuple[Property, ...]:
        return tuple(p for p in self.all() if p.op == op)

    def ops(self) -> tuple[str, ...]:
        return tuple(sorted({p.op for p in self._props}))

    def unmodeled(self, modeled_ops: Sequence[str]) -> tuple[str, ...]:
        """有性质声明但未在描述中建模的 op —— 报告而非静默丢弃。"""
        known = set(modeled_ops)
        return tuple(sorted(op for op in self.ops() if op not in known))


REGISTRY = PropertyRegistry()


# ---------------------------------------------------------------------------
# 便捷构造器
# ---------------------------------------------------------------------------


def commutative(op: str, fn: Callable[[Any, Any], Any], *, caveats: Sequence[str] = ()) -> Property:
    """f(a, b) == f(b, a)。"""
    return REGISTRY.add(
        Property(
            op=op,
            kind=PropertyKind.COMMUTATIVE,
            predicate=lambda a, b: fn(a, b) == fn(b, a),
            description=f"{op} 应满足交换律：f(a,b) == f(b,a)",
            caveats=tuple(caveats),
        )
    )


def idempotent(op: str, fn: Callable[[Any], Any], *, caveats: Sequence[str] = ()) -> Property:
    """f(f(a)) == f(a)。"""
    return REGISTRY.add(
        Property(
            op=op,
            kind=PropertyKind.IDEMPOTENT,
            predicate=lambda a: fn(fn(a)) == fn(a),
            description=f"{op} 应满足幂等：f(f(a)) == f(a)",
            caveats=tuple(caveats),
        )
    )


def identity_element(
    op: str, fn: Callable[[Any, Any], Any], element: Any, *, caveats: Sequence[str] = ()
) -> Property:
    """f(a, e) == a。"""
    return REGISTRY.add(
        Property(
            op=op,
            kind=PropertyKind.IDENTITY,
            predicate=lambda a: fn(a, element) == a,
            description=f"{op} 应有单位元 {element!r}：f(a,{element!r}) == a",
            caveats=tuple(caveats),
        )
    )


def property_for(
    op: str, description: str, *, caveats: Sequence[str] = ()
) -> Callable[[Callable[..., bool]], Property]:
    """自定义性质装饰器。"""

    def deco(fn: Callable[..., bool]) -> Property:
        return REGISTRY.add(
            Property(
                op=op,
                kind=PropertyKind.CUSTOM,
                predicate=fn,
                description=description,
                caveats=tuple(caveats),
            )
        )

    return deco


# ---------------------------------------------------------------------------
# 示例性质（对应 toy 描述的 op）
# ---------------------------------------------------------------------------
# 这些是**参考实现的**性质，用于验证骨架可运行。真实值语义接入（M3）后，
# predicate 将改为调用描述声明的值函数，从而真正校验描述本身。

import operator  # noqa: E402 - 置于此处以贴近使用点

VADD_COMMUTATIVE = commutative(
    "hivm.hir.vadd",
    operator.add,
    caveats=("浮点加法在含 NaN 时 a+b != b+a 的判定依赖 NaN 语义；整数与规范浮点下成立",),
)

VMUL_COMMUTATIVE = commutative(
    "hivm.hir.vmul",
    operator.mul,
    caveats=("同上：NaN/inf 场景需单独处理",),
)

VADD_IDENTITY = identity_element("hivm.hir.vadd", operator.add, 0)
VMUL_IDENTITY = identity_element("hivm.hir.vmul", operator.mul, 1)


@property_for(
    "hivm.hir.load",
    "load 后 store 回同一位置应等价于恒等（copy 的往返性）",
    caveats=("仅当无布局转换、无 padding 时成立；nd2nz 等布局变换不满足",),
)
def load_store_roundtrip(value: int) -> bool:
    # 参考实现：copy 语义即恒等
    loaded = value
    stored = loaded
    return stored == value


#: 刻意**不**声明的性质，留档以防未来误加：
#: - vsub 交换律（不成立）
#: - 浮点 vadd 结合律（不成立，且这是真实 kernel 精度问题的常见来源）
KNOWN_FALSE_PROPERTIES: dict[str, str] = {
    "hivm.hir.vsub:commutative": "减法不可交换——若有人声明此性质，说明描述理解有误",
    "hivm.hir.vadd:associative": "浮点加法不满足结合律；断言它成立会掩盖真实精度问题",
}


@dataclass
class CounterExample:
    """反例报告。"""

    prop: Property
    inputs: dict[str, Any]
    note: str = ""

    def render(self) -> str:
        return (
            f"反例：{self.prop.op} 违反 {self.prop.kind.value}\n"
            f"  性质：{self.prop.description}\n"
            f"  输入：{self.inputs}\n"
            + (f"  备注：{self.note}\n" if self.note else "")
            + (
                "  已知例外：\n" + "".join(f"    - {c}\n" for c in self.prop.caveats)
                if self.prop.caveats
                else ""
            )
        )


@dataclass
class PropertyReport:
    """性质测试汇总（供 T0.8 账本绊线消费）。"""

    checked: int = 0
    counterexamples: list[CounterExample] = field(default_factory=list)
    unmodeled_ops: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.counterexamples

    def render(self) -> str:
        lines = [f"性质测试：检查 {self.checked} 条"]
        if self.unmodeled_ops:
            lines.append(
                f"  ⚠️ 有性质声明但未建模的 op：{list(self.unmodeled_ops)}（待建模，非静默丢弃）"
            )
        for ce in self.counterexamples:
            lines.append(ce.render())
        if self.ok:
            lines.append("  未发现反例")
        return "\n".join(lines)
