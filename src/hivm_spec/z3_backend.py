"""SymbolicValue → z3 表达式的翻译（T4.1）。

## 本模块是"接后端"，不是重新设计

`SymbolicValue`（T3.1 定稿）已经是一棵受限表达式树：`SYMBOLIC_OPS` 是封闭集合，
不做化简。本模块把它逐节点翻译成 z3 项，**op 函数一行不改**——这正是 T4.1 的
验收标准"op 函数零改动切换符号模式"的含义。

## 数值语义：Real 而非 FP

z3 支持 IEEE-754 浮点（`FPSort`），但本模块用 `Real`（无限精度有理数）。这是
一个**有意的、必须披露的近似**：

- 用 Real：证出的 UNSAT 是"在实数语义下无反例"，不等于"在 f32 下无反例"——
  浮点的舍入与结合律缺失不被建模；
- 用 FP：语义精确，但非线性 FP 约束的求解常常超时，实用性大打折扣。

选 Real 的理由是 M4 定位于**结构等价**（两侧是否做了相同的事），而**数值等价
已由 M3 的具体档 + 容差负责**。二者结论并列，各自声明适用范围（M4 卡 §4.1）。

这条近似会随结论一起输出，不是藏在实现里的秘密。

## exp 走未解释函数

SMT 没有超越函数。`exp` 被建模为未解释函数 `f`，它只保证**同实参必同值**
（congruence）：`x == y → f(x) == f(y)`。

于是"两侧都调用 f 且实参相同"可判等价——但这**只能证明两侧做了相同的事，
不能证明那件事是对的**（M4 卡 §4.2）。报告措辞必须区分，故翻译结果会带上
`uses_uninterpreted` 标记。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hivm_spec.symbolic import require_z3
from hivm_spec.values import SYMBOLIC_OPS, SymbolicValue

__all__ = [
    "TRANSLATOR_VERSION",
    "Translation",
    "UnsupportedSymbolicOp",
    "translate",
]

TRANSLATOR_VERSION = "0.1.0"

#: 直接映射到 z3 算术/比较运算的 op。
_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}

#: 需要未解释函数的 op：SMT 无超越函数。
_UNINTERPRETED = frozenset({"exp"})


class UnsupportedSymbolicOp(ValueError):
    """受限子集内出现了本翻译器不认识的 op。

    这**不是**"用户输入错误"，而是**实现债**：`SYMBOLIC_OPS` 是封闭集合，
    集合里的每一项都应当有翻译。若触发本异常，说明有人扩了子集却没同步翻译
    ——正是 `values.py` 里那句"扩子集要同步 M4 的 z3 翻译，不可只在此处放行"
    所警告的情形。
    """


@dataclass(slots=True)
class Translation:
    """一次翻译的结果与其副作用。"""

    #: z3 表达式（类型为 z3.ExprRef，此处不标注以免核心层硬依赖 z3）
    expr: Any
    #: 叶子名 → z3 常量，供调用方施加约束或取模型
    symbols: dict[str, Any] = field(default_factory=dict)
    #: 用到的未解释函数名（如 exp）。非空 ⇒ 结论只能声称"两侧做了相同的事"
    uninterpreted: frozenset[str] = frozenset()
    #: 除法是否出现——除零在 Real 语义下是未定义的，须披露
    has_division: bool = False

    @property
    def uses_uninterpreted(self) -> bool:
        return bool(self.uninterpreted)


def translate(
    value: SymbolicValue,
    *,
    symbols: dict[str, Any] | None = None,
    functions: dict[str, Any] | None = None,
) -> Translation:
    """把 `SymbolicValue` 翻译成 z3 表达式。

    `symbols`/`functions` 可跨多次调用共享，用于**取回**两侧共用的常量与函数
    （施加额外约束、从模型里读反例时需要）。

    注意：共享**不是正确性的前提**——z3 按名字 intern 常量与未解释函数，
    `z3.Real("x")` 两次调用得到的是同一个 AST 节点（实测 `a.eq(b) is True`）。
    早先本处注释声称"不共享会导致两侧符号不同、判定恒为 SAT"，那是**错的**，
    已按实测更正：不共享只是拿不到句柄，不影响判定结果。
    """
    z3 = require_z3()
    env: dict[str, Any] = symbols if symbols is not None else {}
    fns: dict[str, Any] = functions if functions is not None else {}
    used_fns: set[str] = set()
    state = {"division": False}

    def walk(node: SymbolicValue) -> Any:
        # 叶子：自由符号
        if not node.op:
            if node.name not in env:
                env[node.name] = z3.Real(node.name)
            return env[node.name]

        if node.op not in SYMBOLIC_OPS:  # pragma: no cover - SymbolicValue 已守
            raise UnsupportedSymbolicOp(f"{node.op!r} 不在受限子集内")

        args = [walk(a) for a in node.args]

        if node.op == "neg":
            _expect(node, args, 1)
            return -args[0]

        if node.op == "div":
            _expect(node, args, 2)
            # 除零在 Real 语义下未定义；z3 会给它一个任意但一致的解释。
            # 不在此处加 b != 0 约束——那会**改变**被验证的语义（把
            # "这段代码可能除零"悄悄变成"假设它不会"）。改为披露。
            state["division"] = True
            return args[0] / args[1]

        if node.op == "select":
            _expect(node, args, 3)
            return z3.If(args[0], args[1], args[2])

        if node.op in _UNINTERPRETED:
            _expect(node, args, 1)
            used_fns.add(node.op)
            if node.op not in fns:
                fns[node.op] = z3.Function(node.op, z3.RealSort(), z3.RealSort())
            return fns[node.op](args[0])

        handler = _BINARY.get(node.op)
        if handler is None:
            raise UnsupportedSymbolicOp(
                f"受限子集包含 {node.op!r} 但翻译器未实现它——"
                "扩 SYMBOLIC_OPS 时必须同步扩本模块，否则子集形同虚设"
            )
        _expect(node, args, 2)
        return handler(args[0], args[1])

    expr = walk(value)
    return Translation(
        expr=expr,
        symbols=env,
        uninterpreted=frozenset(used_fns),
        has_division=state["division"],
    )


def _expect(node: SymbolicValue, args: list[Any], n: int) -> None:
    if len(args) != n:
        raise UnsupportedSymbolicOp(
            f"{node.op!r} 需要 {n} 个操作数，实际 {len(args)} 个——表达式树构造有误"
        )
