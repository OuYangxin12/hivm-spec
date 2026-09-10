"""双模值层（T3.1，D4/OD1 方案 A）。

一份 op 语义、两种求值模式——业界模板是 Rosette（同一程序具体/符号双模）与
Triton 的 `TensorHandle`（numpy 具体执行）。

**为什么不直接用 z3 对象做双模**（OD1 选项 B）：z3 会把一个重依赖压进 M3 的
关键路径，而 M3 的不可裁剪内核（D8）是**具体执行差分**，符号只在 M4 才需要
求解能力。故此处只做"受限表达式子集"的轻量句柄：算术 / 比较 / select。
句柄不求解、不化简，只**记录结构**——M4 接 z3 时把这棵树翻译过去即可，
op 函数零改动（OD1 的验收标准）。

## 两种模式

- `ConcreteValue`：numpy 数组载体。真实数值，参与差分对拍与值哈希。
- `SymbolicValue`：表达式树。参与结构等价与"形状/dtype 是否一致"这类检查。

两者都实现 `Value` 协议，故 op 函数写一次即可跑两模。**不允许**混算
（`ConcreteValue + SymbolicValue`）——那会静默产生"一半具体一半符号"的结果，
读者无法判断它是不是真值。混算直接报错，见 `_require_same_mode`。

## 与 VIR 的关系

`vir.ValueSlot` 是**契约里的槽位**（mode / value_hash / symbol），本模块是
**运行时载体**。`slot_of()` 负责从载体生成槽位——契约与实现分离，这样 VIR
不必依赖 numpy，M4 加符号后端也不动 VIR。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from hivm_spec.vir import ValueSlot

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期
    pass

__all__ = [
    "VALUE_PRIMITIVES",
    "ConcreteValue",
    "SymbolicValue",
    "Value",
    "ValueError_",
    "ValueKernel",
    "concrete",
    "exp",
    "neg",
    "parse_value_kernel",
    "slot_of",
    "symbol",
    "value_hash",
]


class ValueError_(Exception):
    """双模值层的错误（与内建 ValueError 区分，避免误捕获）。

    命名带下划线后缀是刻意的：本层的错误**不应**被 `except ValueError` 顺手
    吞掉——它们表示描述或 op 函数写错了（如混算两种模式），属于必须让作者看到
    的问题，不是可恢复的数据异常。
    """


# ---------------------------------------------------------------------------
# 协议
# ---------------------------------------------------------------------------


@runtime_checkable
class Value(Protocol):
    """双模值的公共接口——op 函数只依赖它，故一份代码跑两模。"""

    @property
    def mode(self) -> Literal["concrete", "symbolic"]: ...

    @property
    def dtype(self) -> str:
        """dtype 名（如 "f32"/"f16"）。两模都必须有——差分对拍要比 dtype。"""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """形状。符号模式下也要求已知：M3 只处理静态 shape（M3 卡 §5 止损）。"""
        ...


# ---------------------------------------------------------------------------
# 具体模式
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConcreteValue:
    """具体值：numpy 数组载体。"""

    array: Any  # np.ndarray；不写死类型以免 VIR 层被迫依赖 numpy
    #: dtype 的**规范名**（f32/f16/bf16/i32…）。不直接用 numpy dtype 名是因为
    #: bf16 需要 ml_dtypes，而规范名要在没装 ml_dtypes 时也能表达（T3.3）。
    dtype_name: str = ""

    @property
    def mode(self) -> Literal["concrete"]:
        return "concrete"

    @property
    def dtype(self) -> str:
        return self.dtype_name or str(getattr(self.array, "dtype", ""))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(getattr(self.array, "shape", ()))


# ---------------------------------------------------------------------------
# 符号模式（受限子集：算术 / 比较 / select）
# ---------------------------------------------------------------------------

#: 允许的符号运算符。**封闭集合**——超出即报错，而不是默默构造一个 M4 翻译
#: 不了的节点。受限子集是 OD1 方案 A 的定义部分，不是暫时的偷懒。
_ARITH = frozenset({"add", "sub", "mul", "div", "neg"})
#: 一元逐元素运算。exp 是真实语料的常客（hivm.hir.vexp），
#: 早先只支持二元 elementwise，导致 vexp/vcast 被判成"无可执行值语义"。
_UNARY = frozenset({"neg", "exp"})
_COMPARE = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
_SELECT = frozenset({"select"})
SYMBOLIC_OPS = _ARITH | _COMPARE | _SELECT | _UNARY


@dataclass(frozen=True, slots=True)
class SymbolicValue:
    """符号值：受限表达式树。

    `op == ""` 表示叶子（自由符号），此时 `name` 有意义；否则是内部节点，
    `args` 为子表达式。不做化简——`x + 0` 仍记为 `add(x, 0)`。理由：化简会让
    "结构等价"的判定依赖化简器的完备性，而那是 M4 的求解器该负责的事。
    """

    op: str = ""
    name: str = ""
    args: tuple[SymbolicValue, ...] = ()
    dtype_name: str = "f32"
    shape_: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.op and self.op not in SYMBOLIC_OPS:
            raise ValueError_(
                f"符号运算符 {self.op!r} 不在受限子集内（OD1 方案 A）；"
                f"允许：{sorted(SYMBOLIC_OPS)}。"
                "扩子集要同步 M4 的 z3 翻译，不可只在此处放行"
            )
        if not self.op and not self.name:
            raise ValueError_("符号叶子必须有名字——匿名符号无法在报告里指认")

    @property
    def mode(self) -> Literal["symbolic"]:
        return "symbolic"

    @property
    def dtype(self) -> str:
        return self.dtype_name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.shape_

    def text(self) -> str:
        """可读表达式（确定性：同一棵树恒得同一串，FR8）。"""
        if not self.op:
            return self.name
        return f"{self.op}({', '.join(a.text() for a in self.args)})"


# ---------------------------------------------------------------------------
# 构造与工具
# ---------------------------------------------------------------------------


def concrete(array: Any, dtype_name: str = "") -> ConcreteValue:
    """包一个 numpy 数组为具体值。"""
    return ConcreteValue(array=array, dtype_name=dtype_name)


def symbol(name: str, *, dtype: str = "f32", shape: tuple[int, ...] = ()) -> SymbolicValue:
    """造一个自由符号（叶子）。"""
    return SymbolicValue(name=name, dtype_name=dtype, shape_=shape)


def _require_same_mode(a: Value, b: Value) -> None:
    """禁止混算。

    `concrete + symbolic` 若放行，结果既不是可对拍的真值、也不是完整的符号树，
    却长得像个正常值——这种"半真半假"最难发现，故直接报错（FR6 防自欺）。
    """
    if a.mode != b.mode:
        raise ValueError_(
            f"不允许混算 {a.mode} 与 {b.mode} 值——结果既非真值也非完整符号树，"
            "会静默污染对拍结论。请显式统一模式"
        )


def value_hash(v: Value) -> str:
    """值哈希：逐 op trace 与首发散点定位的基元（T3.6 的落点）。

    **只哈希值内容**（mode/dtype/shape + 归一化字节或表达式文本），不含 op 名、
    SSA 名、源位置——否则任何合法变换（重命名、重排、换实现）都会造成假发散
    （M3 卡 §4 要点 4）。

    浮点注意：哈希是**精确**的，容差比较不是哈希能表达的。故哈希只用于
    "快速筛选相同"与"定位第一个不同"，**判等仍走容差**（同一要点 4）。
    """
    h = hashlib.sha256()
    h.update(v.mode.encode())
    h.update(b"|")
    h.update(v.dtype.encode())
    h.update(b"|")
    h.update(repr(v.shape).encode())
    h.update(b"|")
    if isinstance(v, SymbolicValue):
        h.update(v.text().encode())
    elif isinstance(v, ConcreteValue):
        arr = v.array
        raw = getattr(arr, "tobytes", None)
        if raw is None:
            h.update(repr(arr).encode())
        else:
            # C 序归一化：同一逻辑值在不同内存布局（转置视图等）下必须同哈希，
            # 否则布局变化会伪装成语义发散。
            import numpy as np

            h.update(np.ascontiguousarray(arr).tobytes())
    else:  # pragma: no cover - 协议实现者应属上述两类
        raise ValueError_(f"未知值载体：{type(v).__name__}")
    return "sha256:" + h.hexdigest()


def slot_of(v: Value) -> ValueSlot:
    """载体 → VIR 契约里的 `ValueSlot`。

    契约与实现分离：VIR 不依赖 numpy，M4 换符号后端也不动 VIR（D6）。
    """
    if isinstance(v, SymbolicValue):
        return ValueSlot(mode="symbolic", symbol=v.text())
    return ValueSlot(mode="concrete", value_hash=value_hash(v))


# ---------------------------------------------------------------------------
# 受限子集的运算（op 函数通过这些写一次跑两模）
# ---------------------------------------------------------------------------


def _binary(kind: str, a: Value, b: Value) -> Value:
    _require_same_mode(a, b)
    if isinstance(a, SymbolicValue) and isinstance(b, SymbolicValue):
        return SymbolicValue(
            op=kind, args=(a, b), dtype_name=a.dtype_name, shape_=a.shape_ or b.shape_
        )
    if isinstance(a, ConcreteValue) and isinstance(b, ConcreteValue):
        import numpy as np

        ops = {
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "div": np.divide,
            "eq": np.equal,
            "ne": np.not_equal,
            "lt": np.less,
            "le": np.less_equal,
            "gt": np.greater,
            "ge": np.greater_equal,
        }
        fn = ops.get(kind)
        if fn is None:  # pragma: no cover - kind 由内部调用限定
            raise ValueError_(f"具体模式未实现运算 {kind!r}")
        out = fn(a.array, b.array)
        # 比较结果的 dtype 是 bool，不能继承输入 dtype——否则值哈希会声称
        # 一个 bool 掩码是 f32，掩码变化就可能被误读为数值发散。
        dt = "i1" if kind in _COMPARE else a.dtype_name
        return ConcreteValue(array=out, dtype_name=dt)
    raise ValueError_(f"不支持的值载体组合：{type(a).__name__} 与 {type(b).__name__}")


def _unary(kind: str, a: Value) -> Value:
    """一元逐元素运算（双模）。

    与 `_binary` 分开而不是塞进去凑数：一元的参数个数、符号节点的 args 形状
    都不同，混在一起会让"参数不匹配"这类错误变得难以定位。
    """
    if isinstance(a, SymbolicValue):
        return SymbolicValue(op=kind, args=(a,), dtype_name=a.dtype_name, shape_=a.shape_)
    if isinstance(a, ConcreteValue):
        import numpy as np

        ops = {"neg": np.negative, "exp": np.exp}
        fn = ops.get(kind)
        if fn is None:  # pragma: no cover - kind 由内部调用限定
            raise ValueError_(f"具体模式未实现一元运算 {kind!r}")
        return ConcreteValue(array=fn(a.array), dtype_name=a.dtype_name)
    raise ValueError_(f"不支持的值载体：{type(a).__name__}")


def neg(a: Value) -> Value:
    return _unary("neg", a)


def exp(a: Value) -> Value:
    return _unary("exp", a)


def add(a: Value, b: Value) -> Value:
    return _binary("add", a, b)


def sub(a: Value, b: Value) -> Value:
    return _binary("sub", a, b)


def mul(a: Value, b: Value) -> Value:
    return _binary("mul", a, b)


def div(a: Value, b: Value) -> Value:
    return _binary("div", a, b)


def eq(a: Value, b: Value) -> Value:
    return _binary("eq", a, b)


def lt(a: Value, b: Value) -> Value:
    return _binary("lt", a, b)


def select(cond: Value, a: Value, b: Value) -> Value:
    """三元选择——控制流在符号模式下的表达方式（T3.2 的 if 归约目标）。"""
    _require_same_mode(cond, a)
    _require_same_mode(a, b)
    if isinstance(cond, SymbolicValue) and isinstance(a, SymbolicValue):
        assert isinstance(b, SymbolicValue)
        return SymbolicValue(
            op="select", args=(cond, a, b), dtype_name=a.dtype_name, shape_=a.shape_
        )
    import numpy as np

    assert isinstance(cond, ConcreteValue) and isinstance(a, ConcreteValue)
    assert isinstance(b, ConcreteValue)
    return ConcreteValue(array=np.where(cond.array, a.array, b.array), dtype_name=a.dtype_name)


# ---------------------------------------------------------------------------
# 环境：op 函数的求值上下文
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Env:
    """求值环境：SSA 名 → 值。

    刻意可变（非 frozen）：解释执行本质是逐步写入。但**同名重绑定报错**——
    SSA 的单赋值性质若被悄悄破坏，值哈希 trace 会错位，首发散点定位就会指错
    地方（T3.6 依赖此不变量）。
    """

    values: dict[str, Value] = field(default_factory=dict)

    def bind(self, name: str, v: Value) -> None:
        if name in self.values:
            raise ValueError_(
                f"SSA 值 {name!r} 重复绑定——单赋值被破坏会让值哈希 trace 错位，"
                "首发散点会指向错误的 op"
            )
        self.values[name] = v

    def get(self, name: str) -> Value:
        try:
            return self.values[name]
        except KeyError:
            raise ValueError_(f"未绑定的值：{name!r}") from None


# ---------------------------------------------------------------------------
# 值语义原语：把描述里的 value= 声明变成可执行内核（T3.1 定稿部分）
# ---------------------------------------------------------------------------

#: `value=` 声明支持的原语。**封闭集合**——描述里写了集合外的形式，静态检查
#: 就该报错，而不是运行时才发现"这条语义其实没被执行过"。
#:
#: 形态（取自 specs/toy.py 的既有声明，不另造语法）：
#:   copy(src, into=dst)                 —— 搬运：dst ← src
#:   elementwise(add, a, b, into=out)     —— 逐元素二元：out ← a ⊕ b
VALUE_PRIMITIVES = frozenset({"copy", "elementwise"})

#: elementwise 支持的二元运算（受限子集内，与 SYMBOLIC_OPS 对齐）
_ELEMENTWISE_FNS = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
}

#: 一元逐元素运算表（elementwise(fn, src) 形态）
_UNARY_FNS = {
    "neg": neg,
    "exp": exp,
}


@dataclass(frozen=True, slots=True)
class ValueKernel:
    """一条 `value=` 声明解析后的可执行形式。

    刻意只存**结构**（原语名 + 参数名 + 输出名），不存闭包：这样它可被序列化
    进配置文档、可被 spec-gate 检查、也能在报告里回显"这个结论用的是哪条语义"。
    闭包做不到这三件事。
    """

    primitive: str
    #: 输入参数名（按声明顺序）
    inputs: tuple[str, ...]
    #: 输出参数名（`into=` 的目标）
    output: str
    #: elementwise 的二元运算名；copy 为 ""
    fn: str = ""

    def apply(self, args: dict[str, Value]) -> Value:
        """按参数绑定求值——两模通用（走 values 层的双模运算）。"""
        try:
            operands = [args[n] for n in self.inputs]
        except KeyError as exc:
            raise ValueError_(
                f"值语义 {self.text()} 缺少实参 {exc.args[0]!r}——描述声明的参数名与调用不一致"
            ) from None
        if self.primitive == "copy":
            return operands[0]
        if len(operands) == 1:
            # 一元 elementwise（如 vexp 的 elementwise(exp, src)）
            unary_fn = _UNARY_FNS.get(self.fn)
            if unary_fn is None:
                raise ValueError_(
                    f"一元 elementwise 不支持运算 {self.fn!r}；允许：{sorted(_UNARY_FNS)}"
                )
            return unary_fn(operands[0])
        binary = _ELEMENTWISE_FNS[self.fn]
        acc = operands[0]
        for nxt in operands[1:]:
            acc = binary(acc, nxt)
        return acc

    def text(self) -> str:
        if self.primitive == "copy":
            return f"copy({self.inputs[0]}, into={self.output})"
        return f"elementwise({self.fn}, {', '.join(self.inputs)}, into={self.output})"


def parse_value_kernel(decl: str) -> ValueKernel:
    """解析 `value=` 声明字符串。

    只认 `VALUE_PRIMITIVES` 里的形态。**解析失败必须报错**——此前这些字符串
    从未被执行过（只做存在性检查），若解析器悄悄跳过不认识的声明，等价验证
    就会在"没有值语义"的情况下报 OK，那是最坏的假阴性（FR1/FR6）。
    """
    import re as _re

    text = decl.strip()
    m = _re.fullmatch(r"(\w+)\((.*)\)", text, _re.DOTALL)
    if not m:
        raise ValueError_(f"值语义声明无法解析：{decl!r}（期望 primitive(args...) 形态）")
    prim, body = m.group(1), m.group(2)
    if prim not in VALUE_PRIMITIVES:
        raise ValueError_(
            f"未知值语义原语 {prim!r}；允许：{sorted(VALUE_PRIMITIVES)}。"
            "新增原语须同时提供双模实现，不可只在描述侧写名字"
        )
    parts = [p.strip() for p in body.split(",") if p.strip()]
    into = [p for p in parts if p.startswith("into=")]
    if len(into) != 1:
        raise ValueError_(f"值语义 {decl!r} 必须恰好有一个 into= 输出目标")
    output = into[0][len("into=") :].strip()
    positional = [p for p in parts if not p.startswith("into=")]
    if prim == "copy":
        if len(positional) != 1:
            raise ValueError_(f"copy 需恰好一个源参数，得到 {positional}")
        return ValueKernel(primitive="copy", inputs=(positional[0],), output=output)
    # elementwise(fn, a[, b, ...])：一元与多元都支持
    if len(positional) < 2:
        raise ValueError_(f"elementwise 需 fn 与至少一个操作数，得到 {positional}")
    fn = positional[0]
    allowed = set(_ELEMENTWISE_FNS) | set(_UNARY_FNS)
    if fn not in allowed:
        raise ValueError_(f"elementwise 运算 {fn!r} 不在受限子集内；允许：{sorted(allowed)}")
    if len(positional) == 2 and fn not in _UNARY_FNS:
        raise ValueError_(
            f"elementwise({fn}, …) 是多元运算，需至少两个操作数；一元可用：{sorted(_UNARY_FNS)}"
        )
    return ValueKernel(primitive="elementwise", inputs=tuple(positional[1:]), output=output, fn=fn)
