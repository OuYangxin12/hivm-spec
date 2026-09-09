"""T3.1 双模值层测试。

验收核心（milestone-plan §5 T3.1）：**同一 op 函数双模式输出一致**。
这里"一致"不是指数值相等（符号模式没有数值），而是指：同一份 op 代码在两种
模式下都能跑通，且结构性属性（dtype/shape/运算次序）一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from hivm_spec.values import (
    ConcreteValue,
    Env,
    SymbolicValue,
    ValueError_,
    add,
    concrete,
    lt,
    mul,
    select,
    slot_of,
    symbol,
    value_hash,
)

# ---------------------------------------------------------------------------
# T3.1 核心验收：一份 op 函数跑两模
# ---------------------------------------------------------------------------


def _fused_multiply_add(a, b, c):  # type: ignore[no-untyped-def]
    """被测的"op 函数"：刻意只用双模 API，不碰 numpy/符号构造器。

    这正是 OD1 方案 A 的验收对象——op 作者写一次，具体与符号两模都能跑。
    """
    return add(mul(a, b), c)


def test_same_op_function_runs_in_both_modes() -> None:
    """同一函数、两种模式，都能求值且结构属性一致（T3.1 验收）。"""
    ca = concrete(np.array([1.0, 2.0], dtype=np.float32), "f32")
    cb = concrete(np.array([3.0, 4.0], dtype=np.float32), "f32")
    cc = concrete(np.array([5.0, 6.0], dtype=np.float32), "f32")
    cres = _fused_multiply_add(ca, cb, cc)

    sa = symbol("a", dtype="f32", shape=(2,))
    sb = symbol("b", dtype="f32", shape=(2,))
    sc = symbol("c", dtype="f32", shape=(2,))
    sres = _fused_multiply_add(sa, sb, sc)

    # 具体模式给出真实数值
    assert isinstance(cres, ConcreteValue)
    np.testing.assert_allclose(cres.array, np.array([8.0, 14.0], dtype=np.float32))

    # 符号模式给出表达式结构，且运算次序与具体模式一致
    assert isinstance(sres, SymbolicValue)
    assert sres.text() == "add(mul(a, b), c)"

    # 两模的结构性属性一致——这才是"双模一致"的可检验含义
    assert cres.dtype == sres.dtype == "f32"
    assert cres.shape == sres.shape == (2,)
    assert cres.mode == "concrete" and sres.mode == "symbolic"


def test_select_expresses_control_flow_in_both_modes() -> None:
    """select 是 T3.2 里 scf.if 的归约目标，必须双模可用。"""
    cond_c = lt(
        concrete(np.array([1.0, 9.0], dtype=np.float32), "f32"),
        concrete(np.array([5.0, 5.0], dtype=np.float32), "f32"),
    )
    got = select(
        cond_c,
        concrete(np.array([10.0, 10.0], dtype=np.float32), "f32"),
        concrete(np.array([20.0, 20.0], dtype=np.float32), "f32"),
    )
    assert isinstance(got, ConcreteValue)
    np.testing.assert_allclose(got.array, np.array([10.0, 20.0], dtype=np.float32))
    # 比较结果的 dtype 必须是 i1，不能继承 f32——否则值哈希会声称一个 bool
    # 掩码是 f32，掩码变化可能被误读成数值发散
    assert cond_c.dtype == "i1"

    s = select(
        lt(symbol("x", shape=(2,)), symbol("y", shape=(2,))),
        symbol("p", shape=(2,)),
        symbol("q", shape=(2,)),
    )
    assert isinstance(s, SymbolicValue)
    assert s.text() == "select(lt(x, y), p, q)"


# ---------------------------------------------------------------------------
# 防自欺：混算、越界子集、重绑定
# ---------------------------------------------------------------------------


def test_mixing_modes_is_rejected() -> None:
    """混算必须报错——半具体半符号的结果最难被发现（FR6）。"""
    with pytest.raises(ValueError_, match="不允许混算"):
        add(concrete(np.array([1.0], dtype=np.float32), "f32"), symbol("x", shape=(1,)))


def test_symbolic_subset_is_closed() -> None:
    """受限子集是封闭的：越界运算符直接报错，不构造 M4 翻译不了的节点。"""
    with pytest.raises(ValueError_, match="不在受限子集内"):
        SymbolicValue(op="matmul", args=(symbol("a"), symbol("b")))
    with pytest.raises(ValueError_, match="匿名符号"):
        SymbolicValue()


def test_env_rejects_ssa_rebinding() -> None:
    """SSA 单赋值被破坏会让首发散点指错 op（T3.6 依赖此不变量）。"""
    env = Env()
    env.bind("%0", symbol("a"))
    assert env.get("%0").mode == "symbolic"
    with pytest.raises(ValueError_, match="重复绑定"):
        env.bind("%0", symbol("b"))
    with pytest.raises(ValueError_, match="未绑定"):
        env.get("%nope")


# ---------------------------------------------------------------------------
# 值哈希：T3.6 的基元
# ---------------------------------------------------------------------------


def test_value_hash_ignores_layout_but_tracks_content() -> None:
    """哈希只看值内容：布局变化不得伪装成语义发散（M3 卡 §4 要点 4）。"""
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    # 真正的非连续视图：F 序副本与 base 逻辑相等但内存布局不同
    f_order = np.asfortranarray(base)
    assert not f_order.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(base, f_order)
    assert value_hash(concrete(base, "f32")) == value_hash(concrete(f_order, "f32"))

    # 转置也不该改变"同一逻辑值"的哈希：转置后再转回，内容不变
    assert value_hash(concrete(base, "f32")) == value_hash(concrete(base.T.T, "f32"))

    # 内容变化必须改变哈希
    other = base.copy()
    other[0, 0] += 1.0
    assert value_hash(concrete(base, "f32")) != value_hash(concrete(other, "f32"))

    # 形状变化也必须改变哈希——同样的字节、不同的形状不是同一个值
    assert value_hash(concrete(base, "f32")) != value_hash(concrete(base.reshape(3, 2), "f32"))


def test_value_hash_is_deterministic_and_mode_aware() -> None:
    """同输入恒同哈希（FR8）；且不同模式不得撞哈希。"""
    a = concrete(np.array([1.0, 2.0], dtype=np.float32), "f32")
    assert value_hash(a) == value_hash(a)
    s = symbol("v", dtype="f32", shape=(2,))
    assert value_hash(a) != value_hash(s)
    # 同名同结构的符号 → 同哈希（确定性）
    assert value_hash(s) == value_hash(symbol("v", dtype="f32", shape=(2,)))


def test_slot_of_maps_to_vir_contract() -> None:
    """载体 → ValueSlot：契约与实现分离，VIR 不依赖 numpy（D6）。"""
    cs = slot_of(concrete(np.array([1.0], dtype=np.float32), "f32"))
    assert cs.mode == "concrete"
    assert cs.value_hash.startswith("sha256:")
    assert cs.symbol == ""

    ss = slot_of(add(symbol("x"), symbol("y")))
    assert ss.mode == "symbolic"
    assert ss.symbol == "add(x, y)"
    assert ss.value_hash == ""


# ---------------------------------------------------------------------------
# T3.1：value= 声明 → 可执行内核
# ---------------------------------------------------------------------------


def test_value_declarations_parse_and_roundtrip() -> None:
    """描述里的既有声明形态必须可解析，且往返一致。"""
    from hivm_spec.values import parse_value_kernel

    k = parse_value_kernel("copy(src, into=dst)")
    assert k.primitive == "copy"
    assert k.inputs == ("src",)
    assert k.output == "dst"
    assert k.text() == "copy(src, into=dst)"

    e = parse_value_kernel("elementwise(add, a, b, into=out)")
    assert e.primitive == "elementwise"
    assert e.fn == "add"
    assert e.inputs == ("a", "b")
    assert e.text() == "elementwise(add, a, b, into=out)"


def test_value_kernel_executes_in_both_modes() -> None:
    """同一条声明，两模都能求值——这是 value= 不再是死字符串的证据。

    此前 `value=` 只做存在性检查、从未被执行；等价验证若在"值语义没被真正
    执行"的情况下报 OK，就是最坏的假阴性（FR1/FR6）。
    """
    from hivm_spec.values import parse_value_kernel

    k = parse_value_kernel("elementwise(add, a, b, into=out)")

    c = k.apply(
        {
            "a": concrete(np.array([2.0, 3.0], dtype=np.float32), "f32"),
            "b": concrete(np.array([10.0, 20.0], dtype=np.float32), "f32"),
        }
    )
    assert isinstance(c, ConcreteValue)
    np.testing.assert_allclose(c.array, np.array([12.0, 23.0], dtype=np.float32))

    s = k.apply({"a": symbol("a", shape=(2,)), "b": symbol("b", shape=(2,))})
    assert isinstance(s, SymbolicValue)
    assert s.text() == "add(a, b)"


def test_unparseable_value_declaration_is_loud() -> None:
    """解析失败必须报错，不能静默跳过。

    静默跳过等于"这条 op 没有值语义"，而等价验证会照常给出结论——那是把
    "没检查"包装成"检查通过"。
    """
    from hivm_spec.values import parse_value_kernel

    with pytest.raises(ValueError_, match="无法解析"):
        parse_value_kernel("this is not a call")
    with pytest.raises(ValueError_, match="未知值语义原语"):
        parse_value_kernel("magic(a, into=b)")
    with pytest.raises(ValueError_, match="into="):
        parse_value_kernel("copy(src)")
    with pytest.raises(ValueError_, match="不在受限子集内"):
        parse_value_kernel("elementwise(matmul, a, b, into=out)")


def test_every_toy_value_declaration_is_executable() -> None:
    """toy 描述的每条非逃生舱 value= 都必须真能执行（回归护栏）。

    锁定的是"描述与值层不脱节"：新增 op 时若写了值层不支持的声明，此测试失败，
    而不是等到等价验证阶段悄悄少验一个 op。
    """
    import importlib.util
    from pathlib import Path

    from hivm_spec.spec import HostFn
    from hivm_spec.values import parse_value_kernel

    spec_path = Path(__file__).resolve().parents[1] / "specs" / "toy.py"
    loader = importlib.util.spec_from_file_location("toy_for_values", spec_path)
    assert loader and loader.loader
    mod = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod)

    checked = 0
    for op in mod.spec.ops:
        if not op.value or isinstance(op.value, HostFn):
            continue
        kernel = parse_value_kernel(op.value)
        args = {n: symbol(n, shape=(2,)) for n in kernel.inputs}
        assert kernel.apply(args).mode == "symbolic"
        checked += 1
    assert checked >= 4, "toy 应有多条值语义声明；数量骤减说明描述被改动"
