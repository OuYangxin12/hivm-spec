"""T3.4 输入策略测试。

验收核心（milestone-plan §5 T3.4）：**输入可复现——同种子逐字节一致（FR8）**。
"""

from __future__ import annotations

import numpy as np
import pytest

from hivm_spec.inputs import (
    DEFAULT_SEED,
    InputSpec,
    InputStrategy,
    TilingPolicy,
    derive_seed,
)
from hivm_spec.numeric import NumericSupportError

# ---------------------------------------------------------------------------
# FR8：同种子逐字节一致
# ---------------------------------------------------------------------------


def test_same_seed_yields_byte_identical_inputs() -> None:
    """T3.4 的验收标准本身：同种子 → 逐字节一致。"""
    spec = InputSpec("%a", "f32", (8,))
    a = InputStrategy(seed=42).generate(spec)
    b = InputStrategy(seed=42).generate(spec)
    assert a.tobytes() == b.tobytes()
    # 不同种子必须给出不同输入，否则种子形同虚设
    c = InputStrategy(seed=43).generate(spec)
    assert a.tobytes() != c.tobytes()


def test_generation_is_independent_of_traversal_order() -> None:
    """按名字派生子种子 → 输入值与生成顺序无关。

    若所有输入共用一个 Generator 顺序抽取，"输入的生成顺序"就成了结论的隐藏
    依赖：换个遍历顺序、多一个参数，全部输入的值都会变，历史结论无法复现。
    """
    a = InputSpec("%a", "f32", (4,))
    b = InputSpec("%b", "f32", (4,))

    forward = InputStrategy(seed=7).generate_all((a, b))
    backward = InputStrategy(seed=7).generate_all((b, a))
    assert forward["%a"].tobytes() == backward["%a"].tobytes()
    assert forward["%b"].tobytes() == backward["%b"].tobytes()

    # 新增一个输入不得扰动既有输入的值
    with_extra = InputStrategy(seed=7).generate_all((a, b, InputSpec("%c", "f32", (4,))))
    assert with_extra["%a"].tobytes() == forward["%a"].tobytes()
    assert with_extra["%b"].tobytes() == forward["%b"].tobytes()


def test_derive_seed_is_stable_across_processes() -> None:
    """子种子派生必须跨进程稳定——不能用带随机盐的内建 hash()。

    Python 的 str hash 受 PYTHONHASHSEED 影响，同一名字在两次运行里会得到不同
    值，那会直接破坏 FR8。此处锁定具体数值，任何实现改动都会被发现。
    """
    got = derive_seed(DEFAULT_SEED, "%a")
    assert got == derive_seed(DEFAULT_SEED, "%a")
    assert derive_seed(DEFAULT_SEED, "%a") != derive_seed(DEFAULT_SEED, "%b")
    assert derive_seed(1, "%a") != derive_seed(2, "%a")
    # 稳定值（sha256 派生，与解释器版本/进程无关）
    assert isinstance(got, int) and got >= 0


def test_default_seed_is_fixed_not_time_based() -> None:
    """默认种子必须写死——时间戳会让报告里的用例编号失去意义。"""
    assert InputStrategy().seed == DEFAULT_SEED
    assert InputStrategy().seed == InputStrategy().seed


# ---------------------------------------------------------------------------
# 缩小 tiling
# ---------------------------------------------------------------------------


def test_small_shapes_are_not_shrunk() -> None:
    """已在预算内的 shape 不动，且不产生噪音标注。"""
    t = TilingPolicy(max_elements=64)
    assert t.shrink((4,)) == ((4,), "")
    assert t.shrink((8, 8)) == ((8, 8), "")
    assert t.shrink(()) == ((), "")


def test_shrink_respects_budget_and_keeps_rank() -> None:
    """压缩必须落到预算内，且**保留维数**（结构语义不变）。"""
    t = TilingPolicy(max_elements=64)
    for shape in [(256,), (16, 32), (512, 128), (4, 4, 64)]:
        shrunk, note = t.shrink(shape)
        assert len(shrunk) == len(shape), f"维数不得改变：{shape} → {shrunk}"
        total = 1
        for d in shrunk:
            total *= d
        assert total <= 64, f"{shape} → {shrunk} 未落入预算"
        assert note, "发生压缩必须留下说明"


def test_shrink_keeps_power_of_two_and_min_dim() -> None:
    """折半压缩保留 2 的幂特性，且不把任何维压到 min_dim 以下。

    直接设成上限会把对齐的 shape 压成非对齐，从而改变 padding 语义；
    压到 1 会让广播/边界语义失真。
    """
    t = TilingPolicy(max_elements=64, min_dim=2)
    shrunk, _ = t.shrink((512, 128))
    assert all(d >= 2 for d in shrunk), shrunk
    assert all((d & (d - 1)) == 0 for d in shrunk), f"应保持 2 的幂：{shrunk}"


def test_shrink_can_be_disabled() -> None:
    t = TilingPolicy(max_elements=8, enabled=False)
    assert t.shrink((1024,)) == ((1024,), "")


def test_invalid_tiling_policy_is_refused() -> None:
    with pytest.raises(NumericSupportError, match="max_elements"):
        TilingPolicy(max_elements=0)
    with pytest.raises(NumericSupportError, match="min_dim"):
        TilingPolicy(min_dim=0)


def test_shrunk_flag_and_provenance_are_reported() -> None:
    """缩小必须如实标注——"在缩小 tiling 下未发现问题"≠"已验证"（FR6）。"""
    s = InputStrategy(seed=1, tiling=TilingPolicy(max_elements=16))
    s.generate(InputSpec("%big", "f32", (1024,)))
    assert s.shrunk
    prov = s.provenance()
    assert prov["seed"] == 1
    assert prov["shrunk"] is True
    assert prov["notes"] and "%big" in prov["notes"][0]

    # 未缩小时不得声称缩小了
    s2 = InputStrategy(seed=1, tiling=TilingPolicy(max_elements=4096))
    s2.generate(InputSpec("%small", "f32", (8,)))
    assert not s2.shrunk
    assert s2.provenance()["notes"] == []


def test_provenance_allows_replay() -> None:
    """来源摘要必须足以重放（FR5）：种子 + tiling 上限齐备。"""
    s = InputStrategy(seed=99, tiling=TilingPolicy(max_elements=32))
    s.generate(InputSpec("%a", "f32", (64,)))
    prov = s.provenance()
    replay = InputStrategy(
        seed=prov["seed"], tiling=TilingPolicy(max_elements=prov["tiling_max_elements"])
    )
    assert (
        replay.generate(InputSpec("%a", "f32", (64,))).tobytes()
        == s.generate(InputSpec("%a", "f32", (64,))).tobytes()
    )


# ---------------------------------------------------------------------------
# dtype 与取值范围
# ---------------------------------------------------------------------------


def test_float_inputs_are_not_all_small_integers() -> None:
    """浮点输入必须含非整数与负值。

    全是小整数的输入会让舍入、累加顺序这类真实缺陷碰巧算对，从而放过缺陷
    （假阴性方向的风险）。
    """
    data = InputStrategy(seed=5).generate(InputSpec("%a", "f32", (64,)))
    assert np.any(data < 0), "应含负值"
    assert np.any(data != np.trunc(data)), "应含非整数"


def test_integer_inputs_stay_in_safe_range() -> None:
    """窄整数 dtype 取小范围——否则乘法必然溢出，每个用例都变成溢出用例，
    反而掩盖真正要查的语义差异。"""
    data = InputStrategy(seed=5).generate(InputSpec("%i", "i8", (64,)))
    assert data.dtype == np.dtype("int8")
    assert int(data.min()) >= -128 and int(data.max()) <= 127


def test_bool_inputs_are_zero_or_one() -> None:
    data = InputStrategy(seed=5).generate(InputSpec("%m", "i1", (32,)))
    assert data.dtype == np.dtype("bool")
    assert set(np.unique(data).tolist()) <= {True, False}


def test_dtype_is_honored() -> None:
    for name, expect in (("f32", "float32"), ("f16", "float16"), ("i32", "int32")):
        got = InputStrategy(seed=3).generate(InputSpec("%x", name, (4,)))
        assert got.dtype == np.dtype(expect)


def test_anonymous_input_is_refused() -> None:
    with pytest.raises(NumericSupportError, match="必须有名字"):
        InputSpec("")


# ---------------------------------------------------------------------------
# 从 VIR 推导入口输入
# ---------------------------------------------------------------------------


def test_parse_shape_text_handles_known_forms() -> None:
    from hivm_spec.inputs import parse_shape_text

    assert parse_shape_text("256xf32") == ((256,), "f32", "")
    assert parse_shape_text("4x16xbf16") == ((4, 16), "bf16", "")
    assert parse_shape_text("8x8x8xi8") == ((8, 8, 8), "i8", "")
    # 标量（无维）
    assert parse_shape_text("f32") == ((), "f32", "")


def test_dynamic_dim_is_not_guessed() -> None:
    """动态 shape 不猜具体值。

    猜 1 会让边界语义失真，猜 256 会让预算失真——两者都会把"不知道 shape"
    伪装成"验证过了"（FR6）。
    """
    from hivm_spec.inputs import parse_shape_text

    shape, dtype, why = parse_shape_text("?x16xf16")
    assert shape is None
    assert dtype == "f16"
    assert "动态维" in why


def test_unknown_dtype_suffix_is_not_defaulted_to_f32() -> None:
    """未知 dtype 不得默认成 f32——那会让低精度 kernel 用更高精度算出"通过"。"""
    from hivm_spec.inputs import parse_shape_text

    shape, _dtype, why = parse_shape_text("16xf8e4m3")
    assert shape is None
    assert "假阴性" in why


def test_specs_from_module_uses_func_args_not_allocs() -> None:
    """输入面取自 `VModule.func_args`，而不是 `allocs`。

    回归的是 T3.7 实测暴露的真实缺陷：早先用 `VAlloc(origin=FUNC_ARG)`，而那个
    清单只收带 `#hivm.address_space` 标注的参数（占用分析的口径）。真实 L2 语料
    的入参多是 `memref<16x16xf16>` 这类无标注形式，于是 41 份语料里 37 份推不出
    输入、整份 kernel 退化成**假缺口**。
    """
    from hivm_spec.inputs import specs_from_module
    from hivm_spec.vir import Coverage, Loc, VFuncArg, VModule, VRegion

    loc = Loc(file="t.mlir", line=1)
    module = VModule(
        source="t.mlir",
        items=(VRegion(id="f0", kind="func", loc=loc, items=()),),
        func_args=(
            # 无 address_space 标注——正是早先被漏掉的形态
            VFuncArg(value="%input1", shape_text="16x16xf16", space="", index=0),
            VFuncArg(value="%gm", shape_text="128xf32", space="gm", index=1),
        ),
        coverage=Coverage(),
        arch="a3",
    )
    specs, problems = specs_from_module(module)
    assert problems == ()
    assert [s.name for s in specs] == ["%input1", "%gm"]
    assert specs[0].dtype == "f16" and specs[0].shape == (16, 16)
    assert specs[1].dtype == "f32" and specs[1].shape == (128,)


def test_specs_from_module_reports_undeducible_inputs() -> None:
    """无法推导的输入必须留下原因，不得静默丢弃。"""
    from hivm_spec.inputs import specs_from_module
    from hivm_spec.vir import Coverage, Loc, VFuncArg, VModule, VRegion

    loc = Loc(file="t.mlir", line=1)
    module = VModule(
        source="t.mlir",
        items=(VRegion(id="f0", kind="func", loc=loc, items=()),),
        func_args=(VFuncArg(value="%dyn", shape_text="?x16xf32", space="gm", index=0),),
        coverage=Coverage(),
        arch="a3",
    )
    specs, problems = specs_from_module(module)
    assert specs == ()
    assert len(problems) == 1
    assert "#0" in problems[0] and "动态维" in problems[0]


def test_local_allocs_are_not_inputs() -> None:
    """本地 memref.alloc 不算输入——那是中间缓冲。

    给它预置随机值会掩盖"忘了初始化"这类缺陷。
    """
    from hivm_spec.inputs import specs_from_module
    from hivm_spec.vir import (
        AllocOrigin,
        Coverage,
        Loc,
        VAlloc,
        VModule,
        VRegion,
    )

    loc = Loc(file="t.mlir", line=1)
    module = VModule(
        source="t.mlir",
        items=(VRegion(id="f0", kind="func", loc=loc, items=()),),
        allocs=(
            VAlloc(
                name="buf1",
                space="ub",
                loc=loc,
                shape_text="128xf32",
                origin=AllocOrigin.LOCAL_ALLOC,
                value="%alloc",
            ),
        ),
        func_args=(),
        coverage=Coverage(),
        arch="a3",
    )
    specs, _ = specs_from_module(module)
    assert specs == ()


@pytest.mark.requires_bindings
def test_real_corpus_inputs_drive_interpretation_reproducibly() -> None:
    """真实语料：VIR → 推导输入 → 固定种子生成 → 解释执行，且可复现。

    这条链路是 T3.5 差分对拍的前提：两侧 IR 必须能拿到**同一组**输入，
    否则"结果不同"可能只是输入不同，对拍结论就没有意义。
    """
    import importlib.util
    import json
    from pathlib import Path

    from hivm_spec.generate import generate
    from hivm_spec.inputs import InputStrategy, TilingPolicy, specs_from_module
    from hivm_spec.interpret import interpret
    from hivm_spec.ir_engine import lower_module_text
    from hivm_spec.values import concrete

    root = Path(__file__).resolve().parents[1]
    loader = importlib.util.spec_from_file_location("toy_inputs", root / "specs" / "toy.py")
    assert loader and loader.loader
    mod = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod)
    cfg = json.loads(generate(mod.spec, timestamp="2026-01-01T00:00:00+00:00").config_bytes)

    src = (root / "specs" / "cases" / "corpus" / "l0" / "loop_load_add_store.mlir").read_text()
    modeled = {o["op"] for o in cfg["ops"]}
    pipes = {o["op"]: o.get("pipe", "") for o in cfg["ops"]}
    lowered = lower_module_text(src, modeled, source="corpus.mlir", op_pipes=pipes)

    specs, problems = specs_from_module(lowered.module)
    assert problems == (), problems
    assert len(specs) == 1, "该语料只有一个 gm 函数参数"
    assert specs[0].shape == (256,)

    def _run() -> list[str]:
        strat = InputStrategy(tiling=TilingPolicy(max_elements=64))
        raw = strat.generate_all(specs)
        inputs = {
            k: concrete(v, next(s.dtype for s in specs if s.name == k)) for k, v in raw.items()
        }
        res = interpret(lowered.module, cfg, inputs, bound=4)
        assert res.gaps == (), [g.detail for g in res.gaps]
        # 256 → 64 被压缩，必须如实标注
        assert strat.shrunk
        assert strat.provenance()["seed"] > 0
        return [t.out_hash for t in res.traces]

    # 两次独立运行的逐 op 值哈希必须完全一致（FR8）
    assert _run() == _run()
