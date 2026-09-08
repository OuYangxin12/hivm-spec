"""IR 接口引擎测试（T1.1）。

分两类：
- **不需 bindings**：`IREngine` 对手写 VIR 结构的行为、程序序不变量、
  尺寸/空间解析的纯函数部分。核心层可独立测试是 D7 的硬要求。
- **需 bindings**（`requires_bindings`）：真实语料的端到端降级。
  公共 CI 无 cp310 绑定，故跳过并登记缺口，由编译服务器/本地 py3.10 承担。
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from hivm_spec.bindings import BindingsError, bindings_available
from hivm_spec.ir_engine import (
    ELEM_BYTES,
    SYNC_KINDS,
    IREngine,
    _nbytes_of,
    _parse_loc,
    _space_of,
    effects_from_spec,
)
from hivm_spec.vir import Loc, SizeOrigin, SyncKind, VLoop, VModule, VNode, VRegion

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "specs" / "cases" / "corpus"
TOY = REPO_ROOT / "specs" / "toy.py"


def _toy_spec() -> object:
    so = importlib.util.spec_from_file_location("toy_engine", TOY)
    assert so and so.loader
    mod = importlib.util.module_from_spec(so)
    so.loader.exec_module(mod)
    return mod.spec


# ---------------------------------------------------------------------------
# 纯函数：类型/位置解析
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("type_str", "expected"),
    [
        ("memref<4x64xf32, #hivm.address_space<ub>>", "ub"),
        ("memref<128xf16, #hivm.address_space<gm>>", "gm"),
        ("memref<8x8xf32, #hivm.address_space<cbuf>>", "cbuf"),
        ("memref<4xf32>", ""),
    ],
)
def test_space_extraction(type_str: str, expected: str) -> None:
    assert _space_of(type_str) == expected


@pytest.mark.parametrize(
    ("type_str", "nbytes", "origin"),
    [
        ("memref<256xf32, #hivm.address_space<ub>>", 1024, SizeOrigin.STATIC_SHAPE),
        ("memref<4x64xf16, #hivm.address_space<ub>>", 512, SizeOrigin.STATIC_SHAPE),
        ("memref<?xf32, #hivm.address_space<ub>>", None, SizeOrigin.UNKNOWN),
        ("memref<4x?xf32>", None, SizeOrigin.UNKNOWN),
        ("memref<4xsomeweirdtype>", None, SizeOrigin.UNKNOWN),
    ],
)
def test_size_computation(type_str: str, nbytes: int | None, origin: SizeOrigin) -> None:
    got_bytes, got_origin, _ = _nbytes_of(type_str)
    assert got_bytes == nbytes
    assert got_origin is origin


def test_dynamic_size_never_substitutes_a_fake_number() -> None:
    """动态维度必须得 None，绝不能拿 0 或 1 顶替。

    实测 123/189 语料含动态 shape，"尺寸未知"是常态。若用假数字顶替，
    占用结论会呈现虚假精确——这是最危险的一类自欺。
    """
    for t in ("memref<?xf32>", "memref<?x?xf16>", "memref<4x?x8xf32>"):
        nbytes, origin, _ = _nbytes_of(t)
        assert nbytes is None, f"{t} 不得推算出具体字节数"
        assert origin is SizeOrigin.UNKNOWN


def test_elem_bytes_table_is_self_consistent() -> None:
    assert ELEM_BYTES["f32"] == 4
    assert ELEM_BYTES["f16"] == ELEM_BYTES["bf16"] == 2
    assert all(v > 0 for v in ELEM_BYTES.values())


def test_loc_parsing_keeps_raw_for_unparseable_forms() -> None:
    """复杂 location（fused/callsite）必须保真存档而非丢弃（不变量 3）。"""
    loc = _parse_loc('loc(fused["a", "b"])', "f.mlir")
    assert loc.file == "f.mlir"
    assert "fused" in loc.raw

    loc2 = _parse_loc('loc("k.mlir":12:5)', "f.mlir")
    assert (loc2.file, loc2.line, loc2.col) == ("k.mlir", 12, 5)

    # "-" 表示 stdin，回退到调用方给的来源名
    loc3 = _parse_loc('loc("-":7:3)', "corpus.mlir")
    assert loc3.file == "corpus.mlir"
    assert loc3.line == 7


def test_sync_kinds_cover_both_intra_and_inter_core() -> None:
    """核内与跨核同步必须区分——死锁判定依赖这个区别。"""
    assert SYNC_KINDS["hivm.hir.set_flag"] is SyncKind.SET_FLAG
    assert SYNC_KINDS["hivm.hir.sync_block_set"] is SyncKind.SYNC_BLOCK_SET
    assert SYNC_KINDS["hivm.hir.pipe_barrier"] is SyncKind.PIPE_BARRIER
    intra = {SyncKind.SET_FLAG, SyncKind.WAIT_FLAG}
    inter = {SyncKind.SYNC_BLOCK_SET, SyncKind.SYNC_BLOCK_WAIT}
    assert not (intra & inter)


def test_static_trip_count_only_for_literal_bounds() -> None:
    engine = IREngine(set())
    assert (
        engine._static_trip_count(
            "%c0 = arith.constant 0", "%c8 = arith.constant 8", "%c1 = arith.constant 1"
        )
        == 8
    )
    assert engine._static_trip_count("%arg0", "%c8 = arith.constant 8", "%c1") is None
    # step <= 0 不给结论而非给个错的
    assert (
        engine._static_trip_count("arith.constant 0", "arith.constant 8", "arith.constant 0")
        is None
    )


# ---------------------------------------------------------------------------
# 程序序：本次修复的回归防线
# ---------------------------------------------------------------------------


def test_program_order_interleaves_nodes_and_subregions() -> None:
    """节点与子区域必须按**程序序**交错。

    回归防线：初版把两者分成 nodes/regions 两个元组，导致"循环之后的 store"
    被排到"循环体内的 vadd"之前。若 wait 在循环体内、set 在循环之后，
    该错误会让 M2 得出**相反**的死锁结论。
    """
    loc = Loc(file="t.mlir", line=1)

    def node(i: str) -> VNode:
        return VNode(id=i, op="hivm.hir.vadd", loc=loc)

    body = VRegion(
        id="r1", kind="for", loc=loc, items=(node("in_loop"),), loop=VLoop(iv="i", trip_count=2)
    )
    # 程序序：before → 循环体 → after
    func = VRegion(id="r0", kind="func", loc=loc, items=(node("before"), body, node("after")))
    m = VModule(source="t.mlir", items=(func,))

    assert m.node_order() == ("before", "in_loop", "after"), (
        "程序序被破坏——这会直接改变 M2 的顺序判定结论"
    )


def test_derived_views_agree_with_items() -> None:
    loc = Loc(file="t.mlir", line=1)
    n = VNode(id="n1", op="hivm.hir.vadd", loc=loc)
    inner = VRegion(id="r1", kind="if", loc=loc, items=())
    r = VRegion(id="r0", kind="func", loc=loc, items=(n, inner))
    assert r.nodes == (n,)
    assert r.regions == (inner,)


# ---------------------------------------------------------------------------
# 描述 → 引擎配置
# ---------------------------------------------------------------------------


def test_effects_from_spec_skips_sync_effects() -> None:
    """同步效应不产生内存 Effect（它们走 VSync 路径）。"""
    spec = _toy_spec()
    effects, pipes = effects_from_spec(spec)
    # set_flag 只有同步效应，故不应出现在内存效应表里
    assert "hivm.hir.set_flag" not in effects
    assert "hivm.hir.load" in effects
    assert pipes.get("hivm.hir.load")


def test_effects_from_spec_preserves_access_kind() -> None:
    from hivm_spec.vir import Access

    spec = _toy_spec()
    effects, _ = effects_from_spec(spec)
    load_effects = effects["hivm.hir.load"]
    assert any(e.access is Access.WRITE for e in load_effects), "load 写入 UB"


# ---------------------------------------------------------------------------
# bindings 环境
# ---------------------------------------------------------------------------


def test_bindings_error_is_distinct_from_ir_problems() -> None:
    """环境问题必须与 IR 问题分开（FR7）。"""
    from hivm_spec.spec import SpecError
    from hivm_spec.vir import VIRError

    assert not issubclass(BindingsError, VIRError)
    assert not issubclass(BindingsError, SpecError)


@pytest.mark.skipif(bindings_available(), reason="bindings 可用时不测失败路径")
def test_load_bindings_fails_loudly_when_unavailable() -> None:
    """不允许静默返回 None——否则"没有 bindings"会被当成"IR 没问题"。"""
    from hivm_spec.bindings import load_bindings

    with pytest.raises(BindingsError, match="环境问题"):
        load_bindings()


# ---------------------------------------------------------------------------
# 端到端（需 bindings）
# ---------------------------------------------------------------------------

requires_bindings = pytest.mark.requires_bindings


@requires_bindings
@pytest.mark.skipif(not bindings_available(), reason="PENDING(env) bindings 不可用")
def test_all_l0_corpus_lowers_to_vir() -> None:
    from hivm_spec.ir_engine import lower_module_text

    spec = _toy_spec()
    effects, pipes = effects_from_spec(spec)
    modeled = {o.op for o in spec.ops}  # type: ignore[attr-defined]

    for f in sorted((CORPUS / "l0").glob("*.mlir")):
        res = lower_module_text(
            f.read_text(encoding="utf-8"),
            modeled,
            source=f.name,
            op_effects=effects,
            op_pipes=pipes,
        )
        assert res.module.node_order(), f"{f.name} 未产出任何节点"


@requires_bindings
@pytest.mark.skipif(not bindings_available(), reason="PENDING(env) bindings 不可用")
def test_deadlock_corpus_preserves_wait_before_set_order() -> None:
    """真实语料上验证"顺序而非计数"：3 个 wait 必须排在首个 set 之前。"""
    from hivm_spec.ir_engine import lower_module_text

    spec = _toy_spec()
    effects, pipes = effects_from_spec(spec)
    modeled = {o.op for o in spec.ops}  # type: ignore[attr-defined]

    f = CORPUS / "l0" / "waits_before_sets_deadlock.mlir"
    res = lower_module_text(
        f.read_text(encoding="utf-8"), modeled, source=f.name, op_effects=effects, op_pipes=pipes
    )
    kinds = [s.kind for s in res.module.sync_order()]
    assert kinds[:3] == [SyncKind.SYNC_BLOCK_WAIT] * 3
    assert SyncKind.SYNC_BLOCK_SET in kinds[3:]
    # 计数配平——正是 FileCheck 会漏过的情形
    assert kinds.count(SyncKind.SYNC_BLOCK_WAIT) == kinds.count(SyncKind.SYNC_BLOCK_SET)


@requires_bindings
@pytest.mark.skipif(not bindings_available(), reason="PENDING(env) bindings 不可用")
def test_unmodeled_op_becomes_coverage_gap_not_silent_drop() -> None:
    """FR4 / 不变量 4：未建模 op 必须落 coverage。"""
    from hivm_spec.ir_engine import lower_module_text
    from hivm_spec.vir import GapKind

    spec = _toy_spec()
    effects, pipes = effects_from_spec(spec)
    modeled = {o.op for o in spec.ops}  # type: ignore[attr-defined]

    f = CORPUS / "l0" / "unmodeled_op.mlir"
    res = lower_module_text(
        f.read_text(encoding="utf-8"), modeled, source=f.name, op_effects=effects, op_pipes=pipes
    )
    assert not res.module.coverage.is_complete
    gaps = [g for g in res.module.coverage.gaps if g.kind is GapKind.UNMODELED_OP]
    assert gaps and any("vexp" in g.op for g in gaps)


@requires_bindings
@pytest.mark.skipif(not bindings_available(), reason="PENDING(env) bindings 不可用")
def test_dynamic_shape_reports_unknown_size_gap() -> None:
    from hivm_spec.ir_engine import lower_module_text
    from hivm_spec.vir import GapKind

    spec = _toy_spec()
    effects, pipes = effects_from_spec(spec)
    modeled = {o.op for o in spec.ops}  # type: ignore[attr-defined]

    f = CORPUS / "l0" / "dynamic_shape_alloc.mlir"
    res = lower_module_text(
        f.read_text(encoding="utf-8"), modeled, source=f.name, op_effects=effects, op_pipes=pipes
    )
    assert any(g.kind is GapKind.UNKNOWN_SIZE for g in res.module.coverage.gaps)
    assert any(a.nbytes is None for a in res.module.allocs)


@requires_bindings
@pytest.mark.skipif(not bindings_available(), reason="PENDING(env) bindings 不可用")
def test_lowering_is_deterministic() -> None:
    """FR8：同一输入两次降级得同一指纹。"""
    from hivm_spec.ir_engine import lower_module_text

    spec = _toy_spec()
    effects, pipes = effects_from_spec(spec)
    modeled = {o.op for o in spec.ops}  # type: ignore[attr-defined]
    f = CORPUS / "l0" / "loop_load_add_store.mlir"
    text = f.read_text(encoding="utf-8")

    a = lower_module_text(text, modeled, source=f.name, op_effects=effects, op_pipes=pipes)
    b = lower_module_text(text, modeled, source=f.name, op_effects=effects, op_pipes=pipes)
    assert a.module.fingerprint() == b.module.fingerprint()


@requires_bindings
@pytest.mark.skipif(not bindings_available(), reason="PENDING(env) bindings 不可用")
def test_program_order_matches_source_line_order_on_real_ir() -> None:
    """真实 IR 上验证程序序：节点的源码行号必须递增。

    这是对本次程序序修复的端到端确认——手写 VIR 的单测无法覆盖
    "引擎是否按 MLIR 实际顺序构造 items"。
    """
    from hivm_spec.ir_engine import lower_module_text

    spec = _toy_spec()
    effects, pipes = effects_from_spec(spec)
    modeled = {o.op for o in spec.ops}  # type: ignore[attr-defined]
    f = CORPUS / "l0" / "loop_load_add_store.mlir"
    res = lower_module_text(
        f.read_text(encoding="utf-8"), modeled, source=f.name, op_effects=effects, op_pipes=pipes
    )
    lines = [n.loc.line for n in res.module.walk_nodes() if n.loc.line]
    assert lines == sorted(lines), f"节点顺序与源码行号不一致：{lines}"


@requires_bindings
@pytest.mark.skipif(not bindings_available(), reason="PENDING(env) bindings 不可用")
def test_bootstrap_registers_hivm_dialect() -> None:
    """框架 §6 强制条款的回归：主仓站点初始化不会自动注册 hivm。"""
    from hivm_spec.bindings import load_bindings

    handle = load_bindings()
    assert handle.context.is_registered_operation("hivm.hir.vadd")
    assert handle.context.is_registered_operation("hivm.hir.sync_block_wait")


@requires_bindings
@pytest.mark.skipif(not bindings_available(), reason="PENDING(env) bindings 不可用")
def test_parse_does_not_allow_unregistered_dialects() -> None:
    """宁可解析失败，也不把 hivm op 降级为通用 op。

    若允许未注册方言，工具会在"看不懂 IR"的情况下给出貌似正常的结论。
    """
    from hivm_spec.bindings import load_bindings

    handle = load_bindings()
    with pytest.raises(Exception, match=r"(?i)unregistered|not.*registered|dialect"):
        handle.parse_module(
            'module { func.func @f() { "totally.made_up_op"() : () -> (); return } }'
        )
