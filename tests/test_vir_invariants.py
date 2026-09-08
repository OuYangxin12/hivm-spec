"""VIR 四不变量的独立复核（T0.0 / D6，框架 §6.1）。

本文件的测试**刻意尝试破坏契约**：只验证 happy path 的测试无法证明不变量成立。
每个不变量都配一组"违规必须被拒绝"的断言。
"""

from __future__ import annotations

import dataclasses

import pytest

from hivm_spec.vir import (
    Access,
    Coverage,
    Derived,
    Effect,
    Gap,
    GapKind,
    Loc,
    SizeOrigin,
    SyncKind,
    VAlloc,
    VIRError,
    VLoop,
    VModule,
    VNode,
    VRegion,
    VSync,
    check_invariants,
)

LOC = Loc(file="t.mlir", line=1, col=1)


def _node(nid: str, op: str = "hivm.hir.vadd", **kw: object) -> VNode:
    return VNode(id=nid, op=op, loc=LOC, **kw)  # type: ignore[arg-type]


def _module(*nodes: VNode, **kw: object) -> VModule:
    """构造合法模块；默认把出现的 op 全部登记为已建模（满足不变量 4）。"""
    kw.setdefault("coverage", Coverage(modeled_ops=tuple(sorted({n.op for n in nodes}))))
    region = VRegion(id="r0", kind="func", loc=LOC, nodes=nodes)
    return VModule(source="t.mlir", regions=(region,), **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 不变量 1：构造后不可变
# ---------------------------------------------------------------------------


def test_inv1_all_structures_are_frozen() -> None:
    for cls in (VModule, VRegion, VNode, VAlloc, VSync, VLoop, Effect, Coverage, Gap):
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} 必须是 frozen dataclass"


def test_inv1_mutation_is_rejected() -> None:
    n = _node("n0")
    with pytest.raises(dataclasses.FrozenInstanceError):
        n.op = "hivm.hir.vmul"  # type: ignore[misc]

    m = _module(n)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.source = "other.mlir"  # type: ignore[misc]


def test_inv1_containers_are_tuples_not_lists() -> None:
    """list 容器会让"不可变"名存实亡（可 append）。"""
    m = _module(_node("n0"), _node("n1"))
    for r in m.walk_regions():
        assert isinstance(r.nodes, tuple)
        assert isinstance(r.regions, tuple)
    assert isinstance(m.allocs, tuple)
    assert isinstance(m.syncs, tuple)


def test_inv1_derived_results_hang_off_to_the_side() -> None:
    """派生分析必须旁挂，且能绑定 VIR 指纹以防错配。"""
    m = _module(_node("n0"))
    d = Derived(producer="occupancy@0.1", payload={"n0": 128}, vir_fingerprint=m.fingerprint())
    assert d.vir_fingerprint == m.fingerprint()
    with pytest.raises(VIRError):
        Derived(producer="")


# ---------------------------------------------------------------------------
# 不变量 2：节点顺序确定性
# ---------------------------------------------------------------------------


def test_inv2_node_order_is_stable_across_calls() -> None:
    m = _module(_node("a"), _node("b"), _node("c"))
    assert m.node_order() == ("a", "b", "c")
    assert m.node_order() == m.node_order()


def test_inv2_nested_region_order_is_depth_first_nodes_before_subregions() -> None:
    """顺序约定必须稳定：先本区域 nodes，再进子区域。

    M2 的同步顺序判定直接依赖该约定——约定一改，死锁判定结论就会变。
    """
    inner = VRegion(
        id="r1",
        kind="for",
        loc=LOC,
        nodes=(_node("inner0"), _node("inner1")),
        loop=VLoop(iv="i", trip_count=4),
    )
    outer = VRegion(id="r0", kind="func", loc=LOC, nodes=(_node("outer0"),), regions=(inner,))
    m = VModule(
        source="t.mlir",
        regions=(outer,),
        coverage=Coverage(modeled_ops=("hivm.hir.vadd",)),
    )
    assert m.node_order() == ("outer0", "inner0", "inner1")


def test_inv2_fingerprint_is_deterministic_and_order_sensitive() -> None:
    m1 = _module(_node("a"), _node("b"))
    m2 = _module(_node("a"), _node("b"))
    assert m1.fingerprint() == m2.fingerprint(), "同一结构必得同一指纹（FR8）"

    m3 = _module(_node("b"), _node("a"))
    assert m1.fingerprint() != m3.fingerprint(), "顺序不同必须产生不同指纹"


def test_inv2_fingerprint_ignores_source_path_and_engine_version() -> None:
    """路径与引擎版本不属语义指纹（同 §7.3 对 spec_hash 的处理）。"""
    r = VRegion(id="r0", kind="func", loc=LOC, nodes=(_node("a"),))
    cov = Coverage(modeled_ops=("hivm.hir.vadd",))
    a = VModule(source="x.mlir", regions=(r,), coverage=cov, engine_version="0.1")
    b = VModule(source="y.mlir", regions=(r,), coverage=cov, engine_version="0.2")
    assert a.fingerprint() == b.fingerprint()


def test_inv2_duplicate_node_ids_are_rejected() -> None:
    with pytest.raises(VIRError, match="重复"):
        _module(_node("dup"), _node("dup"))


# ---------------------------------------------------------------------------
# 不变量 3：可回溯源位置
# ---------------------------------------------------------------------------


def test_inv3_node_without_loc_is_rejected() -> None:
    with pytest.raises(VIRError, match="不变量 3"):
        VNode(id="n0", op="hivm.hir.vadd", loc=Loc())


def test_inv3_loc_always_describes_something() -> None:
    assert Loc(file="a.mlir", line=7, col=3).describe() == "a.mlir:7:3"
    assert Loc(raw="fused<...>").describe() == "fused<...>"
    assert Loc().describe() == "<unknown-loc>"  # 诊断里不允许出现空串


def test_inv3_checker_flags_missing_loc() -> None:
    """绕过构造校验塞入无 loc 节点时，独立复核仍须发现。"""
    m = _module(_node("n0"))
    bad = object.__new__(VNode)
    object.__setattr__(bad, "id", "n1")
    object.__setattr__(bad, "op", "hivm.hir.vadd")
    object.__setattr__(bad, "loc", Loc())
    object.__setattr__(bad, "operands", ())
    object.__setattr__(bad, "results", ())
    object.__setattr__(bad, "effects", ())
    object.__setattr__(bad, "pipe", "")
    object.__setattr__(bad, "value", VNode(id="x", op="o", loc=LOC).value)
    object.__setattr__(bad, "trust", "")
    object.__setattr__(bad, "attrs", {})
    hacked = VModule(
        source="t.mlir",
        regions=(VRegion(id="r0", kind="func", loc=LOC, nodes=(m.regions[0].nodes[0], bad)),),
        coverage=Coverage(modeled_ops=("hivm.hir.vadd",)),
    )
    problems = check_invariants(hacked)
    assert any("不变量 3" in p for p in problems)


# ---------------------------------------------------------------------------
# 不变量 4：未识别必入 coverage（禁止静默丢弃）
# ---------------------------------------------------------------------------


def test_inv4_unmodeled_op_not_in_coverage_is_a_violation() -> None:
    """核心反自欺场景：出现了没建模的 op，却既不登记已建模也不报缺口。"""
    m = VModule(
        source="t.mlir",
        regions=(
            VRegion(id="r0", kind="func", loc=LOC, nodes=(_node("n0", op="hivm.hir.mystery"),)),
        ),
        coverage=Coverage(modeled_ops=()),  # 空：既未建模也未报缺口
    )
    problems = check_invariants(m)
    assert any("不变量 4" in p for p in problems)
    assert any("静默丢弃" in p for p in problems)


def test_inv4_declared_gap_satisfies_the_invariant() -> None:
    m = VModule(
        source="t.mlir",
        regions=(
            VRegion(id="r0", kind="func", loc=LOC, nodes=(_node("n0", op="hivm.hir.mystery"),)),
        ),
        coverage=Coverage(
            gaps=(
                Gap(
                    kind=GapKind.UNMODELED_OP,
                    op="hivm.hir.mystery",
                    detail="描述库未建模该 op",
                    loc=LOC,
                ),
            )
        ),
    )
    assert check_invariants(m) == []
    assert m.coverage.unmodeled_ops() == ("hivm.hir.mystery",)
    assert not m.coverage.is_complete


def test_inv4_gap_must_be_readable() -> None:
    with pytest.raises(VIRError, match="静默"):
        Gap(kind=GapKind.UNRECOGNIZED_STRUCTURE, detail="")
    with pytest.raises(VIRError, match="op 名"):
        Gap(kind=GapKind.UNMODELED_OP, detail="缺建模", op="")


def test_inv4_complete_coverage_is_the_l1_admission_gate() -> None:
    """L1 语料准入门槛（D13）：无缺口。"""
    assert _module(_node("n0")).coverage.is_complete


# ---------------------------------------------------------------------------
# 结构性校验：不允许"幻影"引用与自相矛盾的元数据
# ---------------------------------------------------------------------------


def test_sync_referencing_nonexistent_node_is_rejected() -> None:
    with pytest.raises(VIRError, match=r"幻影|未出现"):
        VModule(
            source="t.mlir",
            regions=(VRegion(id="r0", kind="func", loc=LOC, nodes=(_node("n0"),)),),
            syncs=(VSync(node_id="ghost", kind=SyncKind.SET_FLAG, loc=LOC),),
            coverage=Coverage(modeled_ops=("hivm.hir.vadd",)),
        )


def test_alloc_static_shape_without_size_is_contradiction() -> None:
    with pytest.raises(VIRError, match="矛盾"):
        VAlloc(name="ub0", space="ub", loc=LOC, nbytes=None, size_origin=SizeOrigin.STATIC_SHAPE)


def test_dynamic_shape_alloc_with_unknown_size_is_legal() -> None:
    """实测 123/189 语料含动态 shape，"尺寸未知"必须是一等公民。"""
    a = VAlloc(
        name="ub0",
        space="ub",
        loc=LOC,
        nbytes=None,
        size_origin=SizeOrigin.UNKNOWN,
        shape_text="?x16xf16",
    )
    assert a.nbytes is None


def test_loop_region_must_carry_loop_info() -> None:
    with pytest.raises(VIRError, match="缺 loop"):
        VRegion(id="r1", kind="for", loc=LOC)
    with pytest.raises(VIRError, match="不应携带 loop"):
        VRegion(id="r1", kind="func", loc=LOC, loop=VLoop(iv="i"))


def test_effect_requires_space() -> None:
    with pytest.raises(VIRError, match="space"):
        Effect(access=Access.READ, space="")


def test_module_requires_source() -> None:
    with pytest.raises(VIRError, match="可追溯"):
        VModule(source="")


# ---------------------------------------------------------------------------
# 顺序而非计数：真实死锁案例的契约支撑
# ---------------------------------------------------------------------------


def test_sync_order_exposes_waits_before_sets_though_counts_balance() -> None:
    """真实案例（CreatePreload stage-major 死锁）的契约级复现。

    set/wait 各 3 个、计数完全配平——FileCheck 类结构验证必然漏过；
    只有序列化视图能暴露"3 个 wait 全部先于首个 set"。
    """
    nodes = tuple(_node(f"w{i}", op="hivm.hir.sync_block_wait") for i in range(3)) + tuple(
        _node(f"s{i}", op="hivm.hir.sync_block_set") for i in range(3)
    )
    syncs = tuple(
        VSync(node_id=f"w{i}", kind=SyncKind.SYNC_BLOCK_WAIT, loc=LOC, event_id=15, core="AIC")
        for i in range(3)
    ) + tuple(
        VSync(node_id=f"s{i}", kind=SyncKind.SYNC_BLOCK_SET, loc=LOC, event_id=14, core="AIC")
        for i in range(3)
    )
    m = VModule(
        source="t.mlir",
        regions=(VRegion(id="r0", kind="func", loc=LOC, nodes=nodes),),
        syncs=syncs,
        coverage=Coverage(modeled_ops=("hivm.hir.sync_block_set", "hivm.hir.sync_block_wait")),
    )
    kinds = [s.kind for s in m.sync_order()]
    n_wait = kinds.count(SyncKind.SYNC_BLOCK_WAIT)
    n_set = kinds.count(SyncKind.SYNC_BLOCK_SET)
    assert n_wait == n_set, "计数配平——正是 FileCheck 会放过的原因"

    first_set = kinds.index(SyncKind.SYNC_BLOCK_SET)
    last_wait = max(i for i, k in enumerate(kinds) if k is SyncKind.SYNC_BLOCK_WAIT)
    assert last_wait < first_set, "契约必须能表达'全部 wait 先于首个 set'这一事实"


def test_check_invariants_passes_on_wellformed_module() -> None:
    m = _module(
        _node("n0", op="hivm.hir.load"),
        _node("n1", op="hivm.hir.vadd", effects=(Effect(access=Access.READ, space="ub"),)),
        _node("n2", op="hivm.hir.store"),
    )
    assert check_invariants(m) == []
