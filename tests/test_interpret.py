"""T3.2 控制流解释执行测试。

验收核心（milestone-plan §5 T3.2）：循环/分支/嵌套的值语义正确，且**遍历核唯一**
——解释器复用 `timeline.expand_steps`，不自带第二套程序序。
"""

from __future__ import annotations

import numpy as np
import pytest

from hivm_spec.interpret import interpret
from hivm_spec.timeline import DEFAULT_BOUND, expand_steps
from hivm_spec.values import ValueError_, concrete, symbol
from hivm_spec.vir import (
    Coverage,
    Loc,
    VLoop,
    VModule,
    VNode,
    VRegion,
)

LOC = Loc(file="t.mlir", line=1)


def _cfg() -> dict:
    """最小配置文档：两个有值语义的 op + 一个无值语义的 op。"""
    return {
        "ops": [
            {
                "op": "hivm.hir.vadd",
                "pipe": "PIPE_V",
                "params": [
                    {"name": "a", "kind": "in"},
                    {"name": "b", "kind": "in"},
                    {"name": "out", "kind": "out"},
                ],
                "value": {"kind": "declarative", "expr": "elementwise(add, a, b, into=out)"},
            },
            {
                "op": "hivm.hir.load",
                "pipe": "PIPE_MTE2",
                "params": [
                    {"name": "src", "kind": "in"},
                    {"name": "dst", "kind": "out"},
                ],
                "value": {"kind": "declarative", "expr": "copy(src, into=dst)"},
            },
            {
                "op": "hivm.hir.mmadL1",
                "pipe": "PIPE_M",
                "params": [{"name": "a", "kind": "in"}, {"name": "out", "kind": "out"}],
                "value": {
                    "kind": "host_fn",
                    "name": "mmad_l1_value",
                    "reason": "分形布局代数无法声明式表达",
                },
            },
        ],
        "vm": {"pipes": ["PIPE_V", "PIPE_MTE2", "PIPE_M"]},
        "checks": [{"name": "equivalence", "options": {}}],
    }


def _node(nid: str, op: str, operands: tuple[str, ...], results: tuple[str, ...] = ()) -> VNode:
    return VNode(id=nid, op=op, loc=LOC, operands=operands, results=results)


def _module(items: tuple) -> VModule:
    return VModule(
        source="t.mlir",
        items=(VRegion(id="f0", kind="func", loc=LOC, items=items),),
        coverage=Coverage(modeled_ops=("hivm.hir.vadd",)),
        arch="a3",
    )


def _f32(*vals: float) -> object:
    return concrete(np.array(vals, dtype=np.float32), "f32")


# ---------------------------------------------------------------------------
# 遍历核唯一（D6）
# ---------------------------------------------------------------------------


def test_interpreter_reuses_the_single_traversal_core() -> None:
    """解释器的步骤序列必须与 expand_steps 逐一对应（遍历核唯一）。

    若解释器自带遍历，同一份 IR 就会有两种执行顺序：一旦漂移，时间线结论与
    等价结论互相矛盾，而没有任何测试能指出谁错了。
    """
    body = (_node("n1", "hivm.hir.vadd", ("%a", "%a", "%out")),)
    loop = VRegion(id="L", kind="for", loc=LOC, items=body, loop=VLoop(iv="i", trip_count=3))
    m = _module((loop,))

    exp = expand_steps(m, DEFAULT_BOUND)
    res = interpret(m, _cfg(), {"%a": _f32(1.0, 2.0)}, bound=DEFAULT_BOUND)

    assert len(res.traces) == len(exp.steps)
    for trace, step in zip(res.traces, exp.steps, strict=True):
        assert trace.seq == step.seq
        assert trace.op == step.node.op
        assert trace.loop_path == step.loop_path


# ---------------------------------------------------------------------------
# 循环与嵌套
# ---------------------------------------------------------------------------


def test_loop_iterations_are_traced_separately() -> None:
    """每次迭代单独留痕——"第几次迭代开始发散"是定位数值问题的关键信息。"""
    body = (_node("n1", "hivm.hir.vadd", ("%a", "%a", "%out")),)
    loop = VRegion(id="L", kind="for", loc=LOC, items=body, loop=VLoop(iv="i", trip_count=3))
    res = interpret(_module((loop,)), _cfg(), {"%a": _f32(1.0, 2.0)})

    assert [t.label for t in res.traces] == [
        "hivm.hir.vadd@i0",
        "hivm.hir.vadd@i1",
        "hivm.hir.vadd@i2",
    ]
    # 输入不随迭代变化 → 结果恒定 → 哈希相同。这是正确行为，不是哈希退化。
    assert len({t.out_hash for t in res.traces}) == 1
    assert all(t.out_hash.startswith("sha256:") for t in res.traces)


def test_nested_loops_carry_full_iteration_path() -> None:
    """嵌套循环的迭代路径逐层记录（外.内）。"""
    inner_body = (_node("n1", "hivm.hir.vadd", ("%a", "%a", "%out")),)
    inner = VRegion(
        id="Li", kind="for", loc=LOC, items=inner_body, loop=VLoop(iv="j", trip_count=2)
    )
    outer = VRegion(id="Lo", kind="for", loc=LOC, items=(inner,), loop=VLoop(iv="i", trip_count=2))
    res = interpret(_module((outer,)), _cfg(), {"%a": _f32(1.0)})
    assert [t.label for t in res.traces] == [
        "hivm.hir.vadd@i0.0",
        "hivm.hir.vadd@i0.1",
        "hivm.hir.vadd@i1.0",
        "hivm.hir.vadd@i1.1",
    ]


def test_value_written_in_loop_is_visible_after_loop() -> None:
    """循环内写入的缓冲区，循环之后必须可读。

    回归的是一个真实缺陷：早先按"迭代限定名"查找，导致循环后的 store 报
    "操作数未绑定"，把语义正确的程序判成覆盖缺口（假缺口，NFR2）。
    根因是混淆了 **SSA 值** 与 **memref 缓冲区**——HIVM 的 op 通过 outs() 写
    缓冲区，其最新内容跨迭代、跨作用域可见。
    """
    body = (_node("n1", "hivm.hir.vadd", ("%a", "%a", "%buf")),)
    loop = VRegion(id="L", kind="for", loc=LOC, items=body, loop=VLoop(iv="i", trip_count=2))
    after = _node("n2", "hivm.hir.load", ("%buf", "%gm"))
    res = interpret(_module((loop, after)), _cfg(), {"%a": _f32(3.0), "%gm": _f32(0.0)})

    assert res.gaps == (), f"不应有缺口，实际：{[g.detail for g in res.gaps]}"
    assert not res.has_unmodeled
    # 循环后的 load 读到 vadd 的结果（3+3=6）
    assert res.traces[-1].out_hash == res.traces[-2].out_hash


def test_memref_overwrite_is_not_an_ssa_violation() -> None:
    """反复写同一 memref 是正常语义，不得报"IR 非 SSA"。

    回归缺陷：早先对 memref 目标套用了单赋值检查，于是 store 写回入口的 %gm
    直接抛异常——把合法程序判成结构错误。SSA 结果仍受单赋值保护，二者分开。
    """
    nodes = (
        _node("n1", "hivm.hir.vadd", ("%a", "%a", "%gm")),
        _node("n2", "hivm.hir.vadd", ("%a", "%a", "%gm")),
    )
    res = interpret(_module(nodes), _cfg(), {"%a": _f32(1.0), "%gm": _f32(0.0)})
    assert res.gaps == ()
    assert len(res.traces) == 2


# ---------------------------------------------------------------------------
# 未建模 op：降级而非跳过（M3 卡 §4 要点 7）
# ---------------------------------------------------------------------------


def test_escape_hatch_op_degrades_with_reason() -> None:
    """逃生舱 op 无可执行语义 → 登记缺口并说明原因，不静默跳过。"""
    nodes = (_node("n1", "hivm.hir.mmadL1", ("%a", "%out")),)
    res = interpret(_module(nodes), _cfg(), {"%a": _f32(1.0)})

    assert res.has_unmodeled
    assert len(res.gaps) == 1
    detail = res.gaps[0].detail
    assert "host_fn" in detail, detail
    # 原因必须在场：否则"这个 op 为什么没参与对拍"在报告里没有答案（FR5）
    assert "分形布局代数" in detail, detail
    assert res.traces[0].unmodeled
    assert res.traces[0].out_hash == ""


def test_unknown_op_is_reported_once_with_gap() -> None:
    """描述里没有的 op：登记缺口，同名只报一次（避免淹没摘要）。"""
    nodes = (
        _node("n1", "hivm.hir.mystery", ("%a", "%out")),
        _node("n2", "hivm.hir.mystery", ("%a", "%out")),
    )
    res = interpret(_module(nodes), _cfg(), {"%a": _f32(1.0)})
    assert len(res.gaps) == 1
    assert all(t.unmodeled for t in res.traces)


def test_missing_input_does_not_fabricate_a_value() -> None:
    """输入没给全必须降级——不得臆造默认值把"没给输入"伪装成"算出了结果"。"""
    nodes = (_node("n1", "hivm.hir.vadd", ("%missing", "%missing", "%out")),)
    res = interpret(_module(nodes), _cfg(), {})
    assert res.has_unmodeled
    assert any("未绑定" in g.detail for g in res.gaps)


# ---------------------------------------------------------------------------
# 双模：解释器对符号值同样工作
# ---------------------------------------------------------------------------


def test_interpretation_works_in_symbolic_mode() -> None:
    """同一 IR、同一解释器，符号模式给出表达式结构（T3.1 双模的延伸验收）。"""
    body = (_node("n1", "hivm.hir.vadd", ("%a", "%b", "%out")),)
    loop = VRegion(id="L", kind="for", loc=LOC, items=body, loop=VLoop(iv="i", trip_count=2))
    res = interpret(
        _module((loop,)),
        _cfg(),
        {"%a": symbol("a", shape=(2,)), "%b": symbol("b", shape=(2,))},
    )
    assert res.gaps == ()
    out = res.env.get("%out#i1")
    assert out.mode == "symbolic"
    assert out.text() == "add(a, b)"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 截断：继承 M2 口径
# ---------------------------------------------------------------------------


def test_dynamic_trip_truncation_is_inherited_and_declared() -> None:
    """trip 未知 → 按展开界截断，且必须如实标注。

    截断口径与 M2 一致是硬要求：否则同一份 IR 会有两套"验证到哪"的说法。
    """
    body = (_node("n1", "hivm.hir.vadd", ("%a", "%a", "%out")),)
    loop = VRegion(id="L", kind="for", loc=LOC, items=body, loop=VLoop(iv="i", trip_count=None))
    res = interpret(_module((loop,)), _cfg(), {"%a": _f32(1.0)}, bound=3)
    assert res.truncated
    assert res.truncation_notes
    assert len(res.traces) == 3


def test_mixing_modes_across_inputs_is_rejected() -> None:
    """混模输入必须报错——半具体半符号的结论无法解释（FR6）。"""
    nodes = (_node("n1", "hivm.hir.vadd", ("%a", "%b", "%out")),)
    with pytest.raises(ValueError_, match="不允许混算"):
        interpret(_module(nodes), _cfg(), {"%a": _f32(1.0), "%b": symbol("b", shape=(1,))})


# ---------------------------------------------------------------------------
# 真实语料端到端（需 bindings）
# ---------------------------------------------------------------------------


@pytest.mark.requires_bindings
def test_real_l0_corpus_interprets_end_to_end() -> None:
    """真实 L0 语料（load→循环 vadd→store）必须完整求值、零缺口。

    合成 VIR 证明不了解释器能处理真实 IR：真实语料里操作数是 MLIR 的
    `Value(...)` 全文本、op 没有 SSA 结果（memref 语义写 outs）、循环 trip
    由 lower 阶段推导。这些都是上面合成测试覆盖不到的形态。
    """
    import importlib.util
    import json
    from pathlib import Path

    from hivm_spec.generate import generate
    from hivm_spec.ir_engine import lower_module_text

    root = Path(__file__).resolve().parents[1]
    loader = importlib.util.spec_from_file_location("toy_e2e", root / "specs" / "toy.py")
    assert loader and loader.loader
    mod = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod)
    cfg = json.loads(generate(mod.spec, timestamp="2026-01-01T00:00:00+00:00").config_bytes)

    src = (root / "specs" / "cases" / "corpus" / "l0" / "loop_load_add_store.mlir").read_text()
    modeled = {o["op"] for o in cfg["ops"]}
    pipes = {o["op"]: o.get("pipe", "") for o in cfg["ops"]}
    lowered = lower_module_text(src, modeled, source="corpus.mlir", op_pipes=pipes)

    # 入口实参 = load 的第一个操作数（函数 block argument）
    entry = next(n.operands[0] for n in lowered.module.walk_nodes() if "load" in n.op)
    res = interpret(
        lowered.module,
        cfg,
        {entry: concrete(np.arange(4, dtype=np.float32), "f32")},
        bound=4,
    )

    assert res.gaps == (), f"真实语料不应有缺口：{[g.detail for g in res.gaps]}"
    assert not res.has_unmodeled
    assert not res.truncated
    # load → vadd×4（静态 trip 256/64=4）→ store
    assert [t.op for t in res.traces] == [
        "hivm.hir.load",
        "hivm.hir.vadd",
        "hivm.hir.vadd",
        "hivm.hir.vadd",
        "hivm.hir.vadd",
        "hivm.hir.store",
    ]
    # 数值正确性：vadd 是 a+a，输入 arange(4) → [0,2,4,6]
    final = res.env.values[next(k for k in reversed(list(res.env.values)) if "alloc_0" in k)]
    np.testing.assert_allclose(final.array, np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# VTrace：VIR 契约里为逐 op 值哈希预留的唯一落点（M3 卡 §1）
# ---------------------------------------------------------------------------


class _Recorder:
    """最小 VTrace 实现，记录回调序列。"""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_node(self, node, values) -> None:  # type: ignore[no-untyped-def]
        self.events.append(("node", node.op, dict(values)))

    def on_region_enter(self, region) -> None:  # type: ignore[no-untyped-def]
        self.events.append(("enter", region.id))

    def on_region_exit(self, region) -> None:  # type: ignore[no-untyped-def]
        self.events.append(("exit", region.id))


def test_interpret_emits_through_vtrace_contract() -> None:
    """解释器必须通过 VIR 契约的 VTrace 对外发布事件。

    M3 卡 §1 把 VTrace 指定为"逐 op 值哈希与首发散点定位的**唯一落点**"，并
    要求"不新增并行结构"。若解释器只往私有结构里记，外部工具就得另建一套
    遍历——那正是 D6 要消灭的双源真理。
    """
    body = (_node("n1", "hivm.hir.vadd", ("%a", "%a", "%o")),)
    loop = VRegion(id="L", kind="for", loc=LOC, items=body, loop=VLoop(iv="i", trip_count=2))
    rec = _Recorder()
    interpret(_module((loop,)), _cfg(), {"%a": _f32(1.0)}, trace=rec)

    kinds = [e[0] for e in rec.events]
    assert kinds == ["enter", "enter", "node", "node", "exit", "exit"]
    # 区域进出必须正确嵌套：先进 func 再进 loop，退出反序
    assert rec.events[0][1] == "f0"
    assert rec.events[1][1] == "L"
    assert rec.events[-2][1] == "L"
    assert rec.events[-1][1] == "f0"


def test_vtrace_carries_value_slots_from_vir_contract() -> None:
    """回调携带的是 VIR 的 ValueSlot（mode + value_hash），不是私有结构。"""
    from hivm_spec.vir import ValueSlot

    nodes = (_node("n1", "hivm.hir.vadd", ("%a", "%a", "%o")),)
    rec = _Recorder()
    interpret(_module(nodes), _cfg(), {"%a": _f32(2.0)}, trace=rec)

    # events[0] 是 region enter，节点事件要挑出来
    _kind, _op, values = next(e for e in rec.events if e[0] == "node")
    slot = values["%o"]
    assert isinstance(slot, ValueSlot)
    assert slot.mode == "concrete"
    assert slot.value_hash.startswith("sha256:")


def test_vtrace_also_reports_unmodeled_steps() -> None:
    """未建模的步也要发事件——静默跳过会让消费者把"没算"读成"算过且一致"。"""
    nodes = (_node("n1", "hivm.hir.mmadL1", ("%a", "%out")),)
    rec = _Recorder()
    interpret(_module(nodes), _cfg(), {"%a": _f32(1.0)}, trace=rec)

    node_events = [e for e in rec.events if e[0] == "node"]
    assert len(node_events) == 1
    slot = next(iter(node_events[0][2].values()))
    assert slot.mode == "unbound", "没有值时用 unbound 表达，而不是伪造一个哈希"


def test_trace_is_optional_and_does_not_change_results() -> None:
    """挂不挂 trace 不得影响结论——观测不能改变被观测对象。"""
    nodes = (_node("n1", "hivm.hir.vadd", ("%a", "%a", "%o")),)
    m = _module(nodes)
    without = interpret(m, _cfg(), {"%a": _f32(3.0)})
    with_trace = interpret(m, _cfg(), {"%a": _f32(3.0)}, trace=_Recorder())
    assert [t.out_hash for t in without.traces] == [t.out_hash for t in with_trace.traces]
