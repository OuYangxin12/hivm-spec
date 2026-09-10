"""T4.3 未初始化读检查测试。

验收核心（M4 卡 §3 T4.3）：注入用例检出。
关键纪律 §4.5：用独立 tainted 标记位而非 poison 魔数——魔数可能是合法计算
结果，NaN 更不安全（真实计算本就会产生 NaN）。
"""

from __future__ import annotations

import pytest

from hivm_spec.assemble import run_uninit_read
from hivm_spec.uninit import analyze_uninit_reads
from hivm_spec.verdict import Verdict
from hivm_spec.vir import AllocOrigin, Coverage, Loc, SizeOrigin, VAlloc, VModule, VNode, VRegion

LOC = Loc(file="t.mlir", line=3)


def _cfg(*, with_check: bool = True, cond_write_out: bool = False) -> dict:
    """两个 op：producer 写 dst，consumer 读 src 写 out。"""
    write_kind = "cond_write" if cond_write_out else "write"
    cfg: dict = {
        "ops": [
            {
                "op": "hivm.hir.load",
                "params": [{"name": "src", "kind": "in"}, {"name": "dst", "kind": "out"}],
                "effects": [{"kind": write_kind, "target": "dst", "space": "ub"}],
            },
            {
                "op": "hivm.hir.vadd",
                "params": [
                    {"name": "a", "kind": "in"},
                    {"name": "b", "kind": "in"},
                    {"name": "out", "kind": "out"},
                ],
                "effects": [{"kind": "write", "target": "out", "space": "ub"}],
            },
        ],
        "checks": [],
    }
    if with_check:
        cfg["checks"].append({"name": "uninit_read", "options": {}})
    return cfg


def _alloc(name: str, value: str, origin: AllocOrigin = AllocOrigin.LOCAL_ALLOC) -> VAlloc:
    return VAlloc(
        name=name,
        space="ub",
        nbytes=1024,
        size_origin=SizeOrigin.STATIC_SHAPE,
        loc=LOC,
        origin=origin,
        value=value,
    )


def _module(nodes: tuple[VNode, ...], allocs: tuple[VAlloc, ...]) -> VModule:
    return VModule(
        source="t.mlir",
        items=(VRegion(id="f0", kind="func", loc=LOC, items=nodes),),
        allocs=allocs,
        coverage=Coverage(),
        arch="a3",
    )


def _node(nid: str, op: str, operands: tuple[str, ...]) -> VNode:
    return VNode(id=nid, op=op, loc=LOC, operands=operands)


# ---------------------------------------------------------------------------
# 检出
# ---------------------------------------------------------------------------


def test_read_before_any_write_is_detected() -> None:
    """本地 alloc 未经写入就被读 → 检出。"""
    mod = _module(
        (_node("n0", "hivm.hir.vadd", ("%a", "%a", "%out")),),
        (_alloc("buf_a", "%a"), _alloc("buf_out", "%out")),
    )
    report = analyze_uninit_reads(mod, _cfg())
    assert len(report.reads) == 1
    assert report.reads[0].buffer == "buf_a", "须报友好缓冲名，不是原始 SSA dump"
    assert report.reads[0].param == "a"


def test_write_then_read_is_clean() -> None:
    """先写后读是正常模式，不得误报。"""
    mod = _module(
        (
            _node("n0", "hivm.hir.load", ("%gm", "%a")),
            _node("n1", "hivm.hir.vadd", ("%a", "%a", "%out")),
        ),
        (_alloc("buf_a", "%a"), _alloc("buf_out", "%out")),
    )
    assert analyze_uninit_reads(mod, _cfg()).clean


def test_func_args_are_assumed_initialized() -> None:
    """函数参数由调用方负责初始化——读它不算缺陷。"""
    mod = _module(
        (_node("n0", "hivm.hir.vadd", ("%arg", "%arg", "%out")),),
        (_alloc("arg0", "%arg", AllocOrigin.FUNC_ARG), _alloc("buf_out", "%out")),
    )
    assert analyze_uninit_reads(mod, _cfg()).clean


def test_write_order_matters_within_one_op() -> None:
    """同一 op 内读发生在写之前：读自己的 out 参数仍算未初始化读。"""
    mod = _module(
        (_node("n0", "hivm.hir.vadd", ("%x", "%x", "%x")),),
        (_alloc("buf_x", "%x"),),
    )
    report = analyze_uninit_reads(mod, _cfg())
    assert len(report.reads) == 1, "in 参数先于 out 生效，不能因为同 op 写了就放行"


def test_buffer_reported_once_per_op() -> None:
    """同一缓冲在同一 op 上只报一次，避免循环里刷屏（与 M2 去重口径一致）。"""
    mod = _module(
        (
            _node("n0", "hivm.hir.vadd", ("%a", "%a", "%o1")),
            _node("n1", "hivm.hir.vadd", ("%a", "%a", "%o2")),
        ),
        (_alloc("buf_a", "%a"), _alloc("o1", "%o1"), _alloc("o2", "%o2")),
    )
    reads = analyze_uninit_reads(mod, _cfg()).reads
    assert len([r for r in reads if r.buffer == "buf_a"]) == 1


# ---------------------------------------------------------------------------
# 保守方向（NFR2：压制假阳性），但必须留痕
# ---------------------------------------------------------------------------


def test_unmodeled_op_is_conservatively_treated_as_a_writer() -> None:
    """未建模 op 的读写未知 → 保守假定它写了，宁可漏报不误报。"""
    mod = _module(
        (
            _node("n0", "hivm.hir.mystery", ("%a",)),
            _node("n1", "hivm.hir.vadd", ("%a", "%a", "%out")),
        ),
        (_alloc("buf_a", "%a"), _alloc("buf_out", "%out")),
    )
    report = analyze_uninit_reads(mod, _cfg())
    assert report.clean
    assert "hivm.hir.mystery" in report.unmodeled_ops


def test_conservative_pass_is_disclosed_not_silent() -> None:
    """保守放行必须留痕——否则"没报问题"会被读成"没有问题"（FR6）。"""
    mod = _module(
        (
            _node("n0", "hivm.hir.mystery", ("%a",)),
            _node("n1", "hivm.hir.vadd", ("%a", "%a", "%out")),
        ),
        (_alloc("buf_a", "%a"), _alloc("buf_out", "%out")),
    )
    report = analyze_uninit_reads(mod, _cfg())
    assert any("可能漏报" in n for n in report.notes)


def test_cond_write_does_not_count_as_initialization() -> None:
    """条件写可能没写 → 不能凭它认定已初始化。

    但也不因此报错（分支确实可能写了），故读方仍应报——这是保守的"报"方向：
    条件写之后再读，值确实可能是未初始化的。
    """
    mod = _module(
        (
            _node("n0", "hivm.hir.load", ("%gm", "%a")),
            _node("n1", "hivm.hir.vadd", ("%a", "%a", "%out")),
        ),
        (_alloc("buf_a", "%a"), _alloc("buf_out", "%out")),
    )
    report = analyze_uninit_reads(mod, _cfg(cond_write_out=True))
    assert len(report.reads) == 1, "条件写不足以证明已初始化"


# ---------------------------------------------------------------------------
# 反自欺
# ---------------------------------------------------------------------------


def test_no_local_buffers_is_a_gap_not_a_pass() -> None:
    """一个本地缓冲都没有 → 检查未实际发生，报缺口而非 OK。"""
    mod = _module(
        (_node("n0", "hivm.hir.vadd", ("%arg", "%arg", "%arg")),),
        (_alloc("arg0", "%arg", AllocOrigin.FUNC_ARG),),
    )
    report = analyze_uninit_reads(mod, _cfg())
    assert report.tracked_buffers == 0
    assert any("未实际发生" in n for n in report.notes)

    result = run_uninit_read(_cfg(), mod, "sha256:x")
    assert result.verdict is Verdict.COVERAGE_GAP
    assert result.exit_code == 4


def test_report_always_states_the_func_arg_assumption() -> None:
    """ "函数参数视为已初始化"是一条假设，必须每次声明。"""
    mod = _module(
        (_node("n0", "hivm.hir.load", ("%gm", "%a")),),
        (_alloc("buf_a", "%a"),),
    )
    report = analyze_uninit_reads(mod, _cfg())
    assert any("函数参数视为已初始化" in n for n in report.notes)


def test_no_poison_magic_number_is_used() -> None:
    """**M4 卡 §4.5**：不得用魔数/NaN 做污染标记。

    魔数可能是合法计算结果，NaN 更不安全——真实计算本就会产生 NaN，届时
    无法区分"读了未初始化内存"与"算出了 NaN"。污染状态必须与数值分离。
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "hivm_spec" / "uninit.py").read_text()

    # 只看可执行代码：docstring 里**解释**为何不用魔数是正当的，不该被误判
    tree = ast.parse(src)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        head = body[0]
        if (
            isinstance(head, ast.Expr)
            and isinstance(head.value, ast.Constant)
            and isinstance(head.value.value, str)
        ):
            doc_lines.update(range(head.lineno, (head.end_lineno or head.lineno) + 1))

    code = "\n".join(
        ln.split("#", 1)[0] for i, ln in enumerate(src.splitlines(), start=1) if i not in doc_lines
    )
    for magic in ("0xDEADBEEF", "0xdeadbeef", "float('nan')", "np.nan", "nan"):
        assert magic not in code, f"污染标记不得使用魔数 {magic}"


# ---------------------------------------------------------------------------
# 工具装配层
# ---------------------------------------------------------------------------


def test_tool_reports_mismatch_and_locates_the_read() -> None:
    mod = _module(
        (_node("n0", "hivm.hir.vadd", ("%a", "%a", "%out")),),
        (_alloc("buf_a", "%a"), _alloc("buf_out", "%out")),
    )
    result = run_uninit_read(_cfg(), mod, "sha256:x")
    assert result.verdict is Verdict.MISMATCH
    assert result.exit_code == 1
    err = next(f for f in result.diagnostics if f.severity == "error")
    assert err.loc is not None, "必须可定位（FR5）"
    assert err.extra["buffer"] == "buf_a"


def test_tool_requires_declared_check() -> None:
    mod = _module((_node("n0", "hivm.hir.vadd", ("%a", "%a", "%o")),), (_alloc("buf_a", "%a"),))
    result = run_uninit_read(_cfg(with_check=False), mod, "sha256:x")
    assert result.verdict is Verdict.UNTRUSTED_DESCRIPTION


def test_details_expose_tracked_count() -> None:
    """`tracked_buffers` 是结论的分母——为 0 时"没发现问题"毫无意义。"""
    mod = _module(
        (_node("n0", "hivm.hir.load", ("%gm", "%a")),),
        (_alloc("buf_a", "%a"),),
    )
    result = run_uninit_read(_cfg(), mod, "sha256:x")
    assert result.details["tracked_buffers"] == 1


@pytest.mark.requires_bindings
def test_real_corpus_injection_is_detected_end_to_end() -> None:
    """真实语料端到端：删掉 load 使 vadd 读到未初始化缓冲。"""
    import importlib.util
    import json as _json
    from pathlib import Path

    from hivm_spec.__main__ import _effects_from_config
    from hivm_spec.generate import generate
    from hivm_spec.ir_engine import lower_module_text

    root = Path(__file__).resolve().parents[1]
    loader = importlib.util.spec_from_file_location("toy_u", root / "specs" / "toy.py")
    assert loader and loader.loader
    mod_spec = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod_spec)
    cfg = _json.loads(generate(mod_spec.spec, timestamp="2026-01-01T00:00:00+00:00").config_bytes)
    modeled = {o["op"] for o in cfg["ops"]}
    effects, pipes = _effects_from_config(cfg)

    src = (root / "specs" / "cases" / "corpus" / "l0" / "loop_load_add_store.mlir").read_text()
    broken = "\n".join(ln for ln in src.splitlines() if "hivm.hir.load" not in ln)
    assert broken != src

    healthy = lower_module_text(
        src, modeled, source="h.mlir", op_effects=effects, op_pipes=pipes
    ).module
    injected = lower_module_text(
        broken, modeled, source="b.mlir", op_effects=effects, op_pipes=pipes
    ).module

    assert run_uninit_read(cfg, healthy, "x").verdict is Verdict.OK, "健康语料不得误报"
    bad = run_uninit_read(cfg, injected, "x")
    assert bad.verdict is Verdict.MISMATCH
    assert any("buf" in f.extra.get("buffer", "") for f in bad.diagnostics if f.severity == "error")
