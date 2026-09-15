"""M5：矩阵乘输入来源完整性检查（`operand_wiring`）测试。

验收核心（M5 卡 §1/T5.4、T5.7）：
- R1（输入槽位是编译期零）与 R2（输入槽位来自 `tensor.empty`）各自命中；
- **init/累加器槽位的零不得命中**（这是本检查与"见到 fill 就报警"的分界）；
- 无可判定 op / 结构事实缺失时报缺口，不得报 OK；
- 真实案例（bindings 档）：失败版 MISMATCH、修复版 OK。
"""

from __future__ import annotations

import pytest

from hivm_spec.assemble import run_operand_wiring
from hivm_spec.verdict import Verdict
from hivm_spec.vir import (
    Loc,
    VMatmulSite,
    VModule,
    VNode,
    VRegion,
    VStructure,
)
from hivm_spec.wiring import (
    DATALESS_INPUT_RULE,
    MATMUL_INPUT_RULE,
    analyze_operand_wiring,
)

LOC = Loc(file="k.mlir", line=39, col=28)


def _structure(
    *,
    defs: dict[str, str] | None = None,
    zero: set[str] | None = None,
    operands: dict[str, tuple[str, ...]] | None = None,
    sites: tuple[VMatmulSite, ...] = (),
) -> VStructure:
    return VStructure(
        defs=defs or {},
        operands=operands or {},
        zero_values=frozenset(zero or set()),
        matmul_sites=sites,
    )


def _module(structure: VStructure, *, with_matmul_node: bool = True) -> VModule:
    """一个含 matmul 站点（可选地也建 hivm 节点）的最小模块。"""
    if not with_matmul_node:
        # 站点存在但模块里没有任何 hivm 节点：模拟"结构事实齐、节点流为空"
        return VModule(source="k.mlir", structure=structure)
    node = VNode(
        id="n1",
        op="hivm.hir.copy",
        loc=LOC,
        operands=("%src", "%dst"),
        results=("%dst",),
    )
    return VModule(
        source="k.mlir",
        items=(VRegion(id="r1", kind="func", loc=LOC, items=(node,)),),
        structure=structure,
    )


def _cfg(*, with_check: bool = True) -> dict:
    cfg: dict = {"ops": [], "checks": []}
    if with_check:
        cfg["checks"].append({"name": "operand_wiring", "options": {}})
    return cfg


# ---------------------------------------------------------------------------
# R1：输入槽位是编译期零
# ---------------------------------------------------------------------------


def test_r1_zero_in_input_slot_is_reported() -> None:
    """真实 bug 的形态：B 与 outs 同时指向本块新建的零张量。"""
    st = _structure(
        defs={"%28": "linalg.fill", "%30": "bufferization.to_tensor"},
        zero={"%28"},
        sites=(
            VMatmulSite(
                op="linalg.matmul",
                loc=LOC,
                operands=("%30", "%28"),
                dps_input_count=2,
            ),
        ),
    )
    report = analyze_operand_wiring(_module(st))

    assert len(report.findings) == 1
    f = report.findings[0]
    assert f.rule == MATMUL_INPUT_RULE
    assert f.slot == 1, "命中的必须是 B 槽（第 1 个输入），而不是 A"
    assert f.op == "linalg.matmul"
    assert report.matmul_ops == 1
    assert not report.vacuous


def test_r1_zero_in_init_slot_is_not_reported() -> None:
    """零累加器是常态：`ins(X, W) outs(zero)` 不得命中。"""
    st = _structure(
        defs={
            "%x": "bufferization.to_tensor",
            "%w": "bufferization.to_tensor",
            "%zero": "linalg.fill",
        },
        zero={"%zero"},
        sites=(
            VMatmulSite(
                op="linalg.matmul",
                loc=LOC,
                # 前两个是 DPS 输入（有数据），第三个是 init（零，合法）
                operands=("%x", "%w", "%zero"),
                dps_input_count=2,
            ),
        ),
    )
    report = analyze_operand_wiring(_module(st))

    assert report.findings == ()
    assert report.matmul_ops == 1


def test_scalar_zero_is_not_treated_as_dataless() -> None:
    """标量零不入判定（`real_k = 0` 合法）——引擎只把张量零放进 zero_values。"""
    st = _structure(
        defs={"%a": "bufferization.to_tensor", "%c0": "arith.constant"},
        zero=set(),  # 标量零不在集合里
        sites=(
            VMatmulSite(
                op="hivm.hir.mmadL1",
                loc=LOC,
                operands=("%a", "%c0"),
                dps_input_count=2,
            ),
        ),
    )
    report = analyze_operand_wiring(_module(st))

    assert report.findings == ()


# ---------------------------------------------------------------------------
# R2：输入槽位来自 tensor.empty
# ---------------------------------------------------------------------------


def test_r2_empty_tensor_behind_view_chain_is_reported() -> None:
    st = _structure(
        defs={"%e": "tensor.empty", "%c": "tensor.cast", "%x": "bufferization.to_tensor"},
        operands={"%c": ("%e",)},
        sites=(
            VMatmulSite(
                op="linalg.matmul",
                loc=LOC,
                operands=("%x", "%c"),
                dps_input_count=2,
            ),
        ),
    )
    report = analyze_operand_wiring(_module(st))

    assert len(report.findings) == 1
    assert report.findings[0].rule == DATALESS_INPUT_RULE
    assert report.findings[0].slot == 1
    assert "tensor.cast" in report.findings[0].detail


def test_real_producer_stops_the_walk() -> None:
    """链上有真实生产者（copy/load）→ 有数据来源，不报。"""
    st = _structure(
        defs={
            "%d": "hivm.hir.copy",
            "%c": "memref.memory_space_cast",
            "%t": "bufferization.to_tensor",
        },
        operands={"%t": ("%c",), "%c": ("%d",)},
        sites=(VMatmulSite(op="linalg.matmul", loc=LOC, operands=("%x", "%t"), dps_input_count=2),),
    )
    report = analyze_operand_wiring(_module(st))

    assert report.findings == ()


# ---------------------------------------------------------------------------
# 缺口口径与装配
# ---------------------------------------------------------------------------


def test_no_matmul_sites_is_vacuous_gap() -> None:
    report = analyze_operand_wiring(_module(_structure()))
    assert report.vacuous
    assert any("未实际发生" in n for n in report.notes)


def test_empty_structure_is_vacuous_gap() -> None:
    st = _structure(
        sites=(VMatmulSite(op="linalg.matmul", loc=LOC, operands=("%a", "%b"), dps_input_count=2),)
    )
    report = analyze_operand_wiring(_module(st))
    assert report.vacuous, "defs 为空说明引擎未收集，必须报缺口而不是 OK"


def test_tool_verdict_is_mismatch_on_finding() -> None:
    st = _structure(
        defs={"%28": "linalg.fill"},
        zero={"%28"},
        sites=(
            VMatmulSite(op="linalg.matmul", loc=LOC, operands=("%30", "%28"), dps_input_count=2),
        ),
    )
    result = run_operand_wiring(_cfg(), _module(st))

    assert result.verdict is Verdict.MISMATCH
    assert result.tool == "operand_wiring"
    assert result.diagnostics[0].rule == MATMUL_INPUT_RULE
    assert result.diagnostics[0].loc is not None, "FR5：结论必须可定位"


def test_tool_verdict_is_gap_when_vacuous() -> None:
    result = run_operand_wiring(_cfg(), _module(_structure()))
    assert result.verdict is Verdict.COVERAGE_GAP


def test_tool_verdict_is_ok_on_clean_module() -> None:
    st = _structure(
        defs={"%x": "bufferization.to_tensor", "%w": "linalg.fill"},
        sites=(VMatmulSite(op="linalg.matmul", loc=LOC, operands=("%x", "%w"), dps_input_count=2),),
    )
    result = run_operand_wiring(_cfg(), _module(st))
    assert result.verdict is Verdict.OK


def test_missing_check_declaration_is_untrusted() -> None:
    """描述未声明该 check → 结论不可信（不得报 OK）。"""
    st = _structure(
        defs={"%x": "bufferization.to_tensor"},
        sites=(VMatmulSite(op="linalg.matmul", loc=LOC, operands=("%x", "%y"), dps_input_count=2),),
    )
    result = run_operand_wiring(_cfg(with_check=False), _module(st))
    assert result.verdict is Verdict.UNTRUSTED_DESCRIPTION


# ---------------------------------------------------------------------------
# 真实案例（需要 bindings；无 bindings 时为已登记覆盖缺口）
# ---------------------------------------------------------------------------

_BUGGY = "/home/oyx/proj/AscendNPU-IR/tmp-x/dts-dot-add/fix-test/captured_real.mlir"
_FIXED = "/home/oyx/proj/AscendNPU-IR/tmp-x/dts-dot-add/fix-test/fixed_kernel.mlir"


@pytest.mark.requires_bindings
def test_real_case_buggy_reports_and_fixed_is_clean() -> None:
    """M5 退出标准：真实失败版报 MISMATCH 并定位到 B 槽；修复版 OK。"""
    from pathlib import Path

    from hivm_spec.ir_engine import lower_module_text

    if not Path(_BUGGY).is_file() or not Path(_FIXED).is_file():
        pytest.skip("真实案例 IR 不在本机（临时材料，未入库）")

    cfg = _cfg()
    modeled = {op["op"] for op in cfg["ops"]}

    buggy = lower_module_text(Path(_BUGGY).read_text(), modeled, source=_BUGGY)
    res_buggy = run_operand_wiring(cfg, buggy.module)
    assert res_buggy.verdict is Verdict.MISMATCH
    assert res_buggy.diagnostics[0].rule == MATMUL_INPUT_RULE
    assert res_buggy.diagnostics[0].extra["slot"] == 1

    fixed = lower_module_text(Path(_FIXED).read_text(), modeled, source=_FIXED)
    res_fixed = run_operand_wiring(cfg, fixed.module)
    assert res_fixed.verdict is Verdict.OK, "修复版不得误报"


@pytest.mark.requires_bindings
def test_injected_corpus_case_is_detected_and_control_is_clean() -> None:
    """语料级验收（M5/T5.6、AC1 语义维）：注入例检出、健康对照不误报。"""
    from pathlib import Path

    from hivm_spec.ir_engine import lower_module_text

    root = Path(__file__).resolve().parent.parent
    l0 = root / "specs" / "cases" / "corpus" / "l0"
    cfg = _cfg()
    modeled = {op["op"] for op in cfg["ops"]}

    injected = lower_module_text(
        (l0 / "neutralized_matmul_input.mlir").read_text(), modeled, source="injected.mlir"
    )
    res = run_operand_wiring(cfg, injected.module)
    assert res.verdict is Verdict.MISMATCH
    assert res.diagnostics[0].rule == MATMUL_INPUT_RULE
    assert res.diagnostics[0].extra["slot"] == 1, "必须定位到 B 槽（不是 A）"

    control = lower_module_text(
        (l0 / "matmul_input_with_source.mlir").read_text(), modeled, source="control.mlir"
    )
    assert run_operand_wiring(cfg, control.module).verdict is Verdict.OK
