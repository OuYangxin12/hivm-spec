"""性质测试与账本绊线的测试（T0.7 / T0.8）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hivm_spec.properties import (
    KNOWN_FALSE_PROPERTIES,
    REGISTRY,
    CounterExample,
    Property,
    PropertyKind,
    PropertyRegistry,
    PropertyReport,
    commutative,
)
from hivm_spec.tripwire import OpRegistry, load_registry, report_coverage

REPO_ROOT = Path(__file__).resolve().parent.parent
TOY = REPO_ROOT / "specs" / "toy.py"


def _toy_modeled_ops() -> set[str]:
    spec_obj = importlib.util.spec_from_file_location("toy_tw", TOY)
    assert spec_obj and spec_obj.loader
    mod = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)
    return {op.op for op in mod.spec.ops}


# ---------------------------------------------------------------------------
# T0.7 性质测试骨架
# ---------------------------------------------------------------------------


def test_registry_has_example_properties() -> None:
    assert REGISTRY.all(), "示例性质缺失，骨架无法证明可运行"
    assert "hivm.hir.vadd" in REGISTRY.ops()


def test_properties_are_grouped_by_op() -> None:
    props = REGISTRY.for_op("hivm.hir.vadd")
    kinds = {p.kind for p in props}
    assert PropertyKind.COMMUTATIVE in kinds
    assert PropertyKind.IDENTITY in kinds


def test_property_requires_a_readable_description() -> None:
    with pytest.raises(ValueError, match="描述"):
        Property(
            op="hivm.hir.x",
            kind=PropertyKind.CUSTOM,
            predicate=lambda: True,
            description="",
        )


def test_property_requires_an_op() -> None:
    with pytest.raises(ValueError, match="op"):
        Property(op="", kind=PropertyKind.CUSTOM, predicate=lambda: True, description="d")


@pytest.mark.parametrize(("a", "b"), [(0, 0), (1, 2), (-5, 7), (2**31, 1), (-(2**31), 2**31)])
def test_vadd_commutative_property_holds(a: int, b: int) -> None:
    prop = next(p for p in REGISTRY.for_op("hivm.hir.vadd") if p.kind is PropertyKind.COMMUTATIVE)
    assert prop.predicate(a, b)


def test_property_predicate_actually_detects_violations() -> None:
    """骨架必须能发现反例，否则它只是装饰。

    用减法（不可交换）注册一条交换律，验证 predicate 返回 False。
    """
    local = PropertyRegistry()
    bad = Property(
        op="hivm.hir.vsub",
        kind=PropertyKind.COMMUTATIVE,
        predicate=lambda a, b: (a - b) == (b - a),
        description="减法交换律（刻意错误）",
    )
    local.add(bad)
    assert not bad.predicate(3, 5), "减法不可交换，predicate 必须返回 False"
    assert bad.predicate(4, 4), "a == b 时恰好成立——正是随机搜索容易漏过的情形"


def test_known_false_properties_are_documented() -> None:
    """已知不成立的性质须留档，防止未来有人"顺手"加上。"""
    assert "hivm.hir.vsub:commutative" in KNOWN_FALSE_PROPERTIES
    assert "hivm.hir.vadd:associative" in KNOWN_FALSE_PROPERTIES
    for key, reason in KNOWN_FALSE_PROPERTIES.items():
        assert ":" in key and reason


def test_caveats_are_recorded_not_hidden() -> None:
    """浮点例外必须显式记录，而非假装性质普遍成立。"""
    prop = next(p for p in REGISTRY.for_op("hivm.hir.vadd") if p.kind is PropertyKind.COMMUTATIVE)
    assert prop.caveats, "浮点/NaN 例外须显式留档"


def test_unmodeled_property_ops_are_reported_not_dropped() -> None:
    """有性质声明但未建模的 op 必须被报告（FR4 同一原则）。"""
    local = PropertyRegistry()
    commutative_prop = Property(
        op="hivm.hir.never_modeled",
        kind=PropertyKind.COMMUTATIVE,
        predicate=lambda a, b: True,
        description="d",
    )
    local.add(commutative_prop)
    assert local.unmodeled(["hivm.hir.vadd"]) == ("hivm.hir.never_modeled",)


def test_counterexample_report_is_readable() -> None:
    prop = next(iter(REGISTRY.for_op("hivm.hir.vadd")))
    ce = CounterExample(prop=prop, inputs={"a": 1, "b": 2}, note="示例")
    text = ce.render()
    assert "hivm.hir.vadd" in text and "a" in text


def test_property_report_summarizes() -> None:
    rep = PropertyReport(checked=5, unmodeled_ops=("hivm.hir.x",))
    text = rep.render()
    assert "检查 5 条" in text
    assert "未建模" in text
    assert rep.ok


def test_commutative_helper_registers_into_global_registry() -> None:
    before = len(REGISTRY.all())
    commutative("hivm.hir.test_only_op", lambda a, b: a + b)
    assert len(REGISTRY.all()) == before + 1
    assert "hivm.hir.test_only_op" in REGISTRY.ops()


# ---------------------------------------------------------------------------
# T0.8 账本绊线
# ---------------------------------------------------------------------------


def test_registry_snapshot_exists_and_is_anchored() -> None:
    reg = load_registry()
    assert reg.count > 100, f"HIVM op 数应在百量级，实为 {reg.count}"
    assert len(reg.source_commit) == 40, "注册表快照必须锚定主仓 commit"
    assert reg.extraction, "必须记录抓取方式，否则无法复现"


def test_tripwire_reports_coverage_over_all_registered_ops(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M0 验收：对全量已注册 op 输出覆盖报告。

    **刻意不断言覆盖率下限**：M0 只建模 9 个 op 是预期的，断言下限会诱导
    为过门槛而虚报建模。绊线的价值是让缺口可见，不是让它好看。
    """
    report = report_coverage(_toy_modeled_ops())
    with capsys.disabled():
        print("\n" + report.render())

    assert report.total > 100
    assert report.modeled, "至少应有若干 op 被建模"
    assert len(report.modeled) + len(report.unmodeled) == report.total


def test_tripwire_flags_ops_unknown_to_upstream() -> None:
    """描述里有但主仓没有的 op = 拼写错误或主仓已删除，都必须刺眼。"""
    report = report_coverage({"hivm.hir.vadd", "hivm.hir.totally_made_up_op"})
    assert "hivm.hir.totally_made_up_op" in report.unknown_to_upstream
    assert "描述中存在但主仓未注册" in report.render()


def test_toy_declares_no_op_unknown_to_upstream() -> None:
    """toy 描述的每个 op 都必须真实存在于主仓——防止凭想象编造 op 名。

    本项目已有前例：手写 L0 语料时凭想象编造 op 签名，6/6 全错。
    """
    report = report_coverage(_toy_modeled_ops())
    assert report.unknown_to_upstream == (), (
        f"toy 描述含主仓不存在的 op：{report.unknown_to_upstream}"
    )


def test_coverage_groups_are_reported() -> None:
    report = report_coverage(_toy_modeled_ops())
    assert report.by_group, "应按分组报告，便于排定建模优先级"
    for _group, (modeled, total) in report.by_group.items():
        assert 0 <= modeled <= total


def test_missing_registry_snapshot_fails_loudly(tmp_path: Path) -> None:
    """基准缺失必须显式报错，不得静默跳过绊线。"""
    with pytest.raises(FileNotFoundError, match="无基准可比"):
        load_registry(tmp_path / "nope.json")


def test_coverage_ratio_is_computed() -> None:
    reg = OpRegistry(ops=("a", "b", "c", "d"))
    report = report_coverage({"a", "b"}, registry=reg)
    assert report.coverage_ratio == 0.5


def test_empty_registry_does_not_divide_by_zero() -> None:
    report = report_coverage(set(), registry=OpRegistry(ops=()))
    assert report.coverage_ratio == 0.0
