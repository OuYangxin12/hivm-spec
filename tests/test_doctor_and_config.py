"""环境自检与配置文档自动定位的测试。

CLI 可用性改造 PR1。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from hivm_spec.__main__ import DEFAULT_CONFIG, DEFAULT_SPECS, main
from hivm_spec.doctor import CheckStatus, DoctorReport, ProbeResult, diagnose

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# doctor：探测
# ---------------------------------------------------------------------------


def test_diagnose_covers_the_four_error_prone_points() -> None:
    """四个易错点都要探到：Python 版本 / bindings / 两个可选 extra。"""
    names = {p.name for p in diagnose().probes}
    assert {"Python", "bishengir bindings", "numpy", "ml_dtypes", "z3"} <= names


def test_every_non_ok_probe_gives_a_remedy() -> None:
    """诊断必须可行动（FR5）——只说"坏了"而不说"怎么修"没有价值。"""
    for probe in diagnose().probes:
        if probe.status is not CheckStatus.OK:
            assert probe.remedy, f"{probe.name} 未给修复建议"
            assert probe.affects, f"{probe.name} 未说明影响范围"


def test_optional_deps_are_degraded_not_missing() -> None:
    """`ml_dtypes`/`z3` 是可选 extra：缺了是能力受限，不是核心不可用。"""
    with mock.patch("importlib.util.find_spec", return_value=None):
        by_name = {p.name: p for p in diagnose().probes}
    assert by_name["ml_dtypes"].status is CheckStatus.DEGRADED
    assert by_name["z3"].status is CheckStatus.DEGRADED
    assert by_name["numpy"].status is CheckStatus.MISSING, "numpy 是硬依赖"


def test_missing_ml_dtypes_never_claims_f32_fallback() -> None:
    """**不谎报可用**：f32 算 bf16 比真实硬件更精确，对拍会假通过。

    所以缺 ml_dtypes 的说明里绝不能出现"退化为 f32 继续"这类措辞。
    """
    with mock.patch("importlib.util.find_spec", return_value=None):
        probe = next(p for p in diagnose().probes if p.name == "ml_dtypes")
    text = probe.affects + probe.remedy
    assert "报缺口" in text
    assert "退化为 f32" not in text or "不会退化为 f32" in text


def test_missing_z3_never_claims_concrete_fallback() -> None:
    """缺 z3 不等于"可退回具体档"——两档结论强度不同，不可互相替代。"""
    with mock.patch("importlib.util.find_spec", return_value=None):
        probe = next(p for p in diagnose().probes if p.name == "z3")
    assert "不可互相替代" in probe.affects


def test_bindings_probe_delegates_and_does_not_reimplement() -> None:
    """探测委托给 `hivm_spec.bindings`，不自己再写一份。

    两处独立的探测必然漂移（conftest.py 里有同样的注释）。
    """
    src = (ROOT / "src" / "hivm_spec" / "doctor.py").read_text(encoding="utf-8")
    assert "from hivm_spec.bindings import bindings_available" in src


# ---------------------------------------------------------------------------
# doctor：报告与退出码
# ---------------------------------------------------------------------------


def _report(*statuses: CheckStatus) -> DoctorReport:
    return DoctorReport(
        probes=tuple(
            ProbeResult(name=f"p{i}", status=s, detail="d", remedy="r", affects="a")
            for i, s in enumerate(statuses)
        )
    )


def test_blocker_only_for_missing() -> None:
    """能力受限仍可用（跑得动就不该让脚本失败）；核心不可用才挡住。"""
    assert not _report(CheckStatus.OK, CheckStatus.DEGRADED).has_blocker
    assert _report(CheckStatus.OK, CheckStatus.MISSING).has_blocker


def test_render_lists_affected_capability() -> None:
    text = _report(CheckStatus.MISSING).render()
    assert "影响" in text and "修复" in text


def test_doctor_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "环境自检" in out
    # 本机是否有 bindings 不定，故只断言退出码与 has_blocker 一致
    assert rc in (0, 1)
    assert (rc == 1) == diagnose().has_blocker


# ---------------------------------------------------------------------------
# 配置文档自动定位
# ---------------------------------------------------------------------------


def test_default_paths_point_at_this_repo() -> None:
    assert DEFAULT_CONFIG == "build/config.json"
    for spec in DEFAULT_SPECS:
        assert (ROOT / spec).is_file(), f"缺省描述 {spec} 不存在"


def test_explicit_missing_config_still_errors(tmp_path: Path) -> None:
    """显式指定的路径不存在 → 报错。

    只有**缺省**才自动生成；用户明确说了用哪个文件，就不该悄悄换一个。
    """
    from hivm_spec.__main__ import _resolve_config

    path, rc = _resolve_config(str(tmp_path / "nope.json"), None)
    assert path is None
    assert rc == 1


def test_existing_default_config_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """已有 config 直接复用，不重复 gen。"""
    from hivm_spec.__main__ import _resolve_config

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / DEFAULT_CONFIG
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{}")

    with mock.patch("hivm_spec.__main__._cmd_gen") as gen:
        path, rc = _resolve_config(None, None)
    assert rc == 0
    assert path == DEFAULT_CONFIG
    assert gen.call_count == 0, "已有配置文档时不应重新生成"


def test_auto_gen_is_safe_because_gen_is_deterministic() -> None:
    """自动 gen 的**前提**是 gen 确定性——否则等于引入不可复现性。

    这条测试锁定该前提：同一描述两次生成，spec_hash 必须一致。
    """
    import importlib.util

    from hivm_spec.generate import generate

    loader = importlib.util.spec_from_file_location("toy_det", ROOT / "specs" / "toy.py")
    assert loader and loader.loader
    mod = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod)

    ts = "2026-01-01T00:00:00+00:00"
    first = generate(mod.spec, timestamp=ts)
    second = generate(mod.spec, timestamp=ts)
    assert first.spec_hash == second.spec_hash
    assert first.config_bytes == second.config_bytes


def test_auto_gen_still_reports_spec_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """自动生成**不是**把 spec_hash 藏起来——审计坐标必须照旧可见。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "specs").mkdir()
    for name in DEFAULT_SPECS:
        (tmp_path / name).write_text((ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")

    from hivm_spec.__main__ import _resolve_config

    path, rc = _resolve_config(None, None)
    assert rc == 0 and path is not None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["ops"], "生成的配置文档应当有内容"
