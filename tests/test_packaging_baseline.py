"""包元数据与工程基线的守卫测试。

这些测试守护的是**工程约定本身**（D5–D11 的可执行部分）：版本策略、marker 注册、
门禁脚本可用性。它们不需要 bindings，因此在 CI 的 core job 中始终运行。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_package_imports_and_exposes_version() -> None:
    import hivm_spec

    assert hivm_spec.__version__


def test_requires_python_floor_matches_bindings_abi(pyproject: dict) -> None:
    """D7：3.10 下界由 cp310 绑定 ABI 强制，不得随意放宽。

    放宽到 <3.10 会让 IR 接口层在声明上"可用"而实际无法加载绑定，
    正是架构审查指出的声明-现实脱节。
    """
    assert pyproject["project"]["requires-python"] == ">=3.10"


def test_core_dependencies_are_pinned(pyproject: dict) -> None:
    """FR8 确定性：主依赖必须带版本约束，裸依赖不可复现。"""
    for dep in pyproject["project"]["dependencies"]:
        assert any(op in dep for op in ("~=", "==", ">=", "<")), (
            f"依赖 {dep!r} 无版本约束，违反 FR8 可复现性要求"
        )


def test_markers_registered(pyproject: dict) -> None:
    """CI 引用的 marker 必须已注册，否则 --strict-markers 下门禁静默失效。"""
    declared = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    names = {m.split(":", 1)[0].strip() for m in declared}
    assert {"requires_bindings", "schema_golden", "crosscheck"} <= names


def test_spec_gate_script_runs() -> None:
    """spec-gate 必须可执行——它是 FR6 防自欺的机制载体（D5）。"""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "spec_gate.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "spec-gate" in proc.stdout.lower() or "governance" in proc.stdout.lower()


@pytest.mark.schema_golden
def test_config_document_roundtrip_golden() -> None:
    """OD9/FR8：配置文档 schema 往返 golden。

    T0.3 落地后此测试转为实测（同一描述两次生成逐字节一致 + jsonschema 校验 +
    golden 比对）。当前显式 skip 并带任务号，符合"禁止静默跳过"约定。
    """
    pytest.importorskip("hivm_spec.generator", reason="PENDING(T0.3) 生成器未实现")
