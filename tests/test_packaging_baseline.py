"""包元数据与工程基线的守卫测试。

这些测试守护的是**工程约定本身**（D5–D11 的可执行部分）：版本策略、marker 注册、
门禁脚本可用性。它们不需要 bindings，因此在 CI 的 core job 中始终运行。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# tomllib 自 Python 3.11 进入标准库；D7 的地板是 3.10，故需回退到 tomli。
# 这与 numpy 约束是同一类缺陷（依赖-解释器耦合），同样由 CI py3.10 腿暴露。
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 仅在 3.10 上执行
    import tomli as tomllib

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


def test_no_dependency_excludes_the_python_floor(pyproject: dict) -> None:
    """依赖约束必须在 D7 的 3.10 地板上可解。

    回归防护：`numpy~=2.5` 曾通过本地 3.14 环境的检查，却在 CI py3.10 腿上
    不可安装（numpy 2.5 要求 Python >=3.12，3.10 的上限是 2.2.6）。这正是
    审查中指出的"声明与现实脱节"，故以测试固化：下界不得高于 3.10 支持的
    最高版本。此处为静态哨兵（不联网），编码已知的版本-解释器耦合。
    """
    floor_max = {"numpy": (2, 2)}  # numpy 2.3+ 要求 >=3.11；2.5+ 要求 >=3.12
    for dep in pyproject["project"]["dependencies"]:
        name = re.split(r"[<>=~!\[]", dep, maxsplit=1)[0].strip()
        if name not in floor_max:
            continue
        lower = re.search(r"(?:>=|~=)\s*(\d+)\.(\d+)", dep)
        assert lower, f"{name} 缺少可解析的下界：{dep!r}"
        got = (int(lower.group(1)), int(lower.group(2)))
        assert got <= floor_max[name], (
            f"{name} 下界 {got} 超出 Python 3.10 可安装的最高版本 "
            f"{floor_max[name]}（D7 地板）；CI py3.10 腿将无法安装该依赖。"
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


def test_d12_single_ir_contract_is_documented() -> None:
    """D12：工具输入契约的边界必须在文档与 agent 规约中同时可查。

    D12 是**范围边界**决策（拒绝一类能力），其风险特征是"未来的 agent 在合理推演
    中重新提出越界方案"。故此处不检查实现，而是守护边界的可引用性：决策与其被拒
    备选留在 design-framework，可执行约束留在 AGENTS.md。二者缺一，边界就会退化为
    需要反复重新论证的口头约定。
    """
    framework = (REPO_ROOT / "docs" / "design-framework.md").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "### D12" in framework, "design-framework 缺 D12 决策条目"
    assert "被拒绝的备选" in framework, "D12 须留档被拒备选以防重复提案"

    # 断言实质约束而非仅提及编号：AGENTS.md 的不可违约束表须同时载明
    # "不吃 pass 序列" 与 "不解析转储" 两条禁令，且指明 D12 出处。
    constraint_row = next(
        (ln for ln in agents.splitlines() if "pass 序列" in ln and "D12" in ln), None
    )
    assert constraint_row, "AGENTS.md 架构约束表缺 D12 的单 IR 输入条目"
    assert "print-ir" in constraint_row, "D12 约束须明确拒绝解析 --print-ir-* 转储"
