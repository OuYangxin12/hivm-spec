"""pytest 全局配置：环境能力探测与覆盖缺口的显式登记（D7）。

`requires_bindings` 的跳过不是"测试通过"，而是**已登记的覆盖缺口**。与
`COVERAGE_GAP` verdict 同一哲学：缺口是合法结论，静默不是。因此跳过会在
测试会话结尾打印明确的缺口报告，使 CI 日志里"IR 接口层未被验证"这件事可见。
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

BINDINGS_HINT = (
    "主仓 bishengir bindings 不可用（cp310 ABI）。安装方式见 "
    "docs/milestone-plan.md §1 V1 记录；本地可经编译服务器远端执行。"
)


def _bindings_available() -> bool:
    """探测 bishengir bindings 是否可用。

    **委托给 `hivm_spec.bindings`**，而不是自己再写一份探测逻辑——两处独立的
    探测必然漂移，会造成"conftest 认为不可用而跳过、引擎其实能加载"（或反之）
    的错位。绑定 .so 为 cp310 ABI，且树的位置由 HIVM_SPEC_BINDINGS 指定，
    不一定在 sys.path 上，故不能用 find_spec 判断。
    """
    if sys.version_info[:2] != (3, 10):
        return False
    try:
        from hivm_spec.bindings import bindings_available
    except ImportError:
        return importlib.util.find_spec("bishengir") is not None
    return bindings_available()


BINDINGS = _bindings_available()

_Z3_HINT = (
    "z3 不可用（symbolic extra 未安装）。安装：pip install -e '.[symbolic]'；"
    "决策见 docs/decisions/T4.0-z3-dependency.md。"
)


def _z3_available() -> bool:
    """探测 z3 是否可用。

    与 bindings 同一哲学：缺依赖是**已登记的覆盖缺口**，不是"测试通过"。
    """
    return importlib.util.find_spec("z3") is not None


Z3_AVAILABLE = _z3_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # 两类能力**各自独立**判断：不能因为 bindings 可用就提前 return，否则
    # z3 的 skip 永远挂不上——在装了 bindings 的机器上会静默失效。
    if not BINDINGS:
        skip = pytest.mark.skip(reason=f"requires_bindings: {BINDINGS_HINT}")
        for item in items:
            if "requires_bindings" in item.keywords:
                item.add_marker(skip)

    if not Z3_AVAILABLE:
        z3_skip = pytest.mark.skip(reason=f"requires_z3: {_Z3_HINT}")
        for item in items:
            if "requires_z3" in item.keywords:
                item.add_marker(z3_skip)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """把跳过的测试渲染为显式覆盖缺口，而非沉默的 's'。"""
    z3_skipped = [
        r
        for r in terminalreporter.stats.get("skipped", [])
        if "requires_z3" in str(getattr(r, "longrepr", ""))
    ]
    if z3_skipped:
        terminalreporter.write_sep("=", "COVERAGE GAP (registered)", yellow=True)
        terminalreporter.write_line(
            f"符号后端未被本次运行验证：{len(z3_skipped)} 个用例因缺少 z3 跳过。"
        )
        terminalreporter.write_line(f"原因：{_Z3_HINT}")
        terminalreporter.write_line("该缺口已在 T4.0 决策中登记（docs/decisions/）。")

    if BINDINGS:
        return
    skipped = [
        r
        for r in terminalreporter.stats.get("skipped", [])
        if "requires_bindings" in str(getattr(r, "longrepr", ""))
    ]
    if not skipped:
        return
    terminalreporter.write_sep("=", "COVERAGE GAP (registered)", yellow=True)
    terminalreporter.write_line(
        f"IR 接口层未被本次运行验证：{len(skipped)} 个用例因缺少 bindings 跳过。"
    )
    terminalreporter.write_line(f"原因：{BINDINGS_HINT}")
    terminalreporter.write_line(
        "该缺口已在 docs/design-framework.md §3 D7 与 §6 中登记；"
        "IR 接口层的回归责任由编译服务器上的远端执行承担。"
    )
