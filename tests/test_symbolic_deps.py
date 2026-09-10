"""T4.0 z3 依赖策略测试。

验收（M4 卡 §3 T4.0）：决策入档；缺 z3 时报 PENDING 而非静默跳过。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

from hivm_spec.symbolic import SymbolicSupportError, require_z3, z3_available

# tomllib 自 Python 3.11 进入标准库；D7 的地板是 3.10，故需回退到 tomli
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 仅 3.10 走这条
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_decision_is_recorded() -> None:
    """决策必须入档——这是 T4.0 的验收项之一，不是可选文档工作。"""
    doc = ROOT / "docs" / "decisions" / "T4.0-z3-dependency.md"
    assert doc.is_file(), "T4.0 决策未入档"
    text = doc.read_text(encoding="utf-8")
    # 必须含实测数据而非估计
    assert "50 MB" in text, "决策须基于实测体积"
    assert "symbolic" in text


def test_z3_stays_an_optional_extra() -> None:
    """z3 **不得**成为硬依赖——四个 CI job 里三个完全不碰符号后端。"""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    hard_deps = " ".join(data["project"]["dependencies"])
    assert "z3" not in hard_deps, "z3 不得进主依赖"

    extras = data["project"]["optional-dependencies"]
    assert any("z3" in dep for dep in extras["symbolic"]), "z3 应在 symbolic extra 中"


def test_version_constraint_allows_current_z3() -> None:
    """**T4.0 修正的真实缺陷**：`~=4.15` 等价于 `>=4.15,==4.*`，会把 z3 锁死在 4.x。

    PyPI 当前最新是 5.x。此前未暴露是因为该 extra 从 M0 预留后一直无人安装——
    这正是"预留的依赖声明必须在首次使用时复核"的例子。
    """
    packaging = pytest.importorskip("packaging.specifiers")

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    z3_dep = next(d for d in data["project"]["optional-dependencies"]["symbolic"] if "z3" in d)
    spec = packaging.SpecifierSet(z3_dep.split("z3-solver", 1)[1])

    assert "5.1.0" in spec, f"约束 {z3_dep!r} 不允许当前 z3 版本 5.1.0"
    assert "4.15" in spec, "不应把下界抬得比既定基线更高"


def test_marker_is_registered() -> None:
    """`requires_z3` 必须注册——strict-markers 下未注册的 marker 会静默失效。"""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = " ".join(data["tool"]["pytest"]["ini_options"]["markers"])
    assert "requires_z3" in markers


def test_probe_does_not_pay_import_cost() -> None:
    """探测用 `find_spec` 而非真导入：导入 z3 要 0.17s，装配时不该付这个代价。"""
    import ast
    import inspect

    from hivm_spec import symbolic

    body = inspect.getsource(symbolic.z3_available)
    assert "find_spec" in body

    # 用 AST 判断有无真正的 import 语句，而不是字符串匹配——docstring 里
    # 提到 "import z3" 是正当的说明文字，不该被误判
    tree = ast.parse(ast.unparse(ast.parse(body).body[0]))
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import | ast.ImportFrom)]
    assert imports == [], "探测函数内不得真的导入 z3（0.17s 开销）"


def test_require_z3_raises_when_unavailable() -> None:
    """不可用时必须抛错，**绝不**返回 None 或假 stub。

    返回 stub 会让调用方在无声的降级路径上继续走，最终产出一个看起来正常、
    实际什么都没验证的结论（FR6）。
    """
    with mock.patch("hivm_spec.symbolic.z3_available", return_value=False):
        with pytest.raises(SymbolicSupportError) as exc:
            require_z3()
    assert "symbolic" in str(exc.value), "错误信息须告诉用户怎么装"


def test_support_error_is_not_an_import_error() -> None:
    """刻意不继承 ImportError：环境问题（报 PENDING）与代码缺陷（应崩溃）必须可区分。"""
    assert not issubclass(SymbolicSupportError, ImportError)
    assert issubclass(SymbolicSupportError, RuntimeError)


def test_availability_matches_reality() -> None:
    assert z3_available() == (importlib.util.find_spec("z3") is not None)


@pytest.mark.requires_z3
def test_z3_actually_solves_when_available() -> None:
    """z3 装上时确实可用——防止 marker 挂了但后端其实是坏的。"""
    z3 = require_z3()
    x, y = z3.BitVecs("x y", 32)
    solver = z3.Solver()
    solver.add(x + y != y + x)  # 位向量加法可交换 → 应当 unsat
    assert str(solver.check()) == "unsat"


def test_conftest_skips_z3_and_bindings_independently() -> None:
    """两类能力必须**各自独立**判断。

    早期写法是 `if BINDINGS: return`，会导致装了 bindings 的机器上 z3 的
    skip 永远挂不上——测试在缺 z3 时直接 ERROR 而不是被登记为缺口。
    """
    src = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    hook = src.split("def pytest_collection_modifyitems")[1].split("\ndef ")[0]
    assert "if not BINDINGS:" in hook, "不得用提前 return 短路掉 z3 分支"
    assert "if not Z3_AVAILABLE:" in hook


def test_skipped_z3_tests_are_reported_as_a_gap() -> None:
    """跳过必须渲染成显式缺口，而不是沉默的 's'（与 bindings 同一哲学）。"""
    src = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    summary = src.split("def pytest_terminal_summary")[1]
    assert "requires_z3" in summary
    assert "COVERAGE GAP" in summary


def test_ci_does_not_silently_install_z3() -> None:
    """CI 不装 z3 是**决策**，不是疏忽——若哪天要装，应是一次独立可评估的变更。"""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "symbolic" not in ci, (
        "CI 现在装了 symbolic extra——这偏离了 T4.0 决策，"
        "如属有意变更请同步更新 docs/decisions/T4.0-z3-dependency.md"
    )
