"""核心命题的机械守卫：扩语义靠写描述，不靠改引擎。

本项目不是"HIVM 语义验证工具"，而是**验证器生成器**（框架 §1、AGENTS.md §3.5）。
这条命题若失守，项目就退化成"逐个建造验证工具"——而那是开篇就否定的路线。

命题不能只写在文档里：文档不会在有人违反时报错。这里把它固化成可执行断言。
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hivm_spec"

#: 允许出现具体 op 名的模块，及其正当理由。
#: 新增豁免必须在此写明理由——这是有意的摩擦，防止"临时加一个"变成常态。
_ALLOWED_OP_NAME_MODULES = {
    # bootstrap 自检：需要一个已知 op 验证方言注册成功
    "bindings.py": "方言注册自检需要一个具体 op 作探针",
    # 性质库：本就是"针对某 op 断言某代数性质"，op 名是其数据而非硬编码逻辑
    "properties.py": "性质断言以 op 为主语，op 名是数据",
    # 绊线：按 op 前缀分类，用的是前缀不是具体 op
    "tripwire.py": "按方言前缀（hivm.hir.）分类，非具体 op",
}

#: 语义引擎——这些文件里出现具体 op 名即为命题失守
_SEMANTIC_ENGINES = (
    "occupancy.py",
    "timeline.py",
    "interpret.py",
    "equivalence.py",
    "assemble.py",
    "values.py",
    "numeric.py",
    "inputs.py",
)

_OP_NAME = re.compile(r"hivm\.[a-z_]+\.[a-z_0-9]+")


def _code_without_comments_or_docstrings(path: Path) -> str:
    """取源码里的**可执行部分**：剔除注释与文档字符串。

    注释里提到 op 名是正常的（举例、说明来由）；硬编码进逻辑才是问题。
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                first = body[0]
                for line in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                    docstrings.add(line)

    out = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if idx in docstrings:
            continue
        out.append(line.split("#", 1)[0])
    return "\n".join(out)


@pytest.mark.parametrize("module", _SEMANTIC_ENGINES)
def test_semantic_engines_hardcode_no_op_names(module: str) -> None:
    """语义引擎不得硬编码 HIVM op 名（AGENTS.md §4 不可违约束）。

    引擎认识的应是**语义原语**（copy/elementwise…），不是 op 名。一旦引擎里
    出现 `if op == "hivm.hir.vadd"`，新增 op 就得改引擎——验证器生成器的命题
    当场作废，退化成逐个建造工具。
    """
    path = SRC / module
    if not path.is_file():  # pragma: no cover - 模块重命名时给出明确信号
        pytest.fail(f"引擎模块 {module} 不存在；若已重命名请同步本清单")

    found = sorted(set(_OP_NAME.findall(_code_without_comments_or_docstrings(path))))
    assert not found, (
        f"{module} 的可执行代码里硬编码了 op 名 {found}。\n"
        f"引擎应只认语义原语，不认具体 op——见 AGENTS.md §3.5。\n"
        f"若确有正当理由，需在 _ALLOWED_OP_NAME_MODULES 写明并说服 reviewer。"
    )


def test_op_name_exemptions_are_documented() -> None:
    """豁免清单里的每一项都必须有理由——防止豁免悄悄扩张。"""
    for module, reason in _ALLOWED_OP_NAME_MODULES.items():
        assert (SRC / module).is_file(), f"豁免项 {module} 已不存在，应从清单移除"
        assert len(reason) > 8, f"{module} 的豁免理由过于敷衍：{reason!r}"


def test_value_primitives_is_a_closed_set() -> None:
    """值语义原语是封闭集合：描述里写未知原语必须报错，不得静默放行。

    允许描述凭空声明语义，等于允许臆造语义（FR6）。
    """
    from hivm_spec.values import VALUE_PRIMITIVES, ValueError_, parse_value_kernel

    assert isinstance(VALUE_PRIMITIVES, frozenset), "必须是不可变集合"

    with pytest.raises(ValueError_, match="未知值语义原语"):
        parse_value_kernel("teleport(src, into=dst)")


def test_new_op_needs_no_engine_change() -> None:
    """**核心命题的守卫**：新增一个 op 只需描述，引擎零改动。

    这里不改真实描述文件，而是直接构造配置文档片段喂给内核解析——等价于
    "描述里新增了一个 op"，验证引擎能否不经修改就执行它。

    实测参照（M3 末）：往 specs/toy.py 加 hivm.hir.vdiv 后 `git status src/`
    全空，三个工具同时支持该 op，等价验证可报出首发散点。
    """
    from hivm_spec.interpret import _kernels_of

    # 一个此前完全不存在的 op，语义用现有原语表达
    invented = {
        "ops": [
            {
                "op": "hivm.hir.totally_new_op",
                "pipe": "PIPE_V",
                "params": [
                    {"name": "a", "kind": "in"},
                    {"name": "b", "kind": "in"},
                    {"name": "out", "kind": "out"},
                ],
                "value": {"kind": "declarative", "expr": "elementwise(div, a, b, into=out)"},
            }
        ]
    }
    kernels, reasons = _kernels_of(invented)

    assert "hivm.hir.totally_new_op" in kernels, (
        f"新增 op 未能直接执行，命题失守。原因：{reasons}\n"
        "若这是因为需要新原语，那属于合理的引擎扩展；"
        "若是因为引擎硬编码了 op 名，必须修复。"
    )
    assert reasons == {}


def test_engine_extension_axis_is_primitives_not_ops() -> None:
    """引擎的可扩展轴是原语数量，不是 op 数量。

    锁定这条是为了让"能力"有个正确的度量口径：谈覆盖面应谈原语能表达哪些
    语义类别，而不是"建模了几个 op / 方言共几个 op"——后者会把人引向
    "引擎应内置所有 op 语义"，正是本项目否定的路线。
    """
    # 一个原语服务于多个 op：elementwise 覆盖 add/sub/mul/div/neg/exp…
    from hivm_spec.values import _ELEMENTWISE_FNS, _UNARY_FNS, VALUE_PRIMITIVES

    covered_fns = set(_ELEMENTWISE_FNS) | set(_UNARY_FNS)
    assert len(covered_fns) > len(VALUE_PRIMITIVES), (
        "原语应当是跨 op 复用的语义类别：一个 elementwise 覆盖多种运算。"
        "若运算数不多于原语数，说明有人在为单个 op 造原语（AGENTS.md §3.5 反例）。"
    )


def test_agents_md_states_the_core_proposition() -> None:
    """AGENTS.md 必须写明核心命题——它是 agent 每次干活直接读的强制约束。

    命题只写在 design-framework.md 里不够：实际发生过的偏差是，agent 用
    "建模 18 个 op / 方言 100+ 个 = 15% 覆盖率"来汇报能力，隐含了"引擎应内置
    所有 op 语义"的错误框架。
    """
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "验证器生成器" in text
    assert "正确性类别有限" in text
    assert "不靠改引擎" in text or "不是改引擎" in text
    # 必须给出判断路径，而不只是口号
    assert "VALUE_PRIMITIVES" in text


def test_shipped_description_stays_within_primitives() -> None:
    """随仓描述里声明的值语义必须都能被引擎解析或有明确的降级理由。

    这条守的是"描述侧写了名字、实现侧没有"的错配——那会让一个 op 看起来
    已建模，实际执行时退化成覆盖缺口。
    """
    cfg_path = ROOT / "tests" / "golden" / "toy_config.json"
    if not cfg_path.is_file():
        pytest.skip("golden 配置不存在")

    from hivm_spec.interpret import _kernels_of

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    _kernels, reasons = _kernels_of(cfg)

    # 允许两类"不可执行"：本就无值语义的同步 op、显式逃生舱
    for op, why in reasons.items():
        assert "未声明值语义" in why or "host_fn" in why or "未知值语义原语" in why, (
            f"{op} 的值语义无法解析，且不属于已知的合法降级：{why}"
        )
