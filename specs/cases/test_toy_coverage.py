"""对拍用例：toy 描述的覆盖面 vs 真实语料（R1 的载体）。

**为什么这是"对拍"而非普通单测**：它拿**描述**（`specs/toy.py`）去比对**真实 IR 语料**
（`specs/cases/corpus/`），任何一侧变动都会在此暴露。R1 规则要求"描述变更必跑对拍集"，
本文件即其执行对象。

本用例不依赖 bishengir bindings（D7：核心层可独立测试），故用文本扫描提取语料中的
op 名。文本扫描的局限已明示在 `_ops_in`：它只做**覆盖面**对拍，语义级对拍待 T1.1
的 IR 接口引擎就位后由 VIR 承担。
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = REPO_ROOT / "specs" / "cases" / "corpus"
TOY = REPO_ROOT / "specs" / "toy.py"

#: 匹配 `hivm.hir.<name>` 形态的 op 名。
#: 局限：会漏掉 generic form（`"hivm.hir.x"(...)`）之外的特殊写法，也不理解
#: 注释——故先剔除注释行。语义级精确提取待 T1.1（VIR）。
OP_RE = re.compile(r"\bhivm\.hir\.[a-zA-Z_0-9]+")


def _load_toy_spec() -> object:
    spec_obj = importlib.util.spec_from_file_location("toy_case", TOY)
    assert spec_obj and spec_obj.loader
    mod = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)
    return mod.spec


def _ops_in(path: Path) -> set[str]:
    """提取一份语料中出现的 hivm op 名（剔除注释与 FileCheck 行）。"""
    ops: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        ops.update(OP_RE.findall(line))
    return ops


def _l0_files() -> list[Path]:
    return sorted((CORPUS / "l0").glob("*.mlir"))


def _l1_files() -> list[Path]:
    return sorted((CORPUS / "l1").glob("*.mlir"))


# ---------------------------------------------------------------------------
# L0：描述必须覆盖手写微例（除刻意留缺口的那个）
# ---------------------------------------------------------------------------


def test_l0_corpus_exists() -> None:
    assert _l0_files(), "L0 语料缺失"


@pytest.mark.parametrize("path", _l0_files(), ids=lambda p: p.name)
def test_l0_ops_are_modeled_or_declared_as_gap(path: Path) -> None:
    """L0 语料的每个 op 要么被 toy 描述建模，要么是刻意的缺口样本。

    `unmodeled_op.mlir` 刻意含未建模 op（验证 COVERAGE_GAP 路径），故豁免；
    其余 L0 语料若出现未建模 op，说明描述与语料脱节，必须有人处理。
    """
    spec = _load_toy_spec()
    modeled = {op.op for op in spec.ops}  # type: ignore[attr-defined]
    found = _ops_in(path)
    unmodeled = found - modeled

    if path.name == "unmodeled_op.mlir":
        assert unmodeled, "该语料的存在意义就是含未建模 op，否则它无法验证缺口路径"
        return

    assert not unmodeled, (
        f"{path.name} 含未被 toy 描述建模的 op：{sorted(unmodeled)}——"
        "要么补描述，要么明确接受 COVERAGE_GAP 并在此登记豁免"
    )


def test_l0_covers_every_structure_the_vir_invariants_need() -> None:
    """L0 的存在理由是覆盖 VIR 四不变量所需结构；缺哪类就说明语料不足。"""
    with (CORPUS / "manifest.json").open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    feats: set[str] = set()
    for e in manifest["entries"]:
        if e["layer"] == "l0":
            feats.update(e["features"])

    required = {
        "static_shape",  # VAlloc 静态尺寸
        "dynamic_shape",  # VAlloc 尺寸未知（一等公民）
        "scf_for",  # VLoop
        "nested_region",  # VRegion 嵌套与遍历顺序
        "order_sensitive",  # sync_order() 的顺序判定
        "coverage_gap",  # 不变量 4
    }
    missing = required - feats
    assert not missing, f"L0 语料未覆盖以下必需结构：{sorted(missing)}"


# ---------------------------------------------------------------------------
# L1：真实语料的覆盖率是 M1 的输入，此处只做诚实计量
# ---------------------------------------------------------------------------


def test_l1_corpus_exists() -> None:
    assert _l1_files(), "L1 语料缺失"


def test_l1_coverage_is_measured_and_reported(capsys: pytest.CaptureFixture[str]) -> None:
    """对 L1 语料计量描述覆盖率。

    **刻意不断言高覆盖率**：toy 描述只有 6 个 op，对真实语料覆盖率必然很低。
    此处的价值是**诚实计量**——把差距变成可见数字，供 M1 排定建模优先级；
    若改为断言"覆盖率 > X%"，就会诱导为过门槛而虚报建模（正是 FR6 要防的）。
    """
    spec = _load_toy_spec()
    modeled = {op.op for op in spec.ops}  # type: ignore[attr-defined]

    all_ops: dict[str, int] = {}
    for f in _l1_files():
        for op in _ops_in(f):
            all_ops[op] = all_ops.get(op, 0) + 1

    covered = {o for o in all_ops if o in modeled}
    uncovered = sorted(set(all_ops) - covered, key=lambda o: (-all_ops[o], o))

    with capsys.disabled():
        print(f"\n[L1 覆盖计量] 语料 {len(_l1_files())} 份，出现 {len(all_ops)} 个唯一 op")
        print(f"  已建模：{len(covered)}/{len(all_ops)}")
        if uncovered:
            top = ", ".join(f"{o}({all_ops[o]})" for o in uncovered[:8])
            print(f"  未建模 top：{top}")

    # 唯一的硬断言：计量本身必须有效（语料非空且能提取到 op）
    assert all_ops, "未能从 L1 语料提取到任何 hivm op —— 提取逻辑或语料有问题"


def test_modeled_ops_actually_appear_in_some_corpus() -> None:
    """描述里建了模但语料中从不出现的 op = 无法被验证的声明。

    这类条目不该被信任升级（无对拍证据），此处提前暴露。
    """
    spec = _load_toy_spec()
    modeled = {op.op for op in spec.ops}  # type: ignore[attr-defined]

    seen: set[str] = set()
    for f in _l0_files() + _l1_files():
        seen |= _ops_in(f)

    never_seen = sorted(modeled - seen)
    assert not never_seen, (
        f"以下 op 已建模但未出现在任何语料中，其描述无法被对拍验证：{never_seen}——"
        "应补语料或移除描述"
    )


# ---------------------------------------------------------------------------
# L2（收割自主仓 pass UT，T1.0b 起）
# ---------------------------------------------------------------------------


def _l2_files() -> list[Path]:
    return sorted((CORPUS / "l2").glob("*.mlir"))


def _all_modeled_ops() -> set[str]:
    """toy + cv 描述的**并集**——真实 kernel 同时用到两个描述文件的 op。"""
    ops: set[str] = set()
    for name, path in (("toy_cov", TOY), ("cv_cov", REPO_ROOT / "specs" / "cv.py")):
        spec_obj = importlib.util.spec_from_file_location(name, path)
        assert spec_obj and spec_obj.loader
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)
        ops |= {o.op for o in mod.spec.ops}
    return ops


def test_l2_corpus_exists() -> None:
    assert _l2_files(), "L2 语料缺失（T1.0 收割后应有 19 份）"


def test_l2_corpus_is_broad_enough() -> None:
    assert len(_l2_files()) >= 19, (
        f"L2 应覆盖 cv-pipelining 全部分节 + preload（≥19），实测 {len(_l2_files())}"
    )


@pytest.mark.parametrize("path", _l2_files(), ids=lambda p: p.name)
def test_l2_ops_are_modeled_in_the_spec_union(path: Path) -> None:
    """L2 是目标语料：出现的每个 hivm op 都必须在描述并集中建模。

    与 L0 的"建模或声明缺口"不同——L2 的意义就在于**目标 kernel 无未建模 op**。
    若主仓 kernel 出现新 op，此处失败即为新增建模需求的信号（绊线的行为面）。
    """
    modeled = _all_modeled_ops()
    present = _ops_in(path)
    unknown = present - modeled
    assert not unknown, f"{path.name} 存在未建模 op：{sorted(unknown)}"


def test_l2_entries_record_stripping() -> None:
    """L2 每条必须留档剥离方式（D13：收割必须可审计）。"""
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    for e in manifest["entries"]:
        if e["layer"] != "l2":
            continue
        assert e.get("extraction"), f"{e['path']} 未记录剥离方式"
        assert "source_path" in e, f"{e['path']} 未记录主仓来源路径"
