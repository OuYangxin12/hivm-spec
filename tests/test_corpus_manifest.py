"""语料库 manifest 契约的守卫（D13 / 框架 §14）。

语料是判定依据的一部分：FR8 要求可复现，FR6 要求不可被外部静默改变。
故这些测试守护三件事——
  ① 每份语料都在 manifest 中登记（缺登记即来源不明）；
  ② manifest 记录的 sha256 与实际内容一致（入库快照未被静默改动）；
  ③ 非 L0 语料必须锚定主仓 commit（漂移可检测，接 D9）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "specs" / "cases" / "corpus"
MANIFEST = CORPUS / "manifest.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    with MANIFEST.open(encoding="utf-8") as fh:
        return json.load(fh)


def _digest(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def _corpus_files() -> list[Path]:
    return sorted(CORPUS.rglob("*.mlir"))


def test_manifest_exists_and_has_entries(manifest: dict) -> None:
    assert manifest["schema_version"] == 1
    assert manifest["entries"], "manifest 不得为空"


def test_every_corpus_file_is_registered(manifest: dict) -> None:
    """缺 manifest 记录即视为来源不明（D13 准入规则）。"""
    registered = {e["path"] for e in manifest["entries"]}
    on_disk = {str(p.relative_to(CORPUS)) for p in _corpus_files()}
    unregistered = on_disk - registered
    assert not unregistered, f"以下语料未在 manifest 登记（来源不明）：{sorted(unregistered)}"


def test_no_manifest_entry_points_at_missing_file(manifest: dict) -> None:
    missing = [e["path"] for e in manifest["entries"] if not (CORPUS / e["path"]).is_file()]
    assert not missing, f"manifest 记录了不存在的语料：{missing}"


def test_digests_match_actual_content(manifest: dict) -> None:
    """内容漂移检测：入库快照被改动而未更新 manifest 时必须失败。"""
    drifted = []
    for e in manifest["entries"]:
        p = CORPUS / e["path"]
        if p.is_file() and _digest(p) != e["sha256"]:
            drifted.append(e["path"])
    assert not drifted, (
        f"以下语料内容与 manifest 记录的 sha256 不符：{drifted}——"
        "若为有意更新，请同步 manifest；否则即为静默改动判定依据（违反 FR6）"
    )


def test_upstream_entries_anchor_a_source_commit(manifest: dict) -> None:
    """非手写语料必须锚定主仓 commit，否则漂移无法检测（D9）。"""
    for e in manifest["entries"]:
        if e["origin"] == "handwritten":
            continue
        commit = e.get("source_commit", "")
        assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), (
            f"{e['path']} 的 source_commit 非法（须 40 位 hex）：{commit!r}"
        )
        assert e.get("source_path"), f"{e['path']} 缺 source_path"
        assert e.get("extraction"), f"{e['path']} 缺 extraction（剥离方式）"


def test_l0_is_handwritten_and_small(manifest: dict) -> None:
    """L0 必须完全自控且微小——它要验的是 VIR 不变量，不是真实语料的复杂度。"""
    l0 = [e for e in manifest["entries"] if e["layer"] == "l0"]
    assert 5 <= len(l0) <= 8, f"L0 规模应为 5–8 个（D13），实为 {len(l0)}"
    for e in l0:
        assert e["origin"] == "handwritten"
        n = len((CORPUS / e["path"]).read_text(encoding="utf-8").splitlines())
        assert n < 30, f"L0 语料 {e['path']} 有 {n} 行，超出 <30 行约束"


def test_l1_scale_within_planned_range(manifest: dict) -> None:
    l1 = [e for e in manifest["entries"] if e["layer"] == "l1"]
    assert 8 <= len(l1) <= 12, f"L1 规模应为 8–12 个（D13），实为 {len(l1)}"


def test_every_entry_records_parse_verification(manifest: dict) -> None:
    """语料必须经真实 parser 验证过——手写语料尤其容易凭想象编造 op 签名。

    本项目在建 L0 时曾 6/6 全部写错（凭想象用 generic form 与不存在的属性），
    经 `bishengir-opt` 严格解析才发现。故"已验证"须留档。
    """
    for e in manifest["entries"]:
        assert e.get("parse_verified"), f"{e['path']} 未记录解析验证方式"


def test_no_negative_cases_in_corpus() -> None:
    """负例（expected-error）属主仓 verifier 职责，不入本语料库（D13）。"""
    offenders = [
        str(p.relative_to(CORPUS))
        for p in _corpus_files()
        if "expected-error" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"语料库不应含负例：{offenders}"


def test_features_are_declared(manifest: dict) -> None:
    """features 供按需选取语料（如只要含 sync_block 的），不得为空。"""
    for e in manifest["entries"]:
        assert e.get("features"), f"{e['path']} 未声明 features"
