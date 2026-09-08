#!/usr/bin/env python3
"""spec-gate：描述库治理门禁（FR6 防自欺的机制化落点，D5）。

设计前提：本项目的验证结论建立在 `specs/` 描述之上，而描述由 AI agent 撰写并
可能由同一 agent 合入。若无机器门禁，"被验证者修改验证依据"这条路径是敞开的
（架构审查发现的 FR6 唯一机制性缺口）。本脚本把 OD4 的 PR 门槛变成可执行检查：

R1  描述变更 → 必须跑对拍用例集（specs/cases/）。
R2  trust 升级（provisional → cross-validated → anchored）→ 必须在同一变更集内
    附带对拍证据（specs/cases/ 下的文件变更）。OD8 逃生舱条目信任封顶
    provisional，升级一律拒绝。
R3  尚未实现的阶段一律报 PENDING(<任务号>) 并保持非零可见性，禁止静默跳过
    （与 COVERAGE_GAP 同一哲学：缺口是合法结论，静默不是）。

用法：
    python scripts/spec_gate.py --base <sha> --head <sha>
    python scripts/spec_gate.py --base origin/main          # 本地预检
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# trust 等级序，用于判定 diff 中是否发生"升级"
TRUST_ORDER = {"provisional": 0, "cross-validated": 1, "anchored": 2}
TRUST_RE = re.compile(r"""trust\s*=\s*["'](provisional|cross-validated|anchored)["']""")

DESC_GLOB = "specs/"
CASES_GLOB = "specs/cases/"


class GateResult:
    """门禁结论累加器：区分 fail（真失败）与 pending（未实现，可见但不阻塞）。"""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.pendings: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def pending(self, task: str, msg: str) -> None:
        self.pendings.append(f"PENDING({task}) {msg}")

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def report_and_exit(self) -> int:
        for n in self.notes:
            print(f"  · {n}")
        for p in self.pendings:
            print(f"  ⏳ {p}")
        for f in self.failures:
            print(f"  ✗ {f}")
        if self.failures:
            print(f"\nspec-gate FAIL ({len(self.failures)} violation(s))")
            return 1
        if self.pendings:
            print("\nspec-gate PASS (with pending stages — 见上方 PENDING 条目)")
            return 0
        print("\nspec-gate PASS")
        return 0


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout


def changed_files(base: str, head: str) -> list[str]:
    """base..head 的变更文件清单。base 为空/无效时退化为 HEAD 单提交。"""
    if base and base != "0" * 40:
        out = git("diff", "--name-only", f"{base}...{head}")
        if out.strip():
            return [ln for ln in out.splitlines() if ln.strip()]
    out = git("show", "--name-only", "--pretty=format:", head or "HEAD")
    return [ln for ln in out.splitlines() if ln.strip()]


def added_lines(base: str, head: str, path_filter: str) -> list[str]:
    diff = git("diff", "-U0", f"{base}...{head}", "--", path_filter)
    return [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def removed_lines(base: str, head: str, path_filter: str) -> list[str]:
    diff = git("diff", "-U0", f"{base}...{head}", "--", path_filter)
    return [ln[1:] for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]


def max_trust(lines: list[str]) -> int:
    levels = [TRUST_ORDER[m.group(1)] for ln in lines for m in [TRUST_RE.search(ln)] if m]
    return max(levels) if levels else -1


def run_pytest(marker_or_path: list[str], label: str, res: GateResult) -> None:
    # 先区分"工具不可用"与"用例失败"——二者混淆会把环境问题误报为验证失败，
    # 正是 FR7 元可信性要求区分的两类错误。
    if importlib.util.find_spec("pytest") is None:
        res.fail(
            f'{label}：pytest 不可用（环境问题，非用例失败）。请先 `pip install -e ".[test]"`。'
        )
        return

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *marker_or_path, "-q"],
        cwd=REPO_ROOT,
        check=False,
    )
    # pytest exit 5 = 未收集到用例
    if proc.returncode == 5:
        res.fail(
            f"{label}：描述发生变更但未收集到任何对拍用例。"
            f"R1 要求描述变更必须有对拍证据（specs/cases/）。"
        )
    elif proc.returncode != 0:
        res.fail(f"{label}：对拍用例失败（exit {proc.returncode}）")
    else:
        res.note(f"{label}：通过")


def _exports_spec(path: Path) -> bool:
    """粗判某 .py 是否是描述文件（导出名为 spec 的对象）。

    用文本匹配而非导入：spec-gate 必须在不执行任意代码的前提下做出判断。
    """
    if path.suffix != ".py" or not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return "spec = Spec(" in text or "spec: Spec" in text


def main() -> int:
    ap = argparse.ArgumentParser(description="spec-gate: description governance checks")
    ap.add_argument("--base", default="origin/main", help="base ref/sha")
    ap.add_argument("--head", default="HEAD", help="head ref/sha")
    args = ap.parse_args()

    res = GateResult()
    files = changed_files(args.base, args.head)

    desc_changed = [f for f in files if f.startswith(DESC_GLOB) and not f.startswith(CASES_GLOB)]
    cases_changed = [f for f in files if f.startswith(CASES_GLOB)]

    print(f"spec-gate: {args.base}...{args.head}")
    print(f"  changed={len(files)} descriptions={len(desc_changed)} cases={len(cases_changed)}")

    if not desc_changed:
        res.note("无描述变更，R1/R2 不适用")
        return res.report_and_exit()

    for f in desc_changed:
        res.note(f"描述变更：{f}")

    # --- R2: trust 升级必附证据 ---
    up = max_trust(added_lines(args.base, args.head, DESC_GLOB))
    down = max_trust(removed_lines(args.base, args.head, DESC_GLOB))
    if up > down and up > TRUST_ORDER["provisional"]:
        level = next(k for k, v in TRUST_ORDER.items() if v == up)
        if not cases_changed:
            res.fail(
                f"R2 违规：描述 trust 升级至 '{level}'，但本变更集未附带任何 "
                f"specs/cases/ 对拍证据。OD4 要求升级必须附对拍证据；"
                f"OD8 逃生舱条目信任封顶 provisional 不可升级。"
            )
        else:
            res.note(f"R2：trust 升级至 '{level}'，已附 {len(cases_changed)} 个对拍证据文件")

    # --- R1: 描述变更必跑对拍集 ---
    cases_dir = REPO_ROOT / "specs" / "cases"
    has_cases = any(cases_dir.glob("**/test_*.py")) or any(cases_dir.glob("**/*_test.py"))
    if has_cases:
        run_pytest([str(cases_dir)], "R1 对拍集", res)
    else:
        # T0.6 已落地，R1 已转硬失败：描述变更而无对拍用例即阻塞合入。
        res.fail("描述发生变更但 specs/cases/ 无对拍用例——R1 要求描述变更必须由对拍集验证")

    # --- 描述静态检查（生成前置门，T0.2 已落地）---
    if importlib.util.find_spec("hivm_spec") is None:
        res.pending("env", "hivm_spec 未安装（环境问题，非描述错误）")
    else:
        # 只检查导出了 spec 对象的描述文件；spike 等辅助模块不是描述。
        checkable = [f for f in desc_changed if _exports_spec(REPO_ROOT / f)]
        if not checkable:
            res.note("本次无导出 spec 对象的描述文件变更，静态检查跳过")
        else:
            proc = subprocess.run(
                [sys.executable, "-m", "hivm_spec", "check", *checkable],
                cwd=REPO_ROOT,
                check=False,
            )
            if proc.returncode != 0:
                res.fail(f"描述静态检查失败（exit {proc.returncode}）")
            else:
                res.note(f"描述静态检查：{len(checkable)} 份描述通过")

    return res.report_and_exit()


if __name__ == "__main__":
    sys.exit(main())
