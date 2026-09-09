#!/usr/bin/env python3
"""本地合入前置门禁（D5 的本地路径落点，M2 审查发现 2）。

## 为什么需要这个脚本

milestone-plan §8 规定"所有改动走 feature 分支 → PR → 合入"，且"CI 四门禁必须
全绿方可合入"。但 M2 与前一批评审整改都是**本地 merge**（未推送、无 PR），
远端分支保护与 CI 在这条路径上**结构性地无法生效**：feature 分支没上过远端，
CI 就不可能跑；merge 与 feature 提交只隔 21 秒，本地预检也无从留痕。
结果是 D5 声称的"机器门禁不可绕过"退化成了口头纪律。

本脚本把本地合入路径重新钉在机器上：合入前必须跑通它，并把结果写入
`build/merge-gate.json` 作为**可审计证据**（谁、什么 HEAD、跑了哪些门禁、
结论）。它不替代 CI——推送后 CI 仍是权威；它解决的是"本地合入没有任何
机器证据"这个具体缺口。

## 用法

    python scripts/merge_gate.py            # 跑全部门禁并写证据
    python scripts/merge_gate.py --check    # 只校验证据是否覆盖当前 HEAD

退出码：0 全绿；1 有门禁失败；2 用法/环境错误。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "build" / "merge-gate.json"

#: 与 CI 四 job 对应的本地等价门禁（顺序即执行顺序）
GATES: tuple[tuple[str, list[str]], ...] = (
    ("lint", ["ruff", "check", "."]),
    ("format", ["ruff", "format", "--check", "."]),
    ("types", ["mypy", "src"]),
    ("core-tests", ["pytest", "-m", "not requires_bindings", "-q"]),
    ("corpus-manifest", ["pytest", "tests/test_corpus_manifest.py", "-q"]),
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _spec_gate_base() -> str:
    """spec-gate 的比较基线。

    **不能直接用 origin/main**：本地合入路径下 origin/main 可能落后若干个未推送
    的 merge，用它做 base 会把早已合入的改动重复计入（或反之漏检）。取
    `origin/main` 与 HEAD 的 merge-base 才是"本次未上远端的全部改动"的起点。
    origin/main 不存在时（离线/未配远端）退回 HEAD~1。
    """
    for ref in ("origin/main", "main"):
        try:
            return _git("merge-base", ref, "HEAD")
        except subprocess.CalledProcessError:
            continue
    return "HEAD~1"


def _resolve(cmd: list[str]) -> list[str] | None:
    """把工具名解析到当前解释器环境；缺失则返回 None（如实报告，不静默跳过）。"""
    if Path(cmd[0]).is_absolute():
        return cmd  # 已是具体可执行文件（如 sys.executable 起头的命令）
    exe = shutil.which(cmd[0])
    if exe:
        return [exe, *cmd[1:]]
    # venv 内常见：工具作为模块可用
    if cmd[0] in ("pytest", "mypy", "ruff"):
        return [sys.executable, "-m", *cmd]
    return None


def run_gates() -> tuple[list[dict], bool]:
    gates = [
        *GATES,
        # spec-gate（CI 第四 job）：base 取 merge-base，见 _spec_gate_base
        (
            "spec-gate",
            [
                sys.executable,
                str(REPO / "scripts" / "spec_gate.py"),
                "--base",
                _spec_gate_base(),
                "--head",
                "HEAD",
            ],
        ),
    ]
    results: list[dict] = []
    all_ok = True
    for name, cmd in gates:
        resolved = _resolve(cmd)
        if resolved is None:
            results.append({"gate": name, "status": "unavailable", "cmd": " ".join(cmd)})
            all_ok = False
            print(f"  ✗ {name}: 工具不可用（{cmd[0]}）")
            continue
        proc = subprocess.run(resolved, cwd=REPO, capture_output=True, text=True)
        ok = proc.returncode == 0
        all_ok = all_ok and ok
        results.append(
            {
                "gate": name,
                "status": "pass" if ok else "fail",
                "cmd": " ".join(cmd),
                "returncode": proc.returncode,
                "tail": (proc.stdout or proc.stderr).strip().splitlines()[-3:],
            }
        )
        print(f"  {'✓' if ok else '✗'} {name}")
        if not ok:
            tail_lines = (proc.stdout or proc.stderr).splitlines()[-8:]
            print("\n".join(f"      {line}" for line in tail_lines))
    return results, all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description="本地合入前置门禁（D5 本地路径）")
    ap.add_argument(
        "--check",
        action="store_true",
        help="只校验既有证据是否覆盖当前 HEAD（合入脚本/评审用）",
    )
    args = ap.parse_args()

    head = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")

    if args.check:
        if not EVIDENCE.is_file():
            print(f"✗ 缺少门禁证据 {EVIDENCE.relative_to(REPO)}——先运行 merge_gate.py")
            return 1
        ev = json.loads(EVIDENCE.read_text())
        if ev.get("head") != head:
            print(f"✗ 证据陈旧：记录 HEAD={ev.get('head', '')[:12]}，当前 HEAD={head[:12]}")
            return 1
        if not ev.get("all_green"):
            print("✗ 证据显示门禁未全绿")
            return 1
        print(f"✓ 门禁证据覆盖当前 HEAD（{head[:12]}，{ev.get('generated_at')}）")
        return 0

    dirty = _git("status", "--porcelain")
    print(f"本地合入门禁：branch={branch} head={head[:12]}")
    if dirty:
        print("  ⚠ 工作区有未提交改动——门禁结果不代表任何一个提交的状态")

    results, all_ok = run_gates()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "head": head,
                "branch": branch,
                "dirty": bool(dirty),
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "all_green": all_ok and not dirty,
                "gates": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"\n证据已写入 {EVIDENCE.relative_to(REPO)}")
    if dirty:
        print("结论：工作区不干净 → 不得合入（先提交或 stash）")
        return 1
    print("结论：" + ("全绿，可合入" if all_ok else "有门禁失败，不得合入"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
