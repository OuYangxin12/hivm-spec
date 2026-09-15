#!/usr/bin/env python3
"""定位 bishengir bindings 并校验——不猜、不假装可用。

## 为什么需要它

hivm-spec 的六类检查全部依赖 bishengir Python bindings，而它**不是一份可以
pip 安装的依赖**：

- 它是主仓 **构建树产物**（`build/tools/bishengir/bishengir/python_packages/`）；
- `.so` 为 **cp310 ABI**，且 RUNPATH 里带构建机绝对路径；
- 体积约 246M，上游未发布 wheel。

所以"装这个 skill"的真实含义是：**装上纯 Python 的核心层，然后在这台机器上找到
一份已经构建好的 bindings**。本脚本做后一半，并且只报实测结果。

## 纪律

- **探测 = 真 import**，不看目录名像不像。目录存在但 import 不起来就是不可用。
- **不做全盘猜路径**：猜出来的候选会带来"看着像但其实不是"的树，宁可报未找到。
- 当前解释器不是 3.10 时，说清是**解释器**问题，而不是"bindings 不可用"。
- 没找到时给**可执行的下一步**（FR5）。

用法:
  bootstrap.py                    探测并打印
  bootstrap.py --shell            成功时只输出可 eval 的 shell 片段
  bootstrap.py --search-from DIR  指定向上搜索起点（默认当前目录）

退出码: 0=找到且 import 校验通过; 1=有候选但校验失败(或 ABI 不符); 2=一个候选都没找到
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Candidate",
    "find_candidates",
    "main",
    "verify_import",
]

#: bindings 树的判别特征。
#:
#: **不能要求 `bishengir/__init__.py`**——实测主仓 bindings 的 `bishengir` 是
#: **命名空间包**（`bishengir.__file__ is None`，目录下没有 `__init__.py`，
#: `ir.py`/`passmanager.py` 直接躺在 `bishengir/` 下）。按常规包布局写判别
#: 文件，会把唯一正确的树判成"缺判别文件"。
#:
#: 因此这里只做**廉价预筛**（是目录、底下有 bishengir/），真正的判据是
#: `verify_import` 的真 import——与 `hivm_spec.bindings.load_bindings` 同一条路。
MARKER_SUBDIR = "bishengir"

#: 主仓构建产物相对主仓根的固定路径
REL_IN_MAIN_REPO = Path("build/tools/bishengir/bishengir/python_packages/bishengir")

#: bindings 要求的解释器 ABI（与 hivm_spec.bindings.REQUIRED_PYTHON 同源）
REQUIRED_PYTHON = (3, 10)


def _is_dir(p: Path) -> bool:
    """`is_dir` 但不许权限问题掀桌——扫描器没资格因为读不到某目录就崩。"""
    try:
        return p.is_dir()
    except OSError:
        return False


def _children(d: Path) -> list[Path]:
    try:
        return [c for c in d.iterdir() if _is_dir(c)]
    except OSError:
        return []


@dataclass(frozen=True, slots=True)
class Candidate:
    path: Path
    origin: str

    def has_markers(self) -> bool:
        """廉价预筛：是目录，且底下有 `bishengir/`（命名空间包布局）。"""
        return _is_dir(self.path) and _is_dir(self.path / MARKER_SUBDIR)

    def describe(self) -> str:
        try:
            total = sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file())
            size = f"{total / 2**20:.0f}M"
        except OSError:
            size = "?"
        return f"{self.path}（来源 {self.origin}，约 {size}）"


def _from_env() -> list[Candidate]:
    raw = os.environ.get("HIVM_SPEC_BINDINGS", "").strip()
    if not raw:
        return []
    return [Candidate(Path(raw).expanduser(), "env HIVM_SPEC_BINDINGS")]


def _from_ancestor_builds(start: Path) -> list[Candidate]:
    """向上找主仓构建树——最常见的真实情形：你就站在主仓（或其子目录）里干活。

    只查每层祖先**自身**与它的**直接子目录**（`~/proj` 下的 `AscendNPU-IR`）。
    不做全盘递归：扫到 `/` 既慢又会在无权限目录上抛错，而"向上十层都没有"就说明
    这台机器上没有现成构建树，该走别的路。
    """
    out: list[Candidate] = []
    cur = start
    for _ in range(10):
        for probe in (cur, *_children(cur)):
            cand = probe / REL_IN_MAIN_REPO
            if _is_dir(cand):
                out.append(Candidate(cand, f"主仓构建树 {probe}"))
        if cur.parent == cur:
            break
        cur = cur.parent
    return out


def _from_hivm_spec_repos(start: Path) -> list[Candidate]:
    """hivm-spec 仓自己的 `.bindings`（`scripts/setup_bindings.sh` 的落位点）。

    这是拉取过绑定的机器上的**默认位置**，所以排在构建树探测之前。
    """
    out: list[Candidate] = []
    for base in (start, *_children(start)):
        for cand in (base / ".bindings", base / "hivm-spec" / ".bindings"):
            if _is_dir(cand):
                out.append(Candidate(cand, f"hivm-spec .bindings（{base}）"))
    return out


def _from_cann_roots() -> list[Candidate]:
    """CANN / NPU-IR toolkit 安装目录下的 bindings（若该发行版带）。

    只认显式环境变量直指的安装根——见模块级纪律"不做全盘猜路径"。
    """
    out: list[Candidate] = []
    for var in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_NPU_IR_HOME", "ASCEND_BASE_DIR"):
        root = os.environ.get(var, "").strip()
        if not root:
            continue
        base = Path(root).expanduser()
        for sub in (
            "toolkit/python/site-packages/bishengir",
            "tools/bishengir/bishengir",
            REL_IN_MAIN_REPO,
        ):
            cand = base / sub
            if _is_dir(cand):
                out.append(Candidate(cand, f"{var}={root}"))
    return out


def find_candidates(start: Path | None = None) -> list[Candidate]:
    """按可信度排序的候选树（env 最可信，其次本地构建树，最后 toolkit）。去重按 realpath。"""
    start = (start or Path.cwd()).resolve()
    found: list[Candidate] = []
    for batch in (
        _from_env(),
        _from_hivm_spec_repos(start),
        _from_ancestor_builds(start),
        _from_cann_roots(),
    ):
        found.extend(batch)

    seen: set[str] = set()
    out: list[Candidate] = []
    for c in found:
        key = str(c.path.resolve()) if _is_dir(c.path) else str(c.path)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _cp310_interpreters() -> list[str]:
    """可能承载 cp310 ABI 的解释器，按可信度排序。"""
    cands: list[str] = []
    if sys.version_info[:2] == REQUIRED_PYTHON:
        cands.append(sys.executable)
    for here in (Path(__file__).resolve(), Path.cwd()):
        for up in range(2, 7):
            try:
                base = here.parents[up]
            except IndexError:
                break
            p = base / ".venv310" / "bin" / "python"
            if p.is_file():
                cands.append(str(p))
    cands.append("python3.10")
    out, seen = [], set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _probe_runnable(py: str) -> bool:
    try:
        r = subprocess.run(
            [py, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and r.stdout.strip() == "3.10"


def verify_import(bindings_dir: Path, python: str | None = None) -> tuple[bool, str]:
    """真 import 校验（子进程里做）。

    必须用子进程而不是 `sys.path.insert` + import：`.so` 加载失败会在本进程留下
    脏状态，而且 cp310 ABI 根本无法在非 3.10 的当前解释器里验证。
    """
    code = (
        "import sys;"
        "sys.path.insert(0, sys.argv[1]);"
        "import bishengir.ir as ir;"
        "ctx = ir.Context();"
        "import bishengir;"
        "reg = getattr(bishengir, 'register_dialects', None);"
        "reg(ctx) if reg else None;"
        "print('ok', sys.version.split()[0])"
    )
    py = python or sys.executable
    try:
        r = subprocess.run(
            [py, "-c", code, str(bindings_dir)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"无法启动校验子进程：{type(exc).__name__}: {exc}"
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()
        return False, tail[-1] if tail else f"退出码 {r.returncode}"
    return True, (r.stdout or "").strip()


_NO_CANDIDATE_HINT = """可执行的下一步（任选其一）：
  1) 本机已有主仓构建树：
     export HIVM_SPEC_BINDINGS=<主仓>/build/tools/bishengir/bishengir/python_packages/bishengir
  2) 有 hivm-spec 仓且能访问构建机：
     bash scripts/setup_bindings.sh && export HIVM_SPEC_BINDINGS=$PWD/.bindings
  3) 只有核心层（无 bindings）：
     hivm-spec doctor / check / gen 仍可用；run/tool/verify 六类检查全部不可用。
     这是**能力边界**，不是故障——按 FR7 如实上报，不得报成"验证失败"。"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bootstrap.py",
        description="定位并校验 bishengir bindings（只报实测结果）",
    )
    ap.add_argument("--shell", action="store_true", help="成功时只输出可 eval 的 shell 片段")
    ap.add_argument("--search-from", default=None, help="向上搜索起点（默认当前目录）")
    args = ap.parse_args(argv)

    start = Path(args.search_from).expanduser().resolve() if args.search_from else Path.cwd()
    cands = find_candidates(start)
    if not cands:
        if not args.shell:
            print("未找到 bishengir bindings 候选树。\n")
            print(_NO_CANDIDATE_HINT)
        return 2

    if not args.shell:
        print(f"候选 {len(cands)} 个：")
        for c in cands:
            mark = "✓" if c.has_markers() else "✗ 缺判别文件"
            print(f"  {mark} {c.describe()}")

    valid = [c for c in cands if c.has_markers()]
    if sys.version_info[:2] != REQUIRED_PYTHON:
        if not args.shell:
            print()
            print(
                f"⚠️ 当前解释器是 {sys.version_info[0]}.{sys.version_info[1]}，"
                f"而 bindings 是 cp{REQUIRED_PYTHON[0]}{REQUIRED_PYTHON[1]} ABI——"
                "这是**解释器**问题，不是 bindings 不可用。"
            )
            runnable = next((p for p in _cp310_interpreters() if _probe_runnable(p)), "")
            if runnable:
                print(f"   找到 cp310 解释器：{runnable}")
                print(f"   下一步：{runnable} .agent/skills/hivm-spec/scripts/bootstrap.py")
            else:
                print("   未找到可用的 3.10 解释器。核心层命令（doctor/check/gen）仍可用。")
        return 1 if valid else 2

    for c in valid:
        ok, detail = verify_import(c.path)
        if ok:
            if args.shell:
                print(f'export HIVM_SPEC_BINDINGS="{c.path}"')
            else:
                print(f"\n✓ 可用：{c.describe()}")
                print(f"  import 校验：{detail}")
                print(f'  export HIVM_SPEC_BINDINGS="{c.path}"')
                print("  下一步：hivm-spec doctor")
            return 0
        if not args.shell:
            print(f"\n✗ {c.describe()} 校验失败：{detail}")
            print("  最常见原因：`.so` 的 RUNPATH 写死了构建机绝对路径 → 该树不可搬移；")
            print("  只能在原构建机上使用，或从同源重新构建 bindings。")

    if not args.shell:
        print("\n所有候选均未通过 import 校验——按实测报告，不做乐观假设。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(None))
