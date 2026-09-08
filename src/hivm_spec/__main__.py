"""hivm-spec CLI 入口（D2：统一薄 CLI，实现是"引擎 + 配置文档"）。

当前为骨架：子命令契约已固定，未实现的阶段返回明确的 PENDING 退出码而非
假成功——与 `COVERAGE_GAP`/`UNTRUSTED_DESCRIPTION` 同一原则：缺口必须显式。
"""

from __future__ import annotations

import argparse
import sys

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_PENDING = 3  # 阶段未实现：可被脚本区分，不与"验证失败"混淆


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hivm-spec",
        description="Verifier generator for AscendNPU-IR (HIVM) semantic checks",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen", help="描述 → 配置文档（T0.5）")
    p_gen.add_argument("description", help="描述模块路径（Python 模块或文件）")
    p_gen.add_argument("-o", "--output", default="config.json", help="配置文档输出路径")

    p_check = sub.add_parser("check", help="描述静态检查（T0.2，生成前置门）")
    p_check.add_argument("descriptions", nargs="+", help="待检查的描述文件")

    p_tool = sub.add_parser("tool", help="运行 spec 工具（M1+）")
    p_tool.add_argument("name", help="工具名，如 ub_occupancy / timeline")
    p_tool.add_argument("inputs", nargs="+", help="输入 MLIR 文件")

    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    pending = {
        "gen": "T0.3/T0.5 归一化与配置文档产出",
        "check": "T0.2 静态检查器",
        "tool": "T1.x spec 工具装配",
    }
    task = pending.get(args.cmd)
    print(f"PENDING({task}) — 子命令 '{args.cmd}' 的契约已定义，实现未落地。")
    print("参见 docs/milestone-plan.md 对应任务；本命令不产出结论以避免误导。")
    return EXIT_PENDING


if __name__ == "__main__":
    sys.exit(main())
