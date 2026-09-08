"""hivm-spec CLI 入口（D2：统一薄 CLI，实现是"引擎 + 配置文档"）。

`gen`/`check` 已实现（T0.2/T0.3/T0.5）；`tool` 仍返回明确的 PENDING 退出码而非
假成功——与 `COVERAGE_GAP`/`UNTRUSTED_DESCRIPTION` 同一原则：缺口必须显式。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hivm_spec.spec import Spec

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
    p_gen.add_argument(
        "--timestamp",
        default=None,
        help="固定账本时间戳（用于可复现构建；配置文档本身与时间无关）",
    )

    p_check = sub.add_parser("check", help="描述静态检查（T0.2，生成前置门）")
    p_check.add_argument("descriptions", nargs="+", help="待检查的描述文件")

    p_tool = sub.add_parser("tool", help="运行 spec 工具（M1+）")
    p_tool.add_argument("name", help="工具名，如 ub_occupancy / timeline / equivalence")
    # D12：输入恒为单份 IR；仅等价验证取两份（待验 + 锚点）。不接受 pass 序列——
    # 跨 pass 定位由 agent 对每份 dump 分别调用来编排（见 AGENTS.md §6）。
    p_tool.add_argument(
        "inputs",
        nargs="+",
        metavar="IR",
        help="输入 MLIR：单份（占用/时序）或两份 <待验> <锚点>（等价验证）",
    )

    return ap


def _load_spec(path_str: str) -> tuple[Spec | None, str]:
    """从描述文件加载 `spec` 对象。

    描述是 Python 模块（D1：宿主嵌入式 DSL），故用 importlib 按文件路径加载，
    避免要求描述必须位于 sys.path 上。
    """
    path = Path(path_str)
    if not path.is_file():
        return None, f"描述文件不存在：{path}"

    spec_obj = importlib.util.spec_from_file_location(path.stem, path)
    if spec_obj is None or spec_obj.loader is None:
        return None, f"无法加载描述模块：{path}"

    module = importlib.util.module_from_spec(spec_obj)
    try:
        spec_obj.loader.exec_module(module)
    except Exception as exc:  # 描述是用户代码，任何异常都要可读地报出
        return None, f"描述模块执行失败：{path}: {type(exc).__name__}: {exc}"

    candidate = getattr(module, "spec", None)
    if candidate is None:
        return None, f"描述模块未导出名为 'spec' 的对象：{path}"
    return candidate, ""


def _cmd_check(paths: list[str]) -> int:
    from hivm_spec.static_check import check_spec, format_diagnostics, has_errors

    failed = False
    for p in paths:
        spec, err = _load_spec(p)
        if spec is None:
            print(f"{p}: {err}", file=sys.stderr)
            failed = True
            continue
        diags = check_spec(spec)
        print(f"=== {p} ===")
        print(format_diagnostics(diags))
        if has_errors(diags):
            failed = True
    return EXIT_FAIL if failed else EXIT_OK


def _cmd_gen(path: str, output: str, timestamp: str | None) -> int:
    from hivm_spec.generate import generate, validate_config, write_outputs
    from hivm_spec.static_check import check_spec, format_diagnostics, has_errors

    spec, err = _load_spec(path)
    if spec is None:
        print(err, file=sys.stderr)
        return EXIT_FAIL

    # 静态检查是生成的前置门（T0.2）：宁可不产出工具，
    # 也不产出一个语义有洞的工具（FR6）。
    diags = check_spec(spec)
    if has_errors(diags):
        print("静态检查失败，拒绝生成：", file=sys.stderr)
        print(format_diagnostics(diags), file=sys.stderr)
        return EXIT_FAIL
    if diags:
        print(format_diagnostics(diags), file=sys.stderr)

    result = generate(spec, timestamp=timestamp)

    schema_errors = validate_config(result.config)
    if schema_errors:
        print("配置文档不符合 schema：", file=sys.stderr)
        for e in schema_errors:
            print(f"  {e}", file=sys.stderr)
        return EXIT_FAIL

    out = Path(output)
    ledger_path = write_outputs(result, out)
    print(f"配置文档：{out}")
    print(f"信任账本：{ledger_path}")
    print(f"spec_hash：{result.spec_hash}")
    print(
        f"覆盖：{len(result.ledger.entries)} 个 op"
        f"（逃生舱 {len(result.ledger.escape_hatches)}）"
        f"；最高信任：{result.ledger.max_trust()}"
    )
    if result.ledger.frozen_ops():
        print(f"⚠️ 因未处置漂移而冻结升级的 op：{result.ledger.frozen_ops()}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "check":
        return _cmd_check(args.descriptions)
    if args.cmd == "gen":
        return _cmd_gen(args.description, args.output, args.timestamp)

    print("PENDING(T1.x spec 工具装配) — 子命令 'tool' 的契约已定义，实现未落地。")
    print("参见 docs/milestone-plan.md 对应任务；本命令不产出结论以避免误导。")
    return EXIT_PENDING


if __name__ == "__main__":
    sys.exit(main())
