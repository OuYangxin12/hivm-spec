"""hivm-spec CLI 入口（D2：统一薄 CLI，实现是"引擎 + 配置文档"）。

`gen`/`check`/`tool ub_occupancy`/`tool timeline` 已实现（T0.x/T1.6/T2.x）；
`tool equivalence` 仍返回明确的 PENDING 退出码而非假成功——与
`COVERAGE_GAP`/`UNTRUSTED_DESCRIPTION` 同一原则：缺口必须显式。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hivm_spec.vir import Access, Effect

if TYPE_CHECKING:
    from hivm_spec.spec import Spec

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_PENDING = 3  # 阶段未实现：可被脚本区分，不与"验证失败"混淆

#: 已实现的工具（equivalence 属 M3，保持 PENDING）
_IMPLEMENTED_TOOLS = ("ub_occupancy", "timeline")
#: timeline 的策略选择（"random" 展开为全部固定种子，见 timeline.RANDOM_SEEDS）
_STRATEGY_CHOICES = ("all", "sequential", "round_robin", "pipe_priority", "random")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hivm-spec",
        description="Verifier generator for AscendNPU-IR (HIVM) semantic checks",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen", help="描述 → 配置文档（T0.5）")
    p_gen.add_argument(
        "description",
        nargs="+",
        help="描述模块路径（可多份，语义取并集；重名冲突报错）",
    )
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
    p_tool.add_argument("-c", "--config", default="config.json", help="配置文档（由 gen 产出）")
    p_tool.add_argument("--json", metavar="PATH", help="把完整结论写为 JSON")
    p_tool.add_argument("--no-chart", action="store_true", help="不在终端渲染文本图（T1.5/T2.4）")
    p_tool.add_argument(
        "--bound",
        type=int,
        default=None,
        help="timeline：未知 trip 循环的展开界（缺省取描述 unroll_bound，否则 16）",
    )
    p_tool.add_argument(
        "--strategy",
        choices=_STRATEGY_CHOICES,
        default="all",
        help="timeline：交错策略（all=全部策略+固定种子，T2.2）",
    )
    p_tool.add_argument(
        "--trace",
        metavar="PATH",
        help="timeline：把时间线写为 Chrome Trace Event Format JSON（perfetto 可导入）",
    )
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


def _cmd_gen(paths: list[str], output: str, timestamp: str | None) -> int:
    from hivm_spec.generate import generate, validate_config, write_outputs
    from hivm_spec.static_check import check_spec, format_diagnostics, has_errors

    spec, err = _load_spec(paths[0])
    if spec is None:
        print(err, file=sys.stderr)
        return EXIT_FAIL
    for extra in paths[1:]:
        other, err = _load_spec(extra)
        if other is None:
            print(err, file=sys.stderr)
            return EXIT_FAIL
        try:
            spec.merge(other)
        except Exception as exc:  # SpecError
            print(f"合并描述失败：{exc}", file=sys.stderr)
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

    return _cmd_tool(
        args.name,
        args.config,
        args.inputs,
        args.json,
        not args.no_chart,
        bound=args.bound,
        strategy=args.strategy,
        trace=args.trace,
    )


def _timeline_strategies(choice: str) -> tuple[str, ...]:
    """--strategy 选择 → 策略名清单（random 展开为全部固定种子，FR8 可复现）。"""
    from hivm_spec import timeline as tl

    if choice == "all":
        return (*tl.STRATEGIES, *(f"random(seed={k})" for k in tl.RANDOM_SEEDS))
    if choice == "random":
        return tuple(f"random(seed={k})" for k in tl.RANDOM_SEEDS)
    return (choice,)


def _cmd_tool(
    name: str,
    config_path: str,
    inputs: list[str],
    json_out: str | None,
    chart: bool = True,
    *,
    bound: int | None = None,
    strategy: str = "all",
    trace: str | None = None,
) -> int:
    from hivm_spec.assemble import load_config, run_tool
    from hivm_spec.bindings import BindingsError
    from hivm_spec.ir_engine import lower_module_text

    if name not in _IMPLEMENTED_TOOLS:
        print(
            f"PENDING(M3) 工具 {name!r} 的契约已定义，实现未落地。本命令不产出结论以避免误导。",
            file=sys.stderr,
        )
        return EXIT_PENDING

    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        print(
            f"配置文档不存在：{cfg_path}。先运行 hivm-spec gen <描述> -o {cfg_path}",
            file=sys.stderr,
        )
        return EXIT_FAIL
    config, spec_hash = load_config(cfg_path)

    if len(inputs) != 1:
        print(
            f"{name} 恒吃一份 IR（D12），收到 {len(inputs)} 份。"
            "跨 pass 定位请对每份 dump 分别调用。",
            file=sys.stderr,
        )
        return EXIT_FAIL

    ir_path = Path(inputs[0])
    if not ir_path.is_file():
        print(f"IR 文件不存在：{ir_path}", file=sys.stderr)
        return EXIT_FAIL

    modeled = {op["op"] for op in config.get("ops", [])}
    effects, pipes = _effects_from_config(config)

    try:
        lowered = lower_module_text(
            ir_path.read_text(encoding="utf-8"),
            modeled,
            source=str(ir_path),
            op_effects=effects,
            op_pipes=pipes,
            arch=config.get("arch", "a3"),
        )
    except BindingsError as exc:
        # 环境问题必须与"IR 有问题"分开（FR7）
        print(f"环境不可用，未能验证：{exc}", file=sys.stderr)
        return EXIT_PENDING

    for note in lowered.engine_notes:
        print(f"引擎提示：{note}", file=sys.stderr)

    kwargs: dict[str, Any] = {}
    if name == "timeline":
        kwargs = {"bound": bound, "strategies": _timeline_strategies(strategy)}
    result = run_tool(name, config, lowered.module, spec_hash, **kwargs)
    print(result.render())

    if chart:
        if name == "timeline":
            from hivm_spec.render import render_timeline_chart

            chart_text = render_timeline_chart(result.details)
        else:
            from hivm_spec.render import render_occupancy_chart

            chart_text = render_occupancy_chart(result.details)
        if chart_text:
            print(chart_text)

    if trace and name == "timeline":
        from hivm_spec.render import chrome_trace_json

        Path(trace).write_text(chrome_trace_json(result.details), encoding="utf-8")
        print(f"Chrome Trace：{trace}")

    if json_out:
        Path(json_out).write_bytes(result.to_json_bytes())
        print(f"完整结论：{json_out}")

    return result.exit_code


def _effects_from_config(
    config: dict[str, Any],
) -> tuple[dict[str, tuple[Effect, ...]], dict[str, str]]:
    """从配置文档重建引擎所需的效应表。

    刻意从**配置文档**而非描述源文件重建：结论的审计坐标是 spec_hash（配置文档
    的哈希），若运行时读源文件，就可能出现"结论声称基于某 spec_hash，实际用的是
    改过的源文件"——审计链断裂。
    """
    access_map = {"read": Access.READ, "write": Access.WRITE, "cond_write": Access.WRITE}
    effects: dict[str, tuple[Effect, ...]] = {}
    pipes: dict[str, str] = {}
    for op in config.get("ops", []):
        items: list[Effect] = []
        for eff in op.get("effects", []):
            access = access_map.get(eff.get("kind", ""))
            if access is None:
                continue  # 同步效应走 VSync
            items.append(
                Effect(
                    access=access,
                    space=eff.get("space") or "@from_type",
                    target=eff.get("target", ""),
                )
            )
        if items:
            effects[op["op"]] = tuple(items)
        if op.get("pipe"):
            pipes[op["op"]] = op["pipe"]
    return effects, pipes


if __name__ == "__main__":
    sys.exit(main())
