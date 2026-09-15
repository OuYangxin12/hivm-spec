"""hivm-spec CLI 入口（D2：统一薄 CLI，实现是"引擎 + 配置文档"）。

子命令分层：

| 命令 | 层 | 用途 |
|---|---|---|
| `doctor` | 核心层 | 环境自检；缺什么给什么 remedy |
| `check` / `gen` | 核心层 | 描述静态检查 / 描述→配置文档 |
| `tool <name>` | IR 接口层 | 单类检查 |
| `run` | IR 接口层 | 一份 IR 的全套检查（编排，不新增判定） |
| `verify` | IR 接口层 | **agent 回路入口**：doctor 前置门 + 全套 + 能力自述 |

**退出码是契约的一部分**：0=验过无问题 / 1=验出问题 / 2=环境跑不起来 /
3=能力未实现 / 4=覆盖缺口 / 5=描述不可信。缺口与"验证失败"用不同码——脚本
不得把"没验成"读成"验过了"（FR7）。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hivm_spec.vir import Access, Effect

if TYPE_CHECKING:
    from hivm_spec.run_checks import RunReport
    from hivm_spec.spec import Spec

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_PENDING = 3  # 阶段未实现：可被脚本区分，不与"验证失败"混淆
#: 环境前置门未通过（`verify` 专用）。与 1/3/4/5 并列而非替代：
#: "跑不起来"既不是"验出问题"(1)，也不是"这份 IR 验不全"(4)——混用会把环境
#: 故障伪装成被验对象的属性（FR7 明令区分）。
EXIT_ENV = 2

#: 已实现的工具（equivalence 属 M3，保持 PENDING）
_IMPLEMENTED_TOOLS = (
    "ub_occupancy",
    "timeline",
    "equivalence",
    "sync_pairing",
    "uninit_read",
    "operand_wiring",
)
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

    p_doctor = sub.add_parser("doctor", help="环境自检：检查 Python/bindings/依赖是否就绪")
    p_doctor.set_defaults(cmd="doctor")

    # `run` 与 `verify` 共享同一套选择项。**用 parents 共享而非复制**：
    # 两份独立的 add_argument 必然漂移（同一个坑在 doctor/bindings 探测上已经踩过）。
    p_common = argparse.ArgumentParser(add_help=False)
    p_common.add_argument(
        "-c",
        "--config",
        default=None,
        help=f"配置文档。缺省用 {DEFAULT_CONFIG}，不存在则自动生成",
    )
    p_common.add_argument(
        "--spec",
        nargs="+",
        default=None,
        help="自动生成配置文档时用的描述文件（缺省 specs/toy.py specs/cv.py）",
    )
    p_common.add_argument("--json", metavar="PATH", help="把完整结论写为 JSON（供 agent 消费）")
    p_common.add_argument(
        "--bound",
        type=int,
        default=None,
        help="循环展开界（缺省取描述 unroll_bound，否则 16）",
    )
    p_common.add_argument(
        "--mode",
        choices=("concrete", "symbolic"),
        default="concrete",
        help="等价验证走具体档还是符号档（二者结论并列而非替代，故不同时跑）",
    )
    p_common.add_argument(
        "--anchor",
        metavar="IR",
        default=None,
        help="对拍锚点 IR。不给则等价验证记为 COVERAGE_GAP（本工具不做自比）",
    )

    # D12：位置输入恒为一份 IR
    p_run = sub.add_parser(
        "run", parents=[p_common], help="跑一份 IR 的全部适用检查（自动定位配置文档）"
    )
    p_run.add_argument("input", metavar="IR", help="待验 MLIR")

    p_verify = sub.add_parser(
        "verify",
        parents=[p_common],
        help="一键验证：doctor 前置门 + 全套检查 + 能力自述（agent 回路入口）",
    )
    p_verify.add_argument("input", metavar="IR", help="待验 MLIR")
    p_verify.add_argument(
        "--summary-json",
        metavar="PATH",
        default=None,
        help="与 --json 等价（保留给脚本习惯；两者同给只写一次）",
    )

    p_check = sub.add_parser("check", help="描述静态检查（T0.2，生成前置门）")
    p_check.add_argument("descriptions", nargs="+", help="待检查的描述文件")

    p_tool = sub.add_parser("tool", help="运行 spec 工具（M1+）")
    p_tool.add_argument(
        "name", help="工具名，如 ub_occupancy / timeline / equivalence / sync_pairing"
    )
    p_tool.add_argument(
        "-c",
        "--config",
        default=None,
        help=f"配置文档（由 gen 产出）。缺省用 {DEFAULT_CONFIG}，不存在则自动生成",
    )
    p_tool.add_argument(
        "--spec",
        nargs="+",
        default=None,
        help="自动生成配置文档时用的描述文件（缺省 specs/toy.py specs/cv.py）",
    )
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
    # 等价验证的锚点走**选项**而非第二个位置参数（M3 卡 §4 要点 1）：
    # 位置参数恒为一份 IR，D12 的契约不破；同时"谁是被验证对象"在命令行上一眼
    # 可辨——这直接关系到 verdict 归属谁。
    #
    # 注意：**没有"缺省自比"**。不给 --anchor 时 equivalence 报 COVERAGE_GAP，
    # 绝不拿同一份 IR 自比后报 OK（§4 要点 2：那是最典型的自欺形态）。
    p_tool.add_argument(
        "--mode",
        choices=("concrete", "symbolic"),
        default="concrete",
        help=(
            "equivalence：具体档（默认，这组输入上比对）或符号档"
            "（有界内所有输入，Real 语义，需 z3）。二者结论并列而非替代"
        ),
    )
    p_tool.add_argument(
        "--anchor",
        metavar="IR",
        default=None,
        help="equivalence：对拍锚点 IR（通常是变换前的 dump）。缺省不自比，报 COVERAGE_GAP",
    )
    # D12：位置输入恒为单份 IR。不接受 pass 序列——跨 pass 定位由 agent 对每份
    # dump 分别调用来编排（见 AGENTS.md §6）。
    p_tool.add_argument(
        "inputs",
        nargs="+",
        metavar="IR",
        help="待验 MLIR（恒一份；等价验证的锚点用 --anchor）",
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


#: 缺省描述集：此仓自带的两份描述。取并集是 gen 的既有语义。
DEFAULT_SPECS = ("specs/toy.py", "specs/cv.py")
#: 缺省配置文档路径。gen 已实测确定性（同输入两次产出 byte 级一致），故可安全缓存。
DEFAULT_CONFIG = "build/config.json"
#: 环境变量：配置文档位置。给"已安装、在任意目录下调用"的用法一个锚点——
#: 缺省的 `build/config.json` 与缺省描述都是 **CWD 相对**，离开仓目录就找不到。
ENV_CONFIG_PATH = "HIVM_SPEC_CONFIG"


def _cmd_doctor() -> int:
    """环境自检。

    退出码：有 MISSING 项 → EXIT_FAIL；仅 DEGRADED → EXIT_OK。
    理由：能力受限仍可用（跑得动就不该让脚本失败），核心不可用则应当挡住。
    """
    from hivm_spec.doctor import diagnose

    report = diagnose()
    print("hivm-spec 环境自检")
    print(report.render())
    return EXIT_FAIL if report.has_blocker else EXIT_OK


def _resolve_config(config: str | None, specs: list[str] | None) -> tuple[str | None, int]:
    """定位配置文档；不存在则自动 gen。

    返回 `(路径, 退出码)`；路径为 None 表示失败。

    **自动 gen 是安全的**：gen 已实测确定性（同输入两次产出 byte 级一致），
    所以"自动生成"不会引入不可复现性。这不是把 spec_hash 藏起来——每次结论
    里仍然带着它，审计坐标不变。
    """
    if config is not None:
        if not Path(config).is_file():
            # 保留可执行的下一步（FR5）：显式路径不存在时不自动生成——用户
            # 明确说了用哪个文件，就不该悄悄换一个——但要告诉他怎么造出来。
            print(
                f"配置文档不存在：{config}。先运行 hivm-spec gen <描述> -o {config}",
                file=sys.stderr,
            )
            return None, EXIT_FAIL
        return config, EXIT_OK

    out = Path(DEFAULT_CONFIG)
    if out.is_file():
        return str(out), EXIT_OK

    # CWD 下没有：认环境变量（安装后从任意目录调用的标准姿势）
    env_cfg = os.environ.get(ENV_CONFIG_PATH, "").strip()
    if env_cfg:
        p = Path(env_cfg).expanduser()
        if p.is_file():
            return str(p), EXIT_OK
        print(
            f"{ENV_CONFIG_PATH} 指向的文件不存在：{p}。先运行 hivm-spec gen <描述> -o {p}",
            file=sys.stderr,
        )
        return None, EXIT_FAIL

    sources = list(specs) if specs else [s for s in DEFAULT_SPECS if Path(s).is_file()]
    if not sources:
        print(
            f"未找到配置文档 {DEFAULT_CONFIG}，未设 {ENV_CONFIG_PATH}，且缺省描述"
            f"（{', '.join(DEFAULT_SPECS)}）也不存在。"
            "请用 -c 指定配置文档、设 HIVM_SPEC_CONFIG，或用 --spec 指定描述文件",
            file=sys.stderr,
        )
        return None, EXIT_FAIL

    print(f"未找到 {DEFAULT_CONFIG}，从描述自动生成：{' '.join(sources)}", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    rc = _cmd_gen(sources, str(out), None)
    if rc != EXIT_OK:
        return None, rc
    return str(out), EXIT_OK


def _lower_ir(
    path_str: str,
    config: dict[str, Any],
    label: str = "IR",
) -> tuple[Any, int]:
    """把一份 MLIR 文件降为 VIR。

    返回 `(lowered, 退出码)`；lowered 为 None 表示失败。抽出来是因为 `tool` 与
    `run` 都要做这件事，两份拷贝必然漂移。
    """
    from hivm_spec.bindings import BindingsError
    from hivm_spec.ir_engine import lower_module_text

    ir_path = Path(path_str)
    if not ir_path.is_file():
        print(f"{label} 文件不存在：{ir_path}", file=sys.stderr)
        return None, EXIT_FAIL

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
        print(f"环境不可用，未能验证{label}：{exc}", file=sys.stderr)
        return None, EXIT_PENDING
    return lowered, EXIT_OK


def _run_all(args: argparse.Namespace) -> tuple[RunReport | None, str, int]:
    """定位配置 → 降级 IR → 跑全套检查。返回 `(RunReport | None, 配置路径, 退出码)`。

    `run` 与 `verify` 共用这一条通路：两者的差别只在**前置门**与**输出形态**，
    判定与编排必须完全一致，否则"同一份 IR 两种结论"就成了自造的漂移源。
    """
    from hivm_spec.assemble import load_config
    from hivm_spec.run_checks import run_checks

    resolved, rc = _resolve_config(args.config, args.spec)
    if resolved is None:
        return None, "", rc
    config, spec_hash = load_config(Path(resolved))

    lowered, rc = _lower_ir(args.input, config)
    if lowered is None:
        return None, "", rc
    for note in lowered.engine_notes:
        print(f"引擎提示：{note}", file=sys.stderr)

    anchor_module = None
    if args.anchor is not None:
        anchor_lowered, rc = _lower_ir(args.anchor, config, label="锚点 IR")
        if anchor_lowered is None:
            return None, "", rc
        anchor_module = anchor_lowered.module

    report = run_checks(
        lowered.module,
        config,
        spec_hash,
        anchor=anchor_module,
        bound=args.bound,
        mode=args.mode,
    )
    return report, str(resolved), report.exit_code


def _cmd_run(args: argparse.Namespace) -> int:
    """跑全部适用检查（编排；不新增判定逻辑）。"""
    report, _config_path, rc = _run_all(args)
    if report is None:
        return rc
    print(report.render())
    if args.json:
        report.write_json(args.json)
        print(f"结论已写入 {args.json}")
    return rc


def _cmd_verify(args: argparse.Namespace) -> int:
    """一键验证：前置门 + 全套检查 + 能力自述。

    与 `run` 的唯一区别是**跑之前先挡住跑不起来的情形**，并**把能力边界写进结论**。
    """
    from hivm_spec.doctor import diagnose
    from hivm_spec.verify import VerifyReport

    report = diagnose()
    if report.has_blocker:
        # 环境问题 ≠ 覆盖缺口 ≠ 验证失败：三码必须分开（FR7）
        print("hivm-spec verify：环境前置门未通过，未产出任何验证结论", file=sys.stderr)
        print(report.render(), file=sys.stderr)
        return EXIT_ENV

    run_report, config_path, rc = _run_all(args)
    if run_report is None:
        return rc

    verify = VerifyReport(
        run_report=run_report,
        doctor=report,
        mode=args.mode,
        input_ir=args.input,
        config_path=config_path,
        anchor=args.anchor,
    )
    print(verify.render())
    out = args.json or args.summary_json
    if out:
        verify.write_json(out)
        print(f"结论已写入 {out}")
    return run_report.exit_code


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "check":
        return _cmd_check(args.descriptions)
    if args.cmd == "gen":
        return _cmd_gen(args.description, args.output, args.timestamp)

    if args.cmd == "doctor":
        return _cmd_doctor()

    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "verify":
        return _cmd_verify(args)

    return _cmd_tool(
        args.name,
        args.config,
        args.inputs,
        args.json,
        not args.no_chart,
        bound=args.bound,
        strategy=args.strategy,
        trace=args.trace,
        anchor=args.anchor,
        mode=args.mode,
        specs=args.spec,
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
    config_path: str | None,
    inputs: list[str],
    json_out: str | None,
    chart: bool = True,
    *,
    bound: int | None = None,
    strategy: str = "all",
    trace: str | None = None,
    anchor: str | None = None,
    mode: str = "concrete",
    specs: list[str] | None = None,
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

    resolved, rc = _resolve_config(config_path, specs)
    if resolved is None:
        return rc
    config, spec_hash = load_config(Path(resolved))

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
    if name == "equivalence":
        # 锚点单独 lower。**缺锚点不报错也不自比**——交给 run_equivalence 出
        # COVERAGE_GAP（§4 要点 2），这样"没锚点"在结论里留痕，而不是命令失败
        # 后被读作"环境问题"。
        anchor_module = None
        if anchor is not None:
            anchor_path = Path(anchor)
            if not anchor_path.is_file():
                print(f"锚点 IR 不存在：{anchor_path}", file=sys.stderr)
                return EXIT_FAIL
            try:
                anchor_lowered = lower_module_text(
                    anchor_path.read_text(encoding="utf-8"),
                    modeled,
                    source=str(anchor_path),
                    op_effects=effects,
                    op_pipes=pipes,
                    arch=config.get("arch", "a3"),
                )
            except BindingsError as exc:
                print(f"环境不可用，未能验证锚点：{exc}", file=sys.stderr)
                return EXIT_PENDING
            anchor_module = anchor_lowered.module
        kwargs = {"anchor": anchor_module, "bound": bound, "mode": mode}
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
