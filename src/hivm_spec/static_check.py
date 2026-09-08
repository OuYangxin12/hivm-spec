"""描述静态检查器（T0.2）：生成的前置门。

**定位**：描述由 AI agent 撰写，静态检查是第一道防线。它拒绝的是**描述作者**的
错误，故所有诊断必须带描述内定位（op 名 / 参数名 / check 名），而非指向被验证的 IR
——FR7 要求区分"描述问题"与"IR 问题"。

检查失败即拒绝生成：宁可不产出工具，也不产出一个语义有洞的工具（FR6）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hivm_spec.spec import (
    KNOWN_CHECKS,
    EffectKind,
    HostFn,
    OpSpec,
    ParamKind,
    Spec,
    Trust,
)

__all__ = ["Diagnostic", "Severity", "check_spec", "format_diagnostics"]


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    severity: Severity
    #: 描述内定位，如 "op hivm.hir.vadd" / "check timeline"
    where: str
    message: str
    #: 违反的需求/决策编号，便于回溯为什么这条规则存在
    rule: str = ""

    def render(self) -> str:
        tag = f"[{self.rule}] " if self.rule else ""
        return f"{self.severity.value}: {self.where}: {tag}{self.message}"


def check_spec(spec: Spec) -> tuple[Diagnostic, ...]:
    """全量静态检查。返回全部诊断（不短路——agent 需要一次看到所有问题）。"""
    diags: list[Diagnostic] = []
    diags += _check_vm_section(spec)
    for op in spec.ops:
        diags += _check_op(spec, op)
    diags += _check_checks(spec)
    return tuple(diags)


def has_errors(diags: tuple[Diagnostic, ...]) -> bool:
    return any(d.severity is Severity.ERROR for d in diags)


def format_diagnostics(diags: tuple[Diagnostic, ...]) -> str:
    if not diags:
        return "静态检查通过：无诊断。"
    lines = [d.render() for d in diags]
    n_err = sum(1 for d in diags if d.severity is Severity.ERROR)
    n_warn = len(diags) - n_err
    lines.append(f"—— 共 {n_err} 个错误、{n_warn} 个告警")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# vm 段
# ---------------------------------------------------------------------------


def _check_vm_section(spec: Spec) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    if not spec.spaces:
        diags.append(
            Diagnostic(
                Severity.ERROR,
                "vm",
                "未声明任何 space —— 占用类检查无容量基准，结论会假精确",
                rule="FR2",
            )
        )
    for s in spec.spaces:
        if s.capacity is not None and s.capacity <= 0:
            diags.append(
                Diagnostic(Severity.ERROR, f"space {s.name}", f"capacity 必须为正：{s.capacity}")
            )
        if s.align is not None and s.align <= 0:
            diags.append(
                Diagnostic(Severity.ERROR, f"space {s.name}", f"align 必须为正：{s.align}")
            )
    return diags


# ---------------------------------------------------------------------------
# ops 段
# ---------------------------------------------------------------------------


def _check_op(spec: Spec, op: OpSpec) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    where = f"op {op.op}"
    param_names = {p.name for p in op.params}

    # 1. 引用完备性：效应引用的参数必须存在（spike 需求：rd("a") 可被校验）
    for e in op.effects:
        for field_name, ref in (
            ("target", e.target),
            ("when", e.when),
            ("event", e.event),
        ):
            if ref and ref not in param_names:
                diags.append(
                    Diagnostic(
                        Severity.ERROR,
                        where,
                        f"效应 {e.kind.value} 的 {field_name}={ref!r} "
                        f"不是本 op 的参数（现有：{sorted(param_names)}）",
                        rule="T0.2-ref",
                    )
                )

    # 2. 签名-效应一致性：Out 参数应被写、In 参数不应被写
    written = {e.target for e in op.effects if e.kind in (EffectKind.WRITE, EffectKind.COND_WRITE)}
    for p in op.params:
        if p.kind is ParamKind.OUT and p.name not in written:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    where,
                    f"参数 {p.name!r} 标注为 Out 但无写效应 —— "
                    "占用/生存期分析会漏掉该 buffer 的写入点",
                    rule="T0.2-sig",
                )
            )
        if p.kind is ParamKind.IN and p.name in written:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    where,
                    f"参数 {p.name!r} 标注为 In 却有写效应 —— 签名与效应矛盾",
                    rule="T0.2-sig",
                )
            )

    # 3. 同步效应的 pipe/event 须在 vm 段声明
    pipe_names = {p.name for p in spec.pipes}
    for e in op.effects:
        if not e.is_sync:
            continue
        for pipe_ref in (e.from_pipe, e.to_pipe):
            if not pipe_ref:
                continue
            # 参数引用（Attr[pipe]）或 vm 段字面量二者之一
            if pipe_ref not in param_names and pipe_ref not in pipe_names:
                diags.append(
                    Diagnostic(
                        Severity.ERROR,
                        where,
                        f"同步效应引用的 pipe {pipe_ref!r} 既非本 op 参数，也未在 vm 段声明",
                        rule="T0.2-sync",
                    )
                )

    # 4. 空间引用：非 @from_type 的 space 必须已声明
    space_names = {s.name for s in spec.spaces}
    for e in op.effects:
        if e.space and e.space != "@from_type" and e.space not in space_names:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    where,
                    f"效应引用未声明的 space {e.space!r}（vm 段现有：{sorted(space_names)}）",
                    rule="T0.2-space",
                )
            )

    # 5. 逃生舱 trust 封顶（OD8）
    if isinstance(op.value, HostFn) and op.trust is not Trust.PROVISIONAL:
        diags.append(
            Diagnostic(
                Severity.ERROR,
                where,
                f"逃生舱条目声明 trust={op.trust.value}，但 host_fn 值语义"
                "无法被静态检查或结构化对拍，信任封顶 provisional",
                rule="OD8",
            )
        )

    # 6. 无效应且无值语义 = 空描述（静默通过的温床）
    if not op.effects and not op.value:
        diags.append(
            Diagnostic(
                Severity.ERROR,
                where,
                "既无效应也无值语义 —— 空描述会让工具静默通过该 op，"
                "正确做法是不声明它（从而触发 COVERAGE_GAP）",
                rule="FR4",
            )
        )

    # 7. variadic 数量约束合法性
    for p in op.params:
        if p.arity is not None and not p.variadic:
            diags.append(
                Diagnostic(Severity.ERROR, where, f"参数 {p.name!r} 非 variadic 却声明了 arity")
            )
        if p.arity is not None and any(a < 0 for a in p.arity):
            diags.append(Diagnostic(Severity.ERROR, where, f"参数 {p.name!r} 的 arity 含负数"))

    # 8. pipe 归属若声明则须在 vm 段存在
    if op.pipe and op.pipe not in pipe_names:
        diags.append(
            Diagnostic(
                Severity.ERROR,
                where,
                f"pipe 归属 {op.pipe!r} 未在 vm 段声明",
                rule="T0.2-pipe",
            )
        )

    return diags


# ---------------------------------------------------------------------------
# checks 段
# ---------------------------------------------------------------------------


def _check_checks(spec: Spec) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    if not spec.checks:
        diags.append(Diagnostic(Severity.WARNING, "checks", "未声明任何 check —— 不会生成任何工具"))

    for c in spec.checks:
        where = f"check {c.name}"
        if c.name not in KNOWN_CHECKS:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    where,
                    f"未知 check（已知：{sorted(KNOWN_CHECKS)}）—— 拒绝生成一个什么都不做的工具",
                    rule="T0.2-check",
                )
            )
            continue

        # ub_occupancy 需要有容量的 space，否则无法判定溢出
        if c.name == "ub_occupancy":
            spaces = c.options.get("spaces") or [s.name for s in spec.spaces]
            declared = {s.name: s for s in spec.spaces}
            for sp in spaces:
                if sp not in declared:
                    diags.append(Diagnostic(Severity.ERROR, where, f"引用未声明的 space {sp!r}"))
                elif declared[sp].capacity is None:
                    diags.append(
                        Diagnostic(
                            Severity.ERROR,
                            where,
                            f"space {sp!r} 无 capacity —— 占用检查无法判定溢出，结论将不可信",
                            rule="FR2",
                        )
                    )

        # timeline 需要 pipe 与 event 模型
        if c.name == "timeline":
            if not spec.pipes:
                diags.append(
                    Diagnostic(Severity.ERROR, where, "vm 段未声明 pipe —— 无法构造时间线")
                )
            if not spec.events:
                diags.append(
                    Diagnostic(
                        Severity.ERROR,
                        where,
                        "vm 段未声明 event —— 无法判定 set/wait 配对",
                    )
                )
            sync_ops = [o for o in spec.ops if any(e.is_sync for e in o.effects)]
            if not sync_ops:
                diags.append(
                    Diagnostic(
                        Severity.ERROR,
                        where,
                        "无任何 op 声明同步效应 —— 时序/死锁判定将永远得出"
                        "'未发现问题'，属危险的假阴性",
                        rule="FR3",
                    )
                )

        # equivalence 需要值语义
        if c.name == "equivalence":
            valued = [o for o in spec.ops if o.value]
            if not valued:
                diags.append(
                    Diagnostic(
                        Severity.ERROR,
                        where,
                        "无任何 op 声明值语义 —— 等价验证无从比较",
                        rule="FR1",
                    )
                )
    return diags
