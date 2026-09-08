"""T0.0b 语法 spike（D11）：三个代表性 op 的示意描述，用于 API 定稿前探边界。

**这不是实现**，是一次性探针：D11 要求在 T0.1 定稿 Spec API 之前，用示意语法
试写三个代表性 op，确认表达力边界，避免 API 被 toy 描述（load/vadd/store）锁定。

三个 op 按"效应复杂度递增"选取，全部取自真实 ODS 与真实测试语料：

① `hivm.hir.vadd`
   平凡值语义 + 平凡效应 → **可声明式表达**
② `hivm.hir.set_flag` / `wait_flag`
   跨迭代事件对；效应是"同步"而非内存读写 → **可声明式表达**
   （前提：DSL 须有独立的 `sync_*` 效应原语）
③ `hivm.hir.mmadL1`
   布局代数 + variadic `sync_related_args` + 条件语义
   → **部分逃生舱**（效应可声明；值语义走 host 函数）

运行 `python specs/spike_t0_0b.py` 打印结论摘要。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# 示意语法的最小支撑（spike 专用，不进 src/；T0.1 将据此定稿真实 API）
# ---------------------------------------------------------------------------


class Verdict(Enum):
    DECLARATIVE = "可声明式表达"
    PARTIAL_ESCAPE = "部分逃生舱（效应可声明，值语义走 host 函数）"
    FULL_ESCAPE = "完全逃生舱"


@dataclass
class SpikeResult:
    op: str
    verdict: Verdict
    expressible: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    api_requirements: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ① vadd —— 平凡基线
# ---------------------------------------------------------------------------
# 真实语法：hivm.hir.vadd ins(%a, %b : T, T) outs(%c : T) -> T
#
# 示意描述：
#
#   @spec.op("hivm.hir.vadd")
#   def vadd(a: In[Buf], b: In[Buf], c: Out[Buf]) -> Buf:
#       """effects 由签名自动推导：读 a、读 b、写 c"""
#       return elementwise(add, a, b, into=c)
#
SPIKE_VADD = SpikeResult(
    op="hivm.hir.vadd",
    verdict=Verdict.DECLARATIVE,
    expressible=[
        "效应：读 a/b、写 c —— 可由 In/Out 类型标注自动推导，无需手写 effects",
        "值语义：elementwise(add)，M3 直接可用",
        "pipe 归属：PIPE_V，静态标注",
    ],
    api_requirements=[
        "In[]/Out[] 参数标注 → 自动推导 Effect（减少描述样板，降低 agent 出错面）",
        "elementwise 组合子（M3 值执行复用）",
    ],
)

# ---------------------------------------------------------------------------
# ② set_flag / wait_flag —— 跨迭代事件对
# ---------------------------------------------------------------------------
# 真实语法：hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE3>, <EVENT_ID0>]
#           hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE3>, <EVENT_ID0>]
#
# 关键发现：这里的"效应"不是内存读写，而是**对同步状态的读写**。若 DSL 只有
# rd/wr(space) 一种效应原语，本 op 无法表达 —— 必须有独立的 sync 效应类别。
#
# 示意描述：
#
#   @spec.op("hivm.hir.set_flag")
#   def set_flag(set_pipe: Attr[Pipe], wait_pipe: Attr[Pipe], event: Attr[EventId]):
#       effects = [sync_set(event=event, from_pipe=set_pipe, to_pipe=wait_pipe)]
#
#   @spec.op("hivm.hir.wait_flag")
#   def wait_flag(set_pipe: Attr[Pipe], wait_pipe: Attr[Pipe], event: Attr[EventId]):
#       effects = [sync_wait(event=event, from_pipe=set_pipe, to_pipe=wait_pipe)]
#
SPIKE_EVENT_PAIR = SpikeResult(
    op="hivm.hir.set_flag / hivm.hir.wait_flag",
    verdict=Verdict.DECLARATIVE,
    expressible=[
        "同步效应：sync_set / sync_wait，带 event id 与 from/to pipe",
        "跨迭代配对：不在描述中表达 —— 配对是 M2 引擎在 VIR 上的分析结果，"
        "描述只声明单个 op 的效应（职责边界清晰）",
        "属性来源：从 MLIR 属性提取（[<PIPE_X>, <EVENT_IDn>] 形态）",
    ],
    gaps=[
        "语义完整性：'wait 的 credit 从哪来'（seed credit）不是单 op 属性，"
        "属 vm 段的初始状态建模 —— 已在 §4.2 的 vm 段范围内，非缺口",
    ],
    api_requirements=[
        "**独立的 sync 效应原语**（sync_set/sync_wait/barrier），不可用 rd/wr 硬套",
        "Attr[] 标注：声明属性来源于 MLIR 属性而非操作数",
        "EventId / Pipe 作为一等类型（供静态检查校验取值合法性）",
    ],
)

# ---------------------------------------------------------------------------
# ③ mmadL1 —— 预期的逃生舱场景
# ---------------------------------------------------------------------------
# 真实 ODS 签名（HIVMMacroOps.td）：
#   ins a, b, init_condition: I1, real_m/real_k/real_n: Index, c,
#       sync_related_args: Variadic<I64>, unit_flag_cond: Variadic<I1>
#
# 真实用法：
#   %0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c160, %c320, %c80 :
#          tensor<160x320xf16>, tensor<320x80xf16>, i1, index, index, index)
#          outs(%empty : tensor<160x80xf32>) -> tensor<160x80xf32>
#
# 三处棘手点：
#   (a) 布局代数：L1/L0A/L0B 的分形搬运与 nz 布局 —— 声明式表达代价极高；
#   (b) variadic sync_related_args：由 inject-sync pass 管理，长度可变，
#       语义是"同步插桩的挂载点"而非数据；
#   (c) init_condition 造成语义分支（是否清零 L0C）。
#
# 示意描述：
#
#   @spec.op("hivm.hir.mmadL1")
#   def mmadL1(a: In[Buf], b: In[Buf], init_condition: In[Scalar],
#              real_m: In[Index], real_k: In[Index], real_n: In[Index],
#              c: Out[Buf], sync_related_args: Variadic[In[Scalar]] = (),
#              unit_flag_cond: Variadic[In[Scalar]] = ()):
#       # 效应可声明（M1/M2 只需这些）
#       effects = [rd(a, space="cbuf"), rd(b, space="cbuf"),
#                  wr(c, space="cc"),
#                  cond_wr(c, when=init_condition)]   # init_condition 分支
#       # 值语义走逃生舱（OD8）：布局代数不进声明式描述
#       value = host_fn("mmad_reference", trust="provisional")
#
SPIKE_MMAD = SpikeResult(
    op="hivm.hir.mmadL1",
    verdict=Verdict.PARTIAL_ESCAPE,
    expressible=[
        "效应：读 a/b（cbuf）、写 c（cc）—— M1 占用与 M2 时序所需的全部信息",
        "real_m/k/n 作为尺寸来源 provenance（VAlloc.size_origin 的输入）",
        "init_condition 的条件写效应：需 cond_wr 原语",
    ],
    gaps=[
        "值语义（布局代数 + 分形搬运）→ 走 OD8 逃生舱 host 函数，trust 封顶 provisional",
        "variadic sync_related_args 语义为'同步插桩挂载点'，"
        "描述层只需声明其存在与数量约束，不建模其内容",
    ],
    api_requirements=[
        "Variadic[] 参数标注 + 数量约束校验（ODS 要求 empty 或特定 size）",
        "cond_wr（条件效应）原语 —— 否则 init_condition 语义丢失",
        "host_fn 逃生舱：声明值语义由 Python 函数提供，trust 自动封顶 provisional",
        "空间标注可来自 memref 的 #hivm.address_space<> 而非仅描述硬编码",
    ],
)

ALL_SPIKES = (SPIKE_VADD, SPIKE_EVENT_PAIR, SPIKE_MMAD)


def summarize() -> str:
    lines = ["T0.0b 语法 spike 结论（D11）", "=" * 60]
    for s in ALL_SPIKES:
        lines.append(f"\n[{s.verdict.value}] {s.op}")
        for e in s.expressible:
            lines.append(f"  + {e}")
        for g in s.gaps:
            lines.append(f"  ~ {g}")
        for r in s.api_requirements:
            lines.append(f"  → API 需求：{r}")

    escapes = [s for s in ALL_SPIKES if s.verdict is not Verdict.DECLARATIVE]
    lines.append("\n" + "=" * 60)
    lines.append(f"逃生舱数量：{len(escapes)}/{len(ALL_SPIKES)}")
    if len(escapes) >= 2:
        lines.append(
            "⚠️ D11 止损条件触发（≥2 落逃生舱）：须重估 §4.4 DSL 概念清单边界，"
            "并上报决策后方可定稿 T0.1 API"
        )
    else:
        lines.append("✅ 未触发 D11 止损条件（<2 落逃生舱）：可据上述 API 需求定稿 T0.1")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize())
