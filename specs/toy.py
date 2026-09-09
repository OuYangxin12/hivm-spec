"""toy 描述（T0.6）：M0 的垂直打通样本，也是 M1 的输入。

范围刻意最小：仅 `load` / `vadd` / `store` + 一对 `set_flag`/`wait_flag`，
覆盖 L0 语料 `loop_load_add_store.mlir` 与 `cross_iter_event_pair.mlir`。

三段结构见 design-framework §4.2。所有 trust 均为 `provisional`——尚未与主仓
C++ 链对拍（升级须附对拍证据，由 spec-gate R2 机械强制）。
"""

from __future__ import annotations

from hivm_spec.spec import Attr, In, Out, Spec, sync_barrier, sync_set, sync_wait, wr

# A3（membase）容量常量。真实值应由 D9 的账本绊线与主仓同步；
# 此处为 M0 打通用的占位基线，已在 trust 中体现为 provisional。
UB_SIZE = 192 * 1024
UB_ALIGN = 32
CBUF_SIZE = 512 * 1024

spec = Spec(name="hivm_toy", arch="a3")

# --- vm 段：状态模型 -------------------------------------------------------

spec.space("gm", capacity=None)  # 全局内存不设容量上限
spec.space("ub", capacity=UB_SIZE, align=UB_ALIGN)
spec.space("cbuf", capacity=CBUF_SIZE, align=UB_ALIGN)

spec.pipe("PIPE_V", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_FIX", "PIPE_S")
spec.event(*(f"EVENT_ID{i}" for i in range(8)))

# --- ops 段：op 语义 ------------------------------------------------------
# 真实语法（取自主仓语料）：
#   hivm.hir.load ins(%gm : memref<...>) outs(%ub : memref<...>)

spec.op(
    "hivm.hir.load",
    params={"src": In("buf"), "dst": Out("buf")},
    effects=(wr("dst"),),
    pipe="PIPE_MTE2",
    value="copy(src, into=dst)",
    doc="GM → UB 搬运。space 由 memref 的 #hivm.address_space<> 提取（@from_type）。",
)

spec.op(
    "hivm.hir.store",
    params={"src": In("buf"), "dst": Out("buf")},
    effects=(wr("dst"),),
    pipe="PIPE_MTE3",
    value="copy(src, into=dst)",
    doc="UB → GM 回写。",
)

spec.op(
    "hivm.hir.vadd",
    params={"a": In("buf"), "b": In("buf"), "out": Out("buf")},
    effects=(wr("out"),),
    pipe="PIPE_V",
    value="elementwise(add, a, b, into=out)",
    doc="逐元素加。",
)

spec.op(
    "hivm.hir.vmul",
    params={"a": In("buf"), "b": In("buf"), "out": Out("buf")},
    effects=(wr("out"),),
    pipe="PIPE_V",
    value="elementwise(mul, a, b, into=out)",
    doc="逐元素乘（L0 语料 nested_loop_scope 需要）。",
)

# 同步 op —— spike ② 的结论落地：同步效应不可用 rd/wr 表达。
spec.op(
    "hivm.hir.set_flag",
    params={
        "set_pipe": Attr("pipe"),
        "wait_pipe": Attr("pipe"),
        "event": Attr("event"),
    },
    effects=(sync_set(event="event", from_pipe="set_pipe", to_pipe="wait_pipe"),),
    doc="置事件标志（intra-core）。配对由 M2 引擎在 VIR 上分析，不在描述中表达。",
)

spec.op(
    "hivm.hir.wait_flag",
    params={
        "set_pipe": Attr("pipe"),
        "wait_pipe": Attr("pipe"),
        "event": Attr("event"),
    },
    effects=(sync_wait(event="event", from_pipe="set_pipe", to_pipe="wait_pipe"),),
    doc="等事件标志（intra-core）。",
)

# pipe_barrier：同一核内全 pipe 屏障。
spec.op(
    "hivm.hir.pipe_barrier",
    params={"pipe": Attr("pipe")},
    effects=(sync_barrier(pipe="pipe"),),
    doc="pipe 屏障。屏障不携带 event，故用独立的 sync_barrier 原语。",
)

# 跨核同步（sync_block_*）：与 set_flag/wait_flag 的区别在于作用域是**核间**，
# 这正是 preload 死锁案例的现场。core 归属由 IR 属性提供，描述层只声明效应。
spec.op(
    "hivm.hir.sync_block_set",
    params={
        "core": Attr("core"),
        "set_pipe": Attr("pipe"),
        "wait_pipe": Attr("pipe"),
        "flag": Attr("event"),
    },
    effects=(sync_set(event="flag", from_pipe="set_pipe", to_pipe="wait_pipe"),),
    doc="跨核置标志。",
)

spec.op(
    "hivm.hir.sync_block_wait",
    params={
        "core": Attr("core"),
        "set_pipe": Attr("pipe"),
        "wait_pipe": Attr("pipe"),
        "flag": Attr("event"),
    },
    effects=(sync_wait(event="flag", from_pipe="set_pipe", to_pipe="wait_pipe"),),
    doc="跨核等标志。计数配平不等于顺序可行——顺序判定由 M2 引擎在 VIR 上完成。",
)

# --- checks 段：要生成的工具 ----------------------------------------------

# unroll_bound=16：未知 trip 循环的缺省展开界（M2）。静态 trip 全量展开不受此限；
# 界只影响探索口径，结论恒携带"展开界"限定语（T2.5）。
spec.check("ub_occupancy", spaces=["ub", "cbuf"])
spec.check("timeline", scheduling="conservative", strategies=4, unroll_bound=16)

# --- 语义假设段：未经对拍的口径必须登记（D9 前置，M2 审查发现 3） ----------

# 此前这些假设只写在 docs/tasks/M2.md §9 的散文里——机器读不到、结论里也不
# 声明，等于把全模型最敏感的语义决策留在账本之外。登记后：进配置文档与信任
# 账本（frozen_by_assumption 冻结 trust 升级），直到与主仓 C++ 链对拍销案。
spec.assume(
    "timeline/flag-initial-state",
    "flag 初态 = 已装载（INITIAL_ARM=1，set 视为 re-arm）",
    authority="upstream-cpp",
    rationale=(
        "M1 卡『首个 re-arm set』措辞隐含 flag 出厂即 armed；且初态已装载使等待"
        "更易放行，方向上压制假阳性死锁（NFR2）。同泳道 1:1 配平 + 初始装载 = "
        "自续握手（健康），2:1 起才饿死。"
    ),
    # 若真实初态为未装载（0），wait-before-first-set 的真实死锁会被放行
    risk_direction="false-negative",
    resolve_by="M3",
)
spec.assume(
    "timeline/pair-direction",
    "set/wait 配对只按 event id，不依赖 set_pipe/wait_pipe 的方向约定",
    authority="upstream-cpp",
    rationale=(
        "IR 括号中 set_pipe/wait_pipe 的方向语义尚未与主仓 C++ 链对拍定稿，"
        "判定刻意不依赖方向（方向只影响展示与泳道归属）。"
    ),
    risk_direction="both",
    resolve_by="M3",
)
