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

spec.check("ub_occupancy", spaces=["ub", "cbuf"])
spec.check("timeline", scheduling="conservative", strategies=4, unroll_bound=3)
