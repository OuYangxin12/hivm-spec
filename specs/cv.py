"""cv-pipelining kernel 族的描述（T1.0b）：目标 L2 语料的 9 个未建模 op。

建模依据是 **ODS 真值**（`bishengir/include/bishengir/Dialect/HIVM/IR/*.td`，
主仓 commit `80db8e8c62a6`），不是从 IR 用例反推——用例是 pass 作者写的最小
fixture，其形态（tensor 化、无地址空间标注）不代表 op 语义。

效应级建模（D3）：只声明"动了哪个参数"，值语义能声明则声明为表达式字符串，
不能（布局代数）则走 `host_fn` 逃生舱（trust 封顶 provisional，OD8）。

op 语义层（ins/outs 顺序、效应方向）来源：
- `HIVMMacroOps.td`（mmadL1：init_condition = "The condition under which
  data in L0C is cleared before use"）；
- `HIVMDMAOps.td`（fixpipe = L0C→OUT/L1/UB，OpPipeTrait PIPE_FIX；
  nd2nz = PIPE_MTE2；copy = SinglePipeOpTrait、无默认 pipe）；
- `HIVMVectorOps.td`（vexp/vcast/vbrc 经 vecSpecialTraits 归 PIPE_V）；
- `HIVMOps.td`（set_atomic：影响**后续**写 GM 的原子性，kind=NONE 复位；
  debug：MemoryEffects<[MemRead, MemWrite]>）。
"""

from __future__ import annotations

from hivm_spec.spec import In, Out, Spec, cond_wr, host_fn, rd, wr

spec = Spec(name="hivm_cv", arch="a3")

# --- vm 段：与 toy 同一状态模型（A3 容量占位基线，provisional） ----------

spec.space("gm", capacity=None)
spec.space("ub", capacity=192 * 1024, align=32)
spec.space("cbuf", capacity=512 * 1024, align=32)

spec.pipe("PIPE_V", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_FIX", "PIPE_S")
spec.event(*(f"EVENT_ID{i}" for i in range(8)))

# --- ops 段 ---------------------------------------------------------------

spec.op(
    "hivm.hir.mmadL1",
    params={
        "a": In("buf"),
        "b": In("buf"),
        "init_condition": In("scalar"),
        "real_m": In("scalar"),
        "real_k": In("scalar"),
        "real_n": In("scalar"),
        "out": Out("buf"),
    },
    # ODS：init_condition 决定 L0C 是否先清零——即"条件写 C"。
    # 假条件下的累加语义若用 wr 表达会丢掉"接着累加"的分支。
    effects=(cond_wr("out", when="init_condition"),),
    # ODS：MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">——MTE1 搬 tile、
    # M 计算，双 pipe。M1 占用分析不消费 pipe；M2 时序需要建模双 pipe 归属。
    pipe="PIPE_M",
    value=host_fn(
        "mmad_l1_value",
        reason="MMA 值语义依赖分形布局代数（OpLayoutInterface），无法声明式表达",
    ),
    doc="L1 局部矩阵乘。效应级：cond_wr(C, when=init_condition)。值语义走逃生舱。",
)

spec.op(
    "hivm.hir.fixpipe",
    params={"src": In("buf"), "dst": Out("buf")},
    effects=(wr("dst"),),
    pipe="PIPE_FIX",
    value="copy(src, into=dst)",
    doc="L0C → OUT/L1/UB 搬运（可带量化/ReLU 后处理，效应级视为拷贝）。",
)

spec.op(
    "hivm.hir.nd2nz",
    params={"src": In("buf"), "dst": Out("buf")},
    effects=(wr("dst"),),
    pipe="PIPE_MTE2",
    value="copy_with_layout_transform(src, into=dst)",
    doc="GM→L1/cbuf 装载并做 ND→NZ 分形布局变换（DMA 类，PIPE_MTE2）。",
)

spec.op(
    "hivm.hir.copy",
    params={"src": In("buf"), "dst": Out("buf")},
    effects=(wr("dst"),),
    # ODS：SinglePipeOpTrait 但无默认 pipe——逐实例标注。留空并在 M2 前从
    # 主仓用法补齐，不猜（效应级不受影响）。
    pipe="",
    value="copy(src, into=dst)",
    doc="通用拷贝（DMA 类）。pipe 逐实例标注，描述层暂留空。",
)

spec.op(
    "hivm.hir.vexp",
    params={"src": In("buf"), "out": Out("buf")},
    effects=(wr("out"),),
    pipe="PIPE_V",
    value="elementwise(exp, src, into=out)",
    doc="逐元素指数（向量核，PIPE_V）。",
)

spec.op(
    "hivm.hir.vcast",
    params={"src": In("buf"), "out": Out("buf")},
    effects=(wr("out"),),
    pipe="PIPE_V",
    value="elementwise(cast, src, into=out)",
    doc="逐元素类型转换（向量核，PIPE_V）。",
)

spec.op(
    "hivm.hir.vbrc",
    params={"src": In("scalar"), "out": Out("buf")},
    effects=(wr("out"),),
    pipe="PIPE_V",
    value="broadcast(src, into=out)",
    doc="标量广播到向量（向量核，PIPE_V）。",
)

spec.op(
    "hivm.hir.set_atomic",
    params={},
    effects=(),
    # 值语义即"改模式"：静态检查拒绝零效应且零值语义的空描述（FR4），
    # 而本 op 恰有真实语义——改变后续 GM 写的原子性，必须声明而非留空。
    value="set_mode(atomic_kind, applies_to=subsequent_global_writes, reset_by=NONE)",
    doc=(
        "设置后续写 GM 操作的原子模式（kind=NONE 复位）。它改的是模式寄存器"
        "而非任何 buffer，故无内存效应；对 M2 而言它是影响写语义的状态切换，"
        "届时按状态效应建模。"
    ),
)

spec.op(
    "hivm.hir.debug",
    params={"arg": In("buf")},
    effects=(rd("arg"),),
    doc=(
        "设备端调试打印。ODS 声明 MemoryEffects<[MemRead, MemWrite]>；效应级"
        "只建模读（读延长 arg 的生存期，占用上保守），MemWrite 属调试通道，"
        "不作用于 arg 本身。"
    ),
)

# --- checks 段 --------------------------------------------------------------

spec.check("ub_occupancy", spaces=["ub", "cbuf"])
