"""矩阵乘输入来源完整性检查（`operand_wiring`，M5）。

## 它补的是哪个洞（实测，不是假设）

dcv 前端 pass 的一个真实缺陷：V/C 拆分时 `tl.dot` 的 B 操作数（循环内 `tl.load`
的非携带值）整条跨核通路丢失，CUBE 侧 matmul 的 B 被一个**本地新建的零张量**
顶替（且同一张量同时占据 B 与 outs 两个槽位）→ 每轮点积恒为 0。

该 IR 上既有六项检查**全部给不出结论**（实测见 `docs/tasks/M5.md` §5）：

- `ub_occupancy` / `timeline` / `equivalence` → `COVERAGE_GAP`；
- `uninit_read` → **OK（漏报）**：失败形态是"被零填充顶替"，不是读未初始化；
  且手动制造真·读未写后仍报 OK——链路上的未建模 op 被「保守视为已写入」吞掉；
- `sync_pairing` → OK（正确：它不是配对缺陷）。

`equivalence` 为何也不行：值语义链在 `mmadL1`（逃生舱）/`linalg.matmul`
（社区方言不建 VNode）处断链，且输入是 `?` 动态维推不出具体值。**结论：这一类
不能只靠值等价覆盖。**

本模块换一个问法，它对 layout 代数**免疫**（不需要算分形布局、不需要执行）：

> **这个矩阵乘的输入槽位，有没有真实的数据来源？**

## 两条规则（都可证，不猜）

- **R1 `wiring/neutralized-input`**：DPS 输入槽位是**编译期零**（`linalg.fill`
  零 / `arith.constant` 零 splat）→ 该乘法的贡献恒为 0，这个操作数不含任何
  数据依赖。
- **R2 `wiring/dataless-input`**：DPS 输入槽位经**纯视图链**（cast/view/
  `to_tensor`…）到达 `tensor.empty` → 读的是"内容未定义"的张量。

## 精度取舍（FR2：假阳性是第一压制目标）

- **只看 DPS 输入槽位**，不看 init/累加器槽位：零累加器是常态，零输入才可疑。
  这是本检查与"见到 fill 就报警"的朴素做法拉开差距的地方。
- **只认张量/缓冲型的零**，标量零不入判定（`real_k = 0` 合法）。
- **R2 只认 `tensor.empty`**：alloc 的写者判定需要跨核配对（V→C 成对 cbuf 是
  两个 SSA、由 transfer 标注配对）模型，不在本程范围内（M5 卡 §4）。故 R2 在
  含未建模搬运链的 IR 上可能不发声——**这必须留痕**（见报告 notes），
  不得把"没报"读成"没问题"。
- R1 命中时报告**必须同时给出两种解释与消除办法**（源码确实乘零时该怎么做），
  因为"输入为零"本身在语义上是可满足的退化写法。

## 与值语义的分工（D6）

只消费 `VModule`（尤其是 `VModule.structure` 的结构事实），不访问 MLIR、
不展开循环、不执行，故可离线单测，也不受 `timeline` 展开界与动态 shape 影响。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hivm_spec.vir import Loc, VModule

__all__ = [
    "DATALESS_INPUT_RULE",
    "MATMUL_INPUT_RULE",
    "WIRING_ENGINE_VERSION",
    "WiringFinding",
    "WiringReport",
    "analyze_operand_wiring",
]

#: 引擎版本——结论里回显，便于把历史结论对应到当时的判定口径。
WIRING_ENGINE_VERSION = "0.1.0"

#: R1：输入槽位是编译期零。
MATMUL_INPUT_RULE = "wiring/neutralized-input"
#: R2：输入槽位来自 `tensor.empty`（内容未定义）。
DATALESS_INPUT_RULE = "wiring/dataless-input"

#: 纯传递性 op：不产生数据、也不写入数据，只是换个名字/布局视图。
#: 沿这些 op 可以继续往上追"数据从哪来"。**不在表内的一律停下**——
#: 停下意味着"这个操作数有生产者"，不会误报（保守方向）。
PURE_VIEW_OPS = frozenset(
    {
        "bufferization.to_tensor",
        "memref.cast",
        "memref.subview",
        "memref.reinterpret_cast",
        "memref.collapse_shape",
        "memref.expand_shape",
        "memref.memory_space_cast",
        "memref.view",
        "memref.assume_alignment",
        "tensor.cast",
        "tensor.expand_shape",
        "tensor.collapse_shape",
        "tensor.extract_slice",
    }
)

#: 内容未定义：整条纯视图链追到它就说明"没人给过它数据"。
EMPTY_TENSOR_OP = "tensor.empty"

#: 槽位标签：DPS 输入按 A/B 命名（matmul 家族的前两个输入）。
_SLOT_LABELS = ("A", "B")


def _slot_label(index: int) -> str:
    if index < len(_SLOT_LABELS):
        return _SLOT_LABELS[index]
    return f"输入#{index}"


@dataclass(frozen=True, slots=True)
class WiringFinding:
    """一条输入来源缺陷。"""

    rule: str
    op: str
    slot: int
    loc: Loc
    detail: str
    #: SSA 文本（定位用；报告渲染时可截断）
    value: str = ""
    seq: int = 0
    loop_path: tuple[int, ...] = ()

    @property
    def label(self) -> str:
        if not self.loop_path:
            return self.op
        return f"{self.op}@i{'.'.join(str(i) for i in self.loop_path)}"

    def describe(self, *, with_loc: bool = True) -> str:
        """`with_loc=False` 供 Finding 使用——渲染层已附加位置，避免重复。"""
        where = f" @ {self.loc.describe()}" if with_loc else ""
        return f"{self.label} 的 {_slot_label(self.slot)} 槽位{self.detail}{where}"


@dataclass(frozen=True, slots=True)
class WiringReport:
    findings: tuple[WiringFinding, ...] = ()
    #: 参与判定的矩阵乘族 op 数（分母：为 0 时"没发现问题"毫无意义）
    matmul_ops: int = 0
    #: 结构事实条数（为 0 说明引擎未收集，检查未实际发生）
    structure_facts: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def vacuous(self) -> bool:
        """检查是否**未实际发生**（无矩阵乘可判、或结构事实缺失）。"""
        return self.matmul_ops == 0 or self.structure_facts == 0


def _dataless_chain(ssa: str, module: VModule) -> tuple[str, ...] | None:
    """沿纯视图链回溯；追到 `tensor.empty` 时返回途经的 op 名，否则返回 None。"""
    st = module.structure
    chain: list[str] = []
    seen: set[str] = set()
    cur = ssa
    while True:
        if cur in seen:
            return None
        seen.add(cur)
        def_op = st.def_of(cur)
        if not def_op:
            # 函数参数/块参数：数据来自调用方，不是"没有数据"
            return None
        if def_op == EMPTY_TENSOR_OP:
            return tuple(chain)
        if def_op not in PURE_VIEW_OPS:
            # 有真实生产者（copy/fill/load/matmul/未知 op）→ 停，不猜
            return None
        operands = st.operands.get(cur, ())
        if not operands:
            return None
        chain.append(def_op)
        cur = operands[0]


def analyze_operand_wiring(
    module: VModule, config: dict[str, Any] | None = None, *, bound: int | None = None
) -> WiringReport:
    """矩阵乘输入来源完整性分析。

    只消费 `VModule.structure`（结构事实），**不遍历节点流、不展开循环、不执行**：
    判定对 layout 代数免疫，故也不需要 bindings 之外的值语义支持、不受
    `timeline` 展开界影响。`bound` 形参保留只为与其它检查的装配签名一致。
    """
    del config, bound  # 判定不需要描述库与展开界：这一条是"结构可证"的

    structure = module.structure
    findings: list[WiringFinding] = []

    for site in structure.matmul_sites:
        for slot in range(min(site.dps_input_count, len(site.operands))):
            ssa = site.operands[slot]
            if structure.is_zero(ssa):
                findings.append(
                    WiringFinding(
                        rule=MATMUL_INPUT_RULE,
                        op=site.op,
                        slot=slot,
                        loc=site.loc,
                        detail=(
                            "是一个编译期零张量（本块内新建）：该输入不含任何数据"
                            "依赖，此乘法的贡献恒为 0。常见成因是跨核拆分时该操作数"
                            "的传输通路丢失、被兜底中和；若源码确实显式乘零，请在该"
                            " op 上显式标注以消除本告警"
                        ),
                        value=ssa,
                    )
                )
                continue
            chain = _dataless_chain(ssa, module)
            if chain is not None:
                findings.append(
                    WiringFinding(
                        rule=DATALESS_INPUT_RULE,
                        op=site.op,
                        slot=slot,
                        loc=site.loc,
                        detail=(
                            "来自 `tensor.empty`（内容未定义）："
                            f"途经 {' → '.join(chain) or '直接'}，链上无任何写者"
                        ),
                        value=ssa,
                    )
                )

    notes: list[str] = []
    if not structure.matmul_sites:
        notes.append("模块内没有矩阵乘族 op——本检查未实际发生，不得据此判定无输入来源缺陷")
    if not structure.defs:
        notes.append("结构事实为空（引擎未收集 defs）——本检查未实际发生")
    notes.append(
        "口径：只看 DPS **输入**槽位（零累加器合法，故 init 槽位不判）；"
        "只认编译期零与 `tensor.empty` 两种形态；buffer 写者判定需跨核配对模型，"
        "不在本检查范围（见 M5 卡 §4）"
    )

    return WiringReport(
        findings=tuple(findings),
        matmul_ops=len(structure.matmul_sites),
        structure_facts=len(structure.defs),
        notes=tuple(notes),
    )
