# 写描述：扩展语义覆盖的正确路径

**立命之本**：MLIR 千变万化，正确性类别有限。所以"工具支持的 op 不够"时的默认动作是
**写描述**（`specs/*.py`），而不是改引擎、也不是放弃验证。

实测证据：往描述里加一个此前完全不存在的 `hivm.hir.vdiv`，只写
`value="elementwise(div, a, b, into=out)"` 一行，`git status src/` 全空——占用 / 时序 /
等价三个工具同时获得对它的支持。核心引擎里硬编码的 op 名是 **0** 个。

## 决策树

```
新 op 的值语义能用现有原语表达？
├─ 能  → 只写描述。引擎一行都不要动。                      ← 绝大多数情况
└─ 不能 → ① 逃生舱 host_fn（trust 封顶 provisional）        ← 布局代数这类
          ② 新原语（须双模 + 必须是语义类别）                ← 少数，谨慎
```

## 现网原语集（封闭集合）

`VALUE_PRIMITIVES = {copy, elementwise}`

- `copy(src, into=dst)`
- `elementwise(fn, a, b, into=dst)`，二元 `fn ∈ {add, sub, mul, div}`
- `elementwise(fn, src)`，一元 `fn ∈ {neg, exp}`

一个 `elementwise` 覆盖全部 vadd/vsub/vmul/vdiv/vexp/vcast 之类。**写描述时先想"这是不是
逐元素运算"**，是就一行搞定。

写引擎不认识的原语会**直接报错**而非静默放行——允许描述凭空声明语义，等于允许臆造语义。

**新原语的三门槛**（缺一不得合入）：
1. `values.py` 提供**双模**实现（具体 + 符号），不可只做具体档；
2. 进 `VALUE_PRIMITIVES` 并补解析器分支；
3. 它是一个**语义类别**而非某 op 特例——名字该是 `broadcast`/`reduce`/`cast` 这种跨 op
   复用的词。若只为一个 op 服务，那多半该走逃生舱。

## 描述骨架

```python
from hivm_spec.spec import In, Out, Spec, cond_wr, host_fn, rd, wr

spec = Spec(name="hivm_cv", arch="a3")

# vm 段：状态模型
spec.space("gm", capacity=None)
spec.space("ub", capacity=192 * 1024, align=32)
spec.space("cbuf", capacity=512 * 1024, align=32)
spec.pipe("PIPE_V", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_FIX", "PIPE_S")
spec.event(*(f"EVENT_ID{i}" for i in range(8)))

# ops 段：值语义 + 效应
spec.op(
    "hivm.hir.fixpipe",
    params={"src": In("buf"), "dst": Out("buf")},
    effects=(wr("dst"),),
    pipe="PIPE_FIX",
    value="copy(src, into=dst)",
    doc="L0C → OUT/L1/UB 搬运（效应级视为拷贝）。",
)

# checks 段：要生成哪些工具 + 判定口径
spec.check("ub_occupancy", spaces=["ub", "cbuf"])
spec.check("timeline", scheduling="conservative", strategies=4, unroll_bound=16)
spec.check("equivalence", rtol=1e-5, atol=1e-8, round_mode="rint")
spec.check("sync_pairing")
spec.check("uninit_read")
spec.check("operand_wiring")
```

DSL 词汇表：`In/Out/Attr/Variadic` · `rd/wr/cond_wr` · `sync_set/sync_wait/sync_barrier` ·
`host_fn` · `space/pipe/event/check/assume`。**清单是封闭的**（框架 §4.4）：不要指望加新
构造，能力扩展只有"新原语"和"逃生舱"两条轴。

## 效应建模口径（D3）

- 只声明"动了哪个参数"，方向用 `In`/`Out` + `rd`/`wr`/`cond_wr`。
- `cond_wr("out", when="init_condition")` 表达"条件写"——用 `wr` 表达会丢掉分支语义。
- `pipe=` 可多 pipe（如 mmadL1 同时占 `PIPE_MTE1` 与 `PIPE_M`）。
- **社区方言（scf/arith/memref/tensor）语义不进描述**，由引擎内置。别去描述 `arith.addf`。

## 依据来源必须是 ODS 真值，不是 IR 反推

描述的 op 语义层来源应是主仓 tablegen：
`bishengir/include/bishengir/Dialect/HIVM/IR/{HIVMMacroOps,HIVMDMAOps,HIVMVectorOps,HIVMOps}.td`。

**不要从 IR 用例反推语义**——用例是 pass 作者写的最小 fixture，其形态（tensor 化、无地址
空间标注）不代表 op 语义。每条 `spec.op` 的注释里写明依据（哪个 .td、哪条 trait）。

## 逃生舱（`host_fn`）

```python
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
    effects=(cond_wr("out", when="init_condition"),),
    pipe="PIPE_M",
    value=host_fn(
        "mmad_l1_value", reason="MMA 值语义依赖分形布局代数（OpLayoutInterface），无法声明式表达"
    ),
    doc="L0C 条件清零 + L1 矩阵乘。值语义走逃生舱。",
)
```

- `reason` 必填且要写清**为什么无法声明式表达**。
- 逃生舱条目 trust **封顶 `provisional`，永不可升级**。
- 它及其**下游**会被如实报为 `COVERAGE_GAP`。这是正确行为，不是缺陷：猜一个布局代数写进去
  才是真风险。
- 后果要有预期：一个 `mmadL1` 断链可让整批语料失去可比性。

## 语义假设必须登记（不许只写在散文里）

按某个方向猜了语义、又没和主仓 C++ 链对拍 → 用 `spec.assume()` 登记在**描述里**：

```python
spec.assume(
    "hivm.hir.copy.pipe",
    "copy 无默认 pipe，按 SinglePipeOpTrait 归 PIPE_S",
    risk_direction="false-negative",
    resolve_by="与 GraphSyncSolver 对拍",
)
```

`risk_direction ∈ {false-positive, false-negative, both}`。登记后信任自动封顶
`provisional`（`frozen_by_assumption`），对拍销案后才解除。写在任务卡散文中不算——机器读
不到，结论里也不会声明。

## 语义漂移（D9）处置

描述与主仓 C++ 链行为不一致时：

1. 先登记 drift ledger 条目（该 op 的 trust 升级随即冻结）；
2. **默认权威 = 主仓 C++ 链** → 默认动作是**修描述**；
3. 仅**硬件 golden** 可推翻该默认（条目标 `upstream-suspected`）；
4. 条目未闭环期间，涉及该 op 的结论强制降级标注。

**禁止**为让工具"跑通"而臆造语义。语义不确定时的正确动作是**留空 → 触发
`COVERAGE_GAP`**。缺口是合法结论，静默不是。

## 改描述的配套义务（OD4/OD11）

同一 PR 内必须附对拍用例（`specs/cases/`）：① canonical 手工小例；② 与主仓 C++ 链对拍；
③ 性质测试。信任升级必须附对拍证据，否则 spec-gate 直接拒。

```bash
hivm-spec check specs/toy.py specs/cv.py      # 生成前置门：静态检查描述
hivm-spec gen specs/toy.py specs/cv.py -o build/config.json
pytest specs/cases -q                          # 改 specs/ 后必跑
python -m pytest tests/test_properties_tripwire.py -q   # 注册表覆盖绊线
```

静态检查会拒绝：未声明的 space/pipe/event、参数方向与效应矛盾、非法 check 名
（`KNOWN_CHECKS` 封闭）、逃生舱缺 reason、原语不在封闭集合。

**能力指标**：不要用"建模了几个 op / 方言共几个 op"衡量（那比值无意义）。正确指标是
**原语覆盖率** + **在此原语集下描述已覆盖多少目标语料**。汇报时用后者。
