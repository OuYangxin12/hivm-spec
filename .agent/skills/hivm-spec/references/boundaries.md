# 能力边界：引用工具结论前必读

这些**不是 bug**，是设计后果。不知道它们就引用结论，等于误用工具。

## 1. 信任级别全部是 `provisional`

描述库尚未与主仓 C++ 链系统对拍。每条结论都带此脚注。

- ✅ 适合：**开发回路自检**（写 pass 时快速拿 success/fail 与状态视图）
- ❌ 不适合：**合入门禁拦截**——需先补对拍证据升级信任级

把 provisional 结论当门禁依据，等于把"我的模型说它对"当成"它对"。

## 2. 等价验证只在少数语料上可判

实测快照（41 份自带语料，2026-09）：**equivalence 仅 5/41 可判**，36 份报
`COVERAGE_GAP`。主因：

1. **逃生舱 op 断链**（`mmadL1` 等）：分形布局代数无法声明式表达 → 值链断裂 → 下游整批
   op 失去可比性（一条 `fixpipe→load→vexp` 整链即失效）；
2. **动态 shape 不猜具体值**：`1x?xf32` 之类无法推导输入 →
   `equivalence/undeducible-input`；
3. **未建模 op** 的结果不参与对拍，报告明写"不得据此判定等价"。

**符号档没有绕开这个限制**。它能多判的是结构等价层（未解释函数两侧一致即可），但值链一旦
断裂，后续步骤根本没有值可比。

## 3. 有 op 未建模

语料中出现但未建模（快照）：`set_mask_norm` / `varange` / `gather_load` / `matmul` /
`nz2nd` / `scatter_store` / `custom_macro`。op 出现次数口径下覆盖率约 94%。

未建模的可见性有两处：描述侧 `python -m pytest tests/test_properties_tripwire.py`
（账本绊线，报告未注册/未建模 op），运行侧 `vir.Coverage`（分析期）。
**覆盖率低不是错误，"覆盖率无人知晓"才是。**

## 4. 符号档不覆盖浮点精度

符号档用 **Real 语义（无限精度有理数）**：

- `(a+b)+c == a+(b+c)` 在符号档**等价**，在 f32 下因结合律缺失**不一定**等价；
- 数值等价（含舍入、bf16/fp8 精度）归**具体档**负责，其容差由描述
  `spec.check("equivalence", rtol=1e-5, atol=1e-8, round_mode="rint")` 决定。

两档结论**并列、不可互替**。缺 z3 时不得用具体档结论冒充符号档。

## 5. `sync_pairing` 是账目级

计数配平不蕴含无死锁。`balanced_misordered_same_lane.mlir`：16 wait : 16 set 配平但同泳道
错序 → `sync_pairing` OK，`timeline` `DEADLOCK`。这类形态 FileCheck 必然漏过。**顺序判断只
能靠 `timeline`。**

## 6. vacuous 缺口很常见，别当绿灯

- `timeline/vacuous`：IR 中无显式同步结构（L2 dump 语料普遍如此，管线语义在注解层）→
  "不代表『无死锁风险』，而代表『未能分析』"。
- `uninit/vacuous`：无本地分配缓冲；且**函数参数视为已初始化**（调用方责任），本检查只覆盖
  本地分配。
- `pairing/vacuous`：模块内无可配对事件。
- `wiring/vacuous`：模块内没有矩阵乘族 op（或引擎未收集到结构事实）。

这些 vacuous 的共同文案都是"本检查未实际发生，不得据此判定……"。

## 7. `operand_wiring` 只判两类可证形态

它判的是"矩阵乘的**输入**槽位有没有数据来源"，因此：

- **不判 init/累加器槽位**：零累加器是常态（`outs(零)`）。只有零出现在 **输入**槽位才是信号；
- 只认两类**可证**形态：编译期零张量、经纯视图链到达 `tensor.empty`；
- **buffer 写者判定尚未实现**（"这块 cbuf 没人写过"）：V→C 成对 cbuf 是两个不同 SSA、
  由 transfer 标注配对，朴素写者判定会在**正确**的 IR 上误报。故这一半留作缺口；
- 因此它**不覆盖**：形状/布局错、同步错、数值精度、动态 shape 相关的语义错。
  它的价值是"在 `equivalence` 因逃生舱断链而报缺口时，仍能对一类结构缺陷发声"。

## 8. 容量与对齐是占位基线

描述里 `spec.space("ub", capacity=192*1024, align=32)` / `cbuf 512*1024` 是 A3
**provisional 占位基线**，未与硬件对拍。据此判 `OVERFLOW` 时，容量本身也是待验假设。

## 9. 覆盖语料是分层引入的，不是全量

L0 手写 12 / L1 干净 UT 10 / L2 剥离的 e2e 19（D13）。**不代表真实 pipeline 的全量覆盖
率**。L1 门槛是"整份语料可无 `COVERAGE_GAP` 转 VIR"，达不到就留在 L2，不放宽门槛。

## 10. 绑定不可打包（这条决定 skill 的可安装边界）

六类检查依赖的 bishengir bindings **不是一份能装走的依赖**：

| 事实 | 实测值 |
|---|---|
| 来源 | 主仓**构建树**产物（`build/tools/bishengir/…/python_packages/bishengir/`） |
| ABI | 硬绑 cp310（换 Python 版本即 import 失败） |
| 体积 | 约 246M，21 个 `.so` |
| `.so` RUNPATH | 含构建机**绝对路径**（`$ORIGIN:` 回退可用，但不可跨机搬移） |
| 树内含该绝对路径的文件 | 84 个 |
| 上游 wheel | 未发布（PyPI 无 `bishengir`） |

所以 skill 的真实形态是：**核心层可 pip 安装；六类功能取决于本机是否已有 bindings**。
`scripts/bootstrap.py` 负责在本机找到它（真 import 校验），`verify` 在找不到时退出
码 2 并如实报告能力边界。**不要**对外承诺"装完 skill 就能验全部"。

`bishengir` 还是**命名空间包**（`__file__ is None`，目录下没有 `__init__.py`）——
按常规包布局写判别文件会把唯一正确的树判成"不是绑定树"。

## 11. 契约边界（工具不会做的事）

- 不吃 pass 序列、不解析 `--mlir-print-ir-after-all`、不建模 pass 顺序、不跨调用保状态
  （D12）；
- 不主动上游化到 AscendNPU-IR，本项目是纯下游使用者；
- 不做 lit/FileCheck 的结构性断言（那是主仓 lit 的职责），也不替代主仓 verifier 的
  `expected-error` 负例。

## 复现这些数字

覆盖率/可判率是特定语料快照上的实测值，会随语料与描述演进变化。复现命令见
`docs/tasks/M4.md` §8，或跑：

```bash
scripts/verify-skill.sh          # 本 skill 自带：重跑本文件引用的实测样本
```

**汇报能力时用实测值 + 快照日期，不要引用本文件里的静态数字。**
