# 检查清单：查什么、怎么跑、输出怎么读

> 全部命令以 `hivm-spec` 为 `python -m hivm_spec` 的别名。位置参数**恒为一份待验 IR**。

## 目录

1. [ub_occupancy](#1-ub_occupancy片上内存占用与溢出)
2. [timeline](#2-timeline时序图与结构性死锁)
3. [sync_pairing](#3-sync_pairing同步配对账目)
4. [uninit_read](#4-uninit_read未初始化读)
5. [operand_wiring](#5-operand_wiring矩阵乘输入来源完整性)
6. [equivalence 具体档](#6-equivalence具体执行档)
7. [equivalence 符号档](#7-equivalence符号档z3)
8. [run：一次跑全套](#8-run一次跑全套)

---

## 1. `ub_occupancy`：片上内存占用与溢出

```bash
hivm-spec tool ub_occupancy after.mlir                 # 带文本占用图
hivm-spec tool ub_occupancy after.mlir --no-chart      # 只要结论
hivm-spec tool ub_occupancy after.mlir --json ub.json
```

- **单输入**：溢出是 after 的内在性质，不需要 before 锚点。
- 结论：超容量 → `OVERFLOW`(1)，否则 `OK`(0)。
- 输出给到**可动手的粒度**：空间名、峰值 vs 容量、超量、按尺寸降序的贡献者及其活跃节点
  区间。修的时候从贡献者下手，不是从峰值数字下手。

```
  ub（容量 192.0KB）
    峰值       96B [░…░] 0%
    时间线（节点 0→3，4 点）
    ▂▅█▂
    贡献者（按尺寸降序，区间=节点序号）：
      buf5(32B) 活跃 [0, 2]
      buf7(32B) 活跃 [1, 2]
```

容量来自描述的 `spec.space(...)`，A3 基线：`ub` 192KB（align 32）、`cbuf` 512KB、`gm`
无上限。**这些数值本身是 provisional 占位基线**，未与硬件对拍。

## 2. `timeline`：时序图与结构性死锁

```bash
hivm-spec tool timeline after.mlir                          # 默认全部策略
hivm-spec tool timeline after.mlir --strategy round_robin
hivm-spec tool timeline after.mlir --bound 32               # 未知 trip 循环展开界
hivm-spec tool timeline after.mlir --trace trace.json       # Chrome Trace，perfetto 可开
```

- 策略：`sequential` / `round_robin` / `pipe_priority` / `random`（展开为固定种子
  0–3）/ `all`（默认 = 三策略 + 四个固定种子）。默认用 `all`，别只跑单一策略就下结论。
- `DEADLOCK` 是**结构性**判定：论证形态是"供给 < 需求的纯计数论证"，即任何调度都不能
  放行，因此**与展开界无关**。诊断里会明写"（与展开界无关，确定性结论）"。
- 报的是**首现位置**并附受阻步计数（"共 2 个受阻 wait 步，此处为首现位置"）——修一处即可。
- `--trace` 产 Chrome Trace Event JSON，肉眼看交错比读文本图快得多。

图例：`· op　▲ set　▽ wait　≡ barrier　× 受阻 wait`

**vacuous 情形**（重要）：IR 里没有显式同步结构时，本工具报 `COVERAGE_GAP`
（`timeline/vacuous`），文案是"本结论不代表『无死锁风险』，而代表『未能分析』"。
L2 dump 语料普遍如此——管线语义在注解层而非 op 层。

## 3. `sync_pairing`：同步配对账目

```bash
hivm-spec tool sync_pairing after.mlir
```

账本级检查：set/wait 计数是否配平、有无 orphan-set、缺 set 直接报错。

**它不能替代 `timeline`**，输出里会自己声明：

```
info [pairing/note]: 本检查为账目级：计数平衡不蕴含无死锁，顺序问题请看 timeline 工具
```

计数配平但同泳道错序 → `sync_pairing` 报 OK，`timeline` 报 `DEADLOCK`。这正是
`balanced_misordered_same_lane.mlir` 这份语料存在的理由，也是 lit/FileCheck 必然漏过的
形态。**两个都要跑。**

## 4. `uninit_read`：未初始化读

```bash
hivm-spec tool uninit_read after.mlir
```

- 只看**本地分配**的缓冲；**函数参数视为已初始化**（调用方责任），这是口径不是漏洞。
- 模块内没有本地分配缓冲时 → `COVERAGE_GAP`（`uninit/vacuous`，"本检查未实际发生，
  不得据此判定无未初始化读"）。别把这种 OK/缺口读成绿灯。
- 命中时定位到代码行。

## 5. `operand_wiring`：矩阵乘输入来源完整性

```bash
hivm-spec tool operand_wiring after.mlir
```

- **单输入**：判定的是 after 的**内在结构性质**，不需要锚点。
- 判定**只看 DPS 输入槽位**（A/B）。零出现在 **init/累加器**槽位是常态，不判。
- 两类命中（都可证，不靠值语义，故对 layout 代数免疫）：
  - `wiring/neutralized-input`：输入槽位是编译期零张量（`linalg.fill` 零 /
    `arith.constant` 零 splat）——该输入没有数据依赖。**典型成因**：前端 V/C 拆分
    丢了跨核操作数传输，用零张量兜底（此时 B 与 outs 常是同一个新零张量）。
  - `wiring/dataless-input`：输入槽位经纯视图链来自 `tensor.empty`（内容未定义）。
- 结论：命中 → `MISMATCH`(1)，给到 op + 槽位 + 源码位置；没有可判定的矩阵乘 →
  `COVERAGE_GAP`(4)（vacuous，**不得**读作通过）。
- 为什么 `equivalence` 报缺口时它仍能发声：它不需要值语义（`mmadL1` 逃生舱、
  `linalg.matmul` 这类社区方言都不阻碍它），也不展开循环、不需要具体输入。
- **边界**：buffer 写者判定（这块 cbuf 没人写过）需跨核配对同一性模型，尚未实现。

---

## 6. `equivalence`：具体执行档

```bash
hivm-spec tool equivalence after.mlir --anchor before.mlir
hivm-spec tool equivalence after.mlir --anchor before.mlir --json eq.json
hivm-spec run after.mlir --anchor before.mlir        # 顺带把这套跑进全量回路
```

- **`--anchor` 是变换前的那份 IR**（通常是 pass 前的 dump）。位置参数是被验证对象。
  谁是被验证对象必须在命令行上一眼可辨——这直接决定 verdict 归属。
- **没有缺省自比**：不给 `--anchor` → `COVERAGE_GAP`。同一份 IR 自比恒为 OK 却什么都没验，
  这是最典型的自欺形态。
- 命中时给**首个发散点** + 两侧实际值 + 偏差 + 受影响步数：

  ```
  error [equivalence/first-divergence]: 首个发散：hivm.hir.vmul@i0 @ after.mlir:12：
    待验=f32[256][2.29704, …] 锚点=f32[256][-3.0312, …]（最大绝对偏差 7.93，相对 1.99）
    （另有 4 步受影响）
  ```

  **修首因，别修下游**：受影响步是传播后果。
- 常见 `COVERAGE_GAP` 触发原因：动态维 `'?'` 无法推导具体输入
  （`equivalence/undeducible-input`）、存在未建模 op（其结果未参与对拍）。
  **不猜动态 shape 的具体值**是设计选择。

## 7. `equivalence`：符号档（z3）

```bash
hivm-spec tool equivalence after.mlir --anchor before.mlir --mode symbolic
hivm-spec run after.mlir --anchor before.mlir --mode symbolic
```

- 有界内**所有输入**上找反例；命中时给反例赋值：
  `error [symbolic/counterexample]: 找到反例，首个发散在 seq1 hivm.hir.vmul：in0=-1`
- 需要 `pip install -e '.[symbolic]'`；`--mode symbolic` 且 z3 缺失时按缺口处理。
- **Real 语义（无限精度有理数）**：`(a+b)+c == a+(b+c)` 在符号档等价，在 f32 下因结合律
  缺失不一定。数值等价归具体档。
- 符号档**没有**绕开逃生舱限制：值链一旦被未解释函数（如 `mmadL1`）打断，下游根本没有
  值可比。它能多判的是"结构等价"这一层，不是万能。
- 两档结论**并列而非替代**：所以 CLI 不提供"同时跑两档"，你必须自己跑两次并分别记录。

## 8. `run`：一次跑全套

```bash
hivm-spec run kernel.mlir                                  # 无锚点：等价记为缺口
hivm-spec run after.mlir --anchor before.mlir              # 含等价
hivm-spec run kernel.mlir --mode symbolic --json run.json
hivm-spec run kernel.mlir --spec specs/toy.py specs/cv.py  # 指定描述（自动生成配置）
```

- `ub_occupancy` / `timeline` / `sync_pairing` / `uninit_read` **恒跑**；
  `equivalence` 需要 `--anchor`，缺锚点时**留一条 `COVERAGE_GAP` 记录**而不是从报告里消失
  ——否则"5 项全 OK"会被读成"这份 IR 没问题"。
- 配置文档缺省 `build/config.json`，不存在则用描述自动生成（缺省描述
  `specs/toy.py specs/cv.py`）。用 `--spec` 换描述，用 `-c` 指定既有配置。
- 综合结论聚合规则见 [verdict-and-exit-codes.md](verdict-and-exit-codes.md)。

实测一次（无锚点，L1 语料）：

```
[run] 综合结论：COVERAGE_GAP
  OK                     ub_occupancy
  COVERAGE_GAP           timeline
      error   未在 IR 中发现任何同步结构…而代表『未能分析』
  COVERAGE_GAP           sync_pairing
      warning 模块内无可配对的同步事件——本检查未实际发生
  OK                     uninit_read
  COVERAGE_GAP           equivalence
      skip    未提供 --anchor，等价验证未执行
注意：3/5 项未能完成验证。「没发现问题」不等于「没有问题」
```

**这份报告的正确读法**：占用与未初始化读验过了；时序、同步配对、等价**根本没验**。
写结论时只能写前半句。
