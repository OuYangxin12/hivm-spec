# 该跑哪个检查：决策表

先定位你的处境，再决定跑什么。**拿不准就全跑**（`hivm-spec run`，几秒级）。

## 按"你在做什么"路由

| 处境 | 跑什么 | 判据 |
|---|---|---|
| 刚写完/改完一个 HIVM pass，想知道有没有弄坏 | `run after.mlir --anchor before.mlir` | 综合 verdict + 逐项；`run` 里 equivalence 若缺锚点会显式记缺口 |
| 改了内存/分配类 pass（plan-memory、multi-buffer、workspace） | `tool ub_occupancy` 全部目标 kernel | verdict 不得从 `OK/OVERFLOW` 恶化为**新增** `OVERFLOW` |
| 改了调度/同步类 pass（cv-pipelining、sync-solver、preload） | `tool timeline` + `tool sync_pairing` | 死锁/时序 verdict 变化必须解释；配平≠无死锁 |
| 改了 op 定义 / ODS（.td） | `hivm-spec check` 全部描述 + 绊线重测 | 覆盖率变化须在 PR 说明 |
| 改了前端数据流拆分（split-dataflow / cv-pipeline / multi-buffer）或怀疑操作数被弄丢 | `tool operand_wiring`（ttadapter 层即可） | 不得新增 `wiring/*` 命中；它只看**输入**槽位，init 的零合法 |
| 怀疑 kernel 读了未初始化缓冲 | `tool uninit_read` | 注意"函数参数视为已初始化"口径 |
| 想证明两个 IR 语义等效 | `tool equivalence --anchor`（具体档）**且**`--mode symbolic` 各跑一次 | 两档结论并列记录 |
| 定位"哪个 pass 引入的问题" | 转储 + 切分 + 逐份 `tool <检查>` | 第一个异常 verdict 即首恶 pass |
| 环境不确定能不能跑 | `doctor` | 有 MISSING 不可跑；仅 DEGRADED 需在结论里注明降级 |

## 按 verdict 路由下一步

| 看到 | 下一步 |
|---|---|
| `OVERFLOW` + 贡献者列表 | 从贡献者缓冲下手（缩 tiling / 换复用区间 / 调 workspace），不是盯着峰值数字 |
| `DEADLOCK` + 首现 wait | 检查该事件的 set 是否存在、是否跨迭代、是否同泳道错序 |
| `MISMATCH` + 首个发散点 | 修首因；"另有 N 步受影响"是传播后果，别去改下游 |
| `COVERAGE_GAP`（未建模 op） | 写描述（能落到原语就一行），或登记为已知缺口并如实汇报 |
| `COVERAGE_GAP`（缺 `--anchor`） | 补变换前 dump；**不要**自比 |
| `COVERAGE_GAP`（动态 shape） | 静态 shape 化语料或接受不可判；工具不会猜具体值 |
| `COVERAGE_GAP`（vacuous） | 该 IR 没有这类结构 → 这项检查对它不适用，写清楚而不是当通过 |
| `UNTRUSTED_DESCRIPTION` | 停：判定依据不可信，先修描述 / 登记 drift |
| `PENDING`(3) | 能力未实现，上报缺口，禁止静默跳过 |

## 三类检查正交，别指望一次调用覆盖全部

```bash
hivm-spec tool timeline      after.mlir
hivm-spec tool ub_occupancy  after.mlir
hivm-spec tool equivalence   after.mlir --anchor before.mlir
```

## 反模式

| 反模式 | 为什么错 |
|---|---|
| `hivm-spec run k.mlir` 后看到"没有 OVERFLOW"就说"内存没问题" | 等价/时序/配对可能是缺口；缺口≠通过 |
| 看到矩阵乘 `ins` 里是零张量就说「这是个零乘，无害」 | 零累加器合法、零**输入**不是：跨核拆分丢操作数时正是这个形态（`wiring/neutralized-input`） |
| 用 `sync_pairing` OK 替代 `timeline` | 账目级，同泳道错序必漏 |
| 同一份 IR 自比得到 OK 当作等价验证通过 | 什么都没验证，最典型的自欺形态 |
| 把 `--mode symbolic` 的等价结论当成"数值上也等价" | Real 语义，不覆盖浮点精度 |
| 只贴"改后绿灯" | 无基线等于无验证 |
| 拿 provisional 结论去拦 MR | 信任级不足，只能做开发回路自检 |
| 为让工具跑通而臆造 op 语义 | 违反 D9；正确动作是留空触发 `COVERAGE_GAP` |
