---
name: hivm-spec
description: >-
  Verify AscendNPU-IR (HIVM / BiShengIR) compiler passes with hivm-spec — a
  verifier generator that models HIVM VM semantics and emits structured verdicts
  on MLIR. Use when you need success/fail feedback for an HIVM pass in the
  development loop, must check UB/CBUF on-chip-memory overflow, structural
  deadlock, set/wait sync pairing, uninitialized buffer reads, or IR equivalence
  before/after a transformation, or must extend semantic coverage by writing a
  description. Prefer it over lit/FileCheck for semantic correctness and over
  real-device E2E for speed. Do NOT use it to parse IR dumps, to model pass
  order, or as a merge gate (its trust level is provisional).
metadata:
  generated: "2026-09-12"
  codebase: "hivm-spec"
  codebase_commit: "db509b6"
  repo_docs: "README.md, AGENTS.md, docs/design-framework.md, docs/tasks/"
  note: "Verdict/trust numbers in references/ are measured snapshots — re-measure with scripts/verify-skill.sh before asserting them."
---

# hivm-spec：HIVM pass 的秒级语义验证

**一句话**：`hivm-spec` 不是"HIVM 验证工具集合"，而是**验证器生成器**——你用 DSL 写语义
描述，它生成吃 MLIR、吐结构化结论的工具。你面对"支持的 op 不够"时的默认动作是
**写描述**，不是改引擎，也不是放弃验证。

## 一句话入口

用户说"验一下这份 IR"时，跑这一条：

```bash
hivm-spec verify <after>.mlir --anchor <before>.mlir     # 无变换前 IR 就省掉 --anchor
```

`verify` = `doctor` 前置门 + 全套六类检查 + **能力自述**。它把"这次到底跑没跑起来、
跑了几项、哪些没跑"和结论绑成同一条输出——所以你可以直接转述它，不必自己判断环境。

```
[verify] COVERAGE_GAP｜执行 5 项，其中 3 项没验成｜信任 provisional
  ⚠️ 未提供 --anchor：等价验证未执行。本轮结论不含「变换前后是否等效」这一维度
  ...逐项 verdict + 诊断 + "没验成：…" + 本轮能力限制
```

**转述纪律**：`其中 N 项没验成` 那半句必须一起转述。只报"OVERFLOW"或只报"没有 OVERFLOW"
都是误报——前者丢了缺口，后者把缺口读成通过。

### 环境三要素（缺一即降级，`verify` 会自己说清缺哪个）

```bash
export HIVM_SPEC_BINDINGS=<bindings 树>       # 缺 → verify 退出码 2，并给修复指引
export HIVM_SPEC_CONFIG=<config.json>         # 缺 → 只能用 CWD 相对的 build/config.json
hivm-spec doctor                              # 权威能力判定；缺什么给什么 remedy
```

 bindings 不是 pip 依赖（主仓构建树产物、cp310 ABI、约 246M）——用
`scripts/bootstrap.py` 在本机**找**它，找不到就按能力边界如实汇报。

## 装到一台新机器

```bash
bash scripts/install.sh --repo <hivm-spec 本地仓>   # 或 --git git+https://…
```

它做四步：找 3.10 解释器 → 装核心层 → 生成配置文档 → 探测 bindings，最后打印**可直接用
的一句话入口**。核心层装得上不代表全功能——以它输出的能力边界为准。

## 该跑哪个检查

| 你改了什么 / 想知道什么 | 跑什么 | 输入 |
|---|---|---|
| 默认回路（推荐） | `hivm-spec verify <after>.mlir --anchor <before>.mlir` | 一份 IR；自带环境门 + 能力自述 |
| 片上内存是否溢出（UB/CBUF） | `tool ub_occupancy <after>.mlir` | 只看 after |
| 死锁 / pipe-event 时序 | `tool timeline <after>.mlir` | 只看 after |
| set/wait 配对是否完整 | `tool sync_pairing <after>.mlir` | 只看 after |
| 矩阵乘的输入有没有真实来源 | `tool operand_wiring <after>.mlir` | 只看 after |
| 是否读了未初始化缓冲 | `tool uninit_read <after>.mlir` | 只看 after |
| 变换前后是否语义等效 | `tool equivalence <after>.mlir --anchor <before>.mlir` | 两份，after 是位置参数 |

细则：[which-check-to-run.md](references/which-check-to-run.md)、
[the-six-checks.md](references/the-six-checks.md)。

## 读结论：verdict 是封闭枚举，退出码区分"验失败"与"没验成"

```
OK 0 · OVERFLOW 1 · DEADLOCK 1 · MISMATCH 1
COVERAGE_GAP 4 · UNTRUSTED_DESCRIPTION 5 · PENDING 3 · 环境不可用 2（仅 verify）
```

`2` 与 `4` 的区别是刻意的：**「我跑不起来」不等于「这份 IR 我没能验全」**。把环境故障
写成覆盖缺口，就是把它伪装成被验对象的属性。
```

**优先级**：`UNTRUSTED_DESCRIPTION` > `COVERAGE_GAP` > 具体问题 > `OK`。缺口**不算**发现
问题，也**永远不等于**通过。

```bash
hivm-spec tool timeline k.mlir; case $? in
  0) echo "验过，无问题" ;;
  1) echo "验过，真有问题 → 按 diagnostics 的 loc 修" ;;
  4|5) echo "没验成 → 补描述/补锚点/换语料，不得当作通过" ;;
  3) echo "该能力未实现 → 上报，不得静默跳过" ;;
esac
```

完整契约与 JSON 字段：[verdict-and-exit-codes.md](references/verdict-and-exit-codes.md)。

## 六条硬规则（违反即结论无效）

1. **`COVERAGE_GAP` 不是通过。** 报告"5 项里 3 项没验成"就是字面意思。禁止用"至少没报
   OVERFLOW/DEADLOCK"来读它。
2. **禁止自比。** 不给 `--anchor` 时 equivalence 报缺口而非 OK；把自己当锚点喂进去得到
   的 OK 什么都没验证。锚点 = **变换前**那份 dump。
3. **信任脚注必须转述。** 现网全部描述为 `provisional`。每条结论必须带上"未经对拍验证"
   这一限定，**禁止**用于合入门禁拦截（可用于开发回路自检）。
4. **不要试图把 pass 序列丢给工具。** 工具输入恒为单份 IR（等价验证为 after+anchor
   两份）：不吃 `--mlir-print-ir-after-all`、不建模 pass 顺序、不跨调用保状态。定位
   "哪个 pass 引入问题"由你用转储 + 逐份调用编排：[cross-pass-orchestration.md](references/cross-pass-orchestration.md)。
5. **`timeline` 与 `sync_pairing` 不可互替。** 后者是账目级（计数配平），配平不蕴含无
   死锁——同泳道错序它必然漏过，那是 `timeline` 的职责。
6. **符号档与具体档结论并列，不互相替代。** 符号档用 Real 语义（无限精度），不覆盖浮点
   精度；数值等价归具体档。缺 z3 不得"退回"具体档充数。

## 能力边界（引用工具结论前必读）

实测快照（41 份自带语料，2026-09）：equivalence 仅 5/41 可判，主因 `mmadL1` 等逃生舱
op 断链；7 个语料内 op 未建模；符号档不吃浮点精度。**这些是设计后果，不是 bug**——
一个诚实说"我看不懂这段 IR"的工具比一个蒙对的工具有用。全表：[boundaries.md](references/boundaries.md)。

## 覆盖不够时：写描述，别改引擎

```
新 op 的值语义能用现有原语表达？
├─ 能  → 只写描述（specs/*.py），引擎一行不动   ← 绝大多数
└─ 不能 → 逃生舱 host_fn（trust 封顶 provisional）  或  新原语（须双模 + 是语义类别）
```

现网原语集只有 `copy` / `elementwise`（`add|sub|mul|div` + 一元 `neg|exp`）——一个
`elementwise` 覆盖全部 vadd/vsub/vmul/vdiv/vexp。**别为单个 op 造原语。**
写法与门禁：[writing-descriptions.md](references/writing-descriptions.md)。

## 改了主仓 pass 之后（T1.8 纪律）

**基线在前、改后对照**：先在改动前的 commit 跑一遍留存 verdict，再在改动后跑，两者一并
贴进 PR 描述。只贴"改后绿灯"等于没有验证。
选哪几项跑、判据：[pass-change-verification.md](references/pass-change-verification.md)。

## 排障

`tool` 全部报缺口 / doctor 说 bindings 不可用 / 换了 Python 版本就崩——见
[environment.md](references/environment.md)（四个必错点与修法）。

## 随附脚本

```bash
scripts/install.sh --repo <hivm-spec 仓>                    # 装核心层 + 探测 bindings
scripts/bootstrap.py [--shell]                             # 只定位 bindings（真 import 校验）
scripts/locate-introducing-pass.sh timeline 'dump_*.mlir'   # 逐份跑同一检查，出 verdict 表 + 首恶候选
scripts/compare-verdicts.py base.json head.json             # 基线对照：判"是否恶化"，而非只看改后绿
scripts/verify-skill.sh                                     # 重跑本 skill 引用的实测断言（改描述后必跑）
```

脚本约定：`HIVM_SPEC` 指定命令（默认 `hivm-spec`，可用 `HIVM_SPEC="python -m hivm_spec"`），
`HIVM_SPEC_BINDINGS` 未设即拒绝运行——不静默降级。

## 仓内开发纪律不在此复制

本仓改动的门禁（spec-gate、描述治理、语料分层、架构不变量）以仓库根
[AGENTS.md](../../../AGENTS.md) 为唯一权威。**动手改 `src/` 或 `specs/` 前先读它。**
