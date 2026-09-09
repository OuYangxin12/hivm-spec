# AGENTS.md — hivm-spec agent 工作规约

本仓库的主要贡献者是 AI agent。本文件是**强制约束**，不是建议：其中的门禁与不变量
直接承载 FR6（防自欺）与 FR7（元可信）——本项目相对 lit/FileCheck 的核心价值。

> 决策依据：`docs/design-framework.md` §3（D1–D11）。冲突时以该文档为准。

---

## 1. 提交前必跑（spec-gate，D5/OD5）

**改动任何代码或描述后，提交前必须依次跑完以下命令并全绿：**

```bash
ruff check . && ruff format --check . && mypy src
pytest -m "not requires_bindings" -q
python scripts/spec_gate.py --base origin/main --head HEAD
```

**改动 `specs/` 下的描述时，额外必跑对拍集：**

```bash
pytest specs/cases -q
```

**本地 merge 到 main（未推送、CI 跑不到）时，必须先跑本地合入门禁：**

```bash
python scripts/merge_gate.py          # 全绿方可 merge，证据落 build/merge-gate.json
```

远端分支保护与 CI 在本地合入路径上结构性失效（feature 分支没上过远端）。
该脚本跑 CI 门禁的本地等价物并留下可审计证据；工作区不干净即拒绝。
详见 milestone-plan §8「本地合入例外」。

**改 HIVM pass（主仓侧）后，须跑对应 spec 工具**（M1 起可用）：

```bash
hivm-spec tool ub_occupancy <after.mlir>   # M1（单输入，D12；溢出是 after 的内在性质，无需 before 锚点）
hivm-spec tool timeline <after.mlir>       # M2
```

失败即停止并修复，**不得**通过放松断言、改期望值、加 skip 或降低门禁使其变绿。

## 2. 描述治理纪律（OD4/OD11 — 信任根）

`specs/` 是验证结论的判定依据。修改它等于修改"什么算正确"，因此：

- **新增/修改 op 描述**必须在同一 PR 内附对拍用例（`specs/cases/`）：
  ① canonical 手工小例；② 与主仓 C++ 链对拍；③ Hypothesis 性质测试。
- **trust 升级**（`provisional` → `cross-validated` → `anchored`）**必须附对拍证据**，
  否则 spec-gate R2 直接拒绝合入。`anchored` 需硬件 golden（流程见框架 §8.1）。
- **逃生舱条目**（注册 host Python 函数，OD8）信任**封顶 `provisional`**，不可升级；
  必须显式声明效应并配性质测试。
- **禁止**为让工具"跑通"而在描述中臆造语义。语义不确定时的正确动作是
  **留空 → 触发 `COVERAGE_GAP`**，而非猜测。缺口是合法结论，静默不是。

## 3. 语义漂移处置（D9）

描述与主仓 C++ 链行为不一致时，**不得自行选择便利解释**：

1. 先登记 drift ledger 条目（该 op 的 trust 升级随即冻结）；
2. **默认权威 = 主仓 C++ 链** → 默认动作是修描述；
3. 仅**硬件 golden** 可推翻上述默认（条目标 `upstream-suspected`）；
4. 条目未闭环期间，涉及该 op 的结论强制降级标注。

## 4. 架构不可违约束

| 约束 | 依据 | 违反后果 |
|---|---|---|
| **只有 IR 接口引擎可 import `bishengir`**；核心层保持纯 Python | D7 | 核心层测试将被稀缺绑定环境阻塞，CI 覆盖崩塌 |
| **所有引擎只消费 VIR**，不得各自遍历 MLIR | D6 §6.1 | 长出第二个解释核，"双源真理"在项目内部重演 |
| **VIR 构造后不可变**；节点顺序确定性；每节点可回溯源位置；未识别结构必入 `coverage` | §6.1 不变量 | 破坏 FR8 确定性与 FR5 可诊断性 |
| **社区方言（scf/arith/memref/tensor）语义不进描述**，由引擎内置 | §4.2 | 描述范围失控 |
| **DSL 概念清单封闭**：§4.2 所列之外的表达能力一律不加 | §4.4 | DSL 范围蔓延 |
| **verdict 为封闭枚举**；`COVERAGE_GAP`/`UNTRUSTED_DESCRIPTION` 是合法结论 | §7.3 | 静默误判，元可信性丧失 |
| **不主动上游化**到 AscendNPU-IR；本项目是纯下游使用者 | milestone §8 | 越界修改主仓 |
| **工具输入恒为单份 IR**（等价验证为待验+锚点两份）：不吃 pass 序列、不解析 `--print-ir-*` 转储、不建模 pass 顺序、不跨调用保持状态 | D12 | 契约随主仓 pipeline 膨胀（冲击 NFR4）；重复 MLIR 与 agent 已有能力 |

## 5. 未实现阶段的诚实报告（FR7）

尚未实现的能力**必须**报 `PENDING(<任务号>)` 并以可区分的方式退出
（CLI 退出码 3；pytest 显式 skip 带任务号），**禁止**：

- 返回假的成功结论；
- 静默跳过检查；
- 把"环境不可用"报告成"验证失败"（二者必须区分——这正是 FR7 的要求）。

## 6. 跨 pass 定位由你编排（D12）

工具只回答"这一份 IR 有没有这个毛病"。定位"哪个 pass 引入了毛病"是**你的工作**，
用既有能力组合即可——不要为此扩展工具契约：

```bash
# 1. 转储各 pass 的 IR（MLIR 标准能力）
bishengir-opt ... --mlir-print-ir-after-all 2> dumps.txt   # 或 --print-ir-after=<pass>

# 2. 按 pass 切分后，对每份分别调用工具；第一个异常 verdict 即首恶 pass
for f in dump_*.mlir; do hivm-spec tool timeline "$f"; done
```

同理，三类检查各自独立调用、结论正交：

```bash
hivm-spec tool timeline      after.mlir                 # 死锁/时序：只看 after
hivm-spec tool ub_occupancy  after.mlir                 # 片上内存溢出：只看 after
hivm-spec tool equivalence   before.mlir after.mlir     # 语义等效：before 为锚点
```

## 6.5 pass 变更触发工具验证（T1.8 spec-gate 首版）

**规则**：凡修改 HIVM pass 实现代码（`bishengir/**` 下的 pass/transform 源文件），
必须对**受影响 kernel 跑对应 spec 工具并把 verdict 贴进 PR 描述**。机器上可机械
校验的部分（hivm-spec 仓库 CI）只校验"描述与语料的一致性"；pass 侧行为是否被
工具复验，由本条纪律约束——因为工具的价值随"pass 改了却没人验"归零。

最小执行集（按改动性质选取，拿不准就全跑）：

| 改动性质 | 必跑 | 判据 |
|---|---|---|
| 内存/分配类（plan-memory、multi-buffer、workspace） | `ub_occupancy` 全部 L2 目标 kernel | verdict 不得从 OK/OVERFLOW 恶化为新增 OVERFLOW |
| 调度/同步类（cv-pipelining、sync-solver、preload） | `timeline`（M2 起可用；此前记录"PENDING(timeline)"） | 死锁/时序 verdict 变化必须解释 |
| op 定义/ODS 变更 | `hivm-spec check` 全部描述 + `registry` 绊线重测 | 覆盖率变化需在 PR 中说明 |

判据基线是**变更前的 verdict**：先在改动前的 commit 跑一遍留存，再在改动后跑，
两者对照贴进 PR。只贴"改后绿灯"不贴基线，等于没有验证。

## 7. 里程碑任务卡（强制）

**每个 Mx 阶段开工前，必须先写好任务卡并落盘上传**（`docs/tasks/M<n>.md`），内容至少含：
执行顺序与依赖图、任务清单（带完成信号与勾选框）、验收命令、风险与止损、需上报决策的情形。

- 计划文档（`milestone-plan.md`）定**做什么**；任务卡定**按什么顺序做、每步完成信号是什么**。
- 任务卡随该里程碑的 PR 持续更新勾选状态，**不允许**开工后才补卡。
- 里程碑收尾时：更新卡内状态 → 同步 `milestone-plan.md` 勾选与框架 §9 → 写下一里程碑的卡。

## 8. 测试语料纪律（D13）

- 语料**分层引入**：L0 手写微例 → L1 主仓干净 UT → L2 按需剥离 → L3 e2e dump；不预先囤积。
- 语料**入库**并在 `specs/cases/corpus/manifest.json` 记录来源文件 + 主仓 commit + 剥离方式。
  缺 manifest 记录即视为来源不明。
- **L1 门槛**：整份语料须可无 `COVERAGE_GAP` 转为 VIR；达不到就留在 L2，**不放宽门槛**。
- **禁止**整批镜像主仓测试目录（189 文件中仅 21 个干净）——会把 `COVERAGE_GAP` 变成噪声，
  使唯一的诚实降级信号失效；也会让 VIR 契约被 lit 夹具细节污染。
- **不引入**负例（`expected-error`）作为解析语料：那是主仓 verifier 的职责。

## 9. 工程约定

- **PR 流程**：main 禁直推；feature 分支 → PR → 合入。单人阶段免**人工**审批，
  但**机器门禁不可绕过**（D5）。修改 `.github/`、`scripts/spec_gate.py`、
  `scripts/merge_gate.py`、本文件需格外谨慎——能放松门禁的人等于能绕过防自欺。
  **本地 merge 亦受此约束**：走 `scripts/merge_gate.py`（§1）。
- **语义假设必须登记（D9 前置）**：按某个方向猜了语义、又没和主仓 C++ 链对拍
  的口径，一律用 `spec.assume(subject, assumed, risk_direction=..., resolve_by=...)`
  登记在描述里——不要只写在任务卡散文中（机器读不到，结论里也不声明）。登记后
  信任自动封顶 `provisional`（`frozen_by_assumption`），对拍销案后才可解除。
- **Python 版本**：核心 `>=3.10`（cp310 绑定 ABI 地板，勿放宽）；IR 层测试标
  `requires_bindings`。
- **依赖**：主依赖钉兼容区间（FR8）；环境用 uv，`~/.cache` 只读时
  `export UV_CACHE_DIR=/tmp/uvcache`；`~/.local/share/uv` 只读时
  `export UV_PYTHON_INSTALL_DIR` 须指向**持久**目录（如仓库内 `.uvpython/`，
  已 gitignore）——3.10 解释器落 /tmp 会随重启丢失并使 `.venv310` 断链
  （评审 2026-09-09 实测发生）。
- **文档联动**：里程碑退出时更新 `design-framework.md` §9 与 `milestone-plan.md`；
  新增决策进 §3（沿用 D<n> 编号），不另起文档。
- **性能预算**：per-tool 预算见 D10；里程碑退出时实测回填。
