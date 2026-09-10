<div align="center">

# hivm-spec

**A verifier generator for AI-assisted AscendNPU-IR (HIVM) compiler development**

用 DSL 建模 HIVM 虚拟机的语义域，按需生成快速验证工具，让 AI 在 pass 开发回路中秒级获得 success/fail 与运行状态视图。

</div>

## 为什么

在 vibecoding 实践中，AI 代理承担编译器 pass 的开发与迭代，但现有验证手段存在三个缺口：

1. **语义缺位**：lit/FileCheck 是结构性验证，且 AI 可通过更新期望值将错误输出"合法化"；
2. **反馈速度**：真机 E2E 资源稀缺、周期长，无法进入开发回路；
3. **维度缺失**：计算语义等效（结构全新的 IR）、死锁、片上内存溢出（UB/CBUF）三类高频验证目标完全无覆盖。

## 快速上手

> 完整环境要求见下文[开发环境](#开发环境)。核心层是纯 Python（≥3.10）；跑 `tool`/`run`
> 还需 **Python 3.10 + bishengir bindings**（cp310 ABI）。

```bash
# 1. 安装（在 Python 3.10 环境里）
uv pip install -e ".[test]"
bash scripts/setup_bindings.sh
export HIVM_SPEC_BINDINGS="$PWD/.bindings"

# 2. 环境自检——一眼看出哪些能力就绪
hivm-spec doctor

# 3. 跑一份 IR 的全部适用检查（首次会自动从描述生成配置文档）
hivm-spec run path/to/kernel.mlir

# 做 pass 前后等价验证时，给出变换前的 IR 作为锚点
hivm-spec run after.mlir --anchor before.mlir
```

单跑某一类检查、或需要结构化输出：

```bash
hivm-spec tool uninit_read  kernel.mlir            # 只读未初始化检查
hivm-spec tool equivalence  after.mlir --anchor before.mlir --json r.json
hivm-spec run kernel.mlir --mode symbolic          # 符号档（需 pip install -e '.[symbolic]'）
```

## 能查什么

一条 `hivm-spec run` 覆盖六类检查。下表是在自带语料（41 份，2026-09 实测）上的能力：

| 检查 | 查什么 | 检出能力 |
|---|---|---|
| `ub_occupancy` | 片上内存（UB/CBUF）占用峰值是否超硬件容量 | 注入超限缺陷 → **OVERFLOW**，定位到贡献者 |
| `timeline` | pipe/event 时间线 + 结构性死锁 | 真实死锁语料 → **DEADLOCK**（4 份） |
| `sync_pairing` | set/wait 配对完整性（账本级） | 删 wait → **orphan-set** 告警；缺 set → 报错 |
| `uninit_read` | 是否读取了未初始化的片上 buffer | 删 load → **MISMATCH**，定位到代码行；语料零假阳性 |
| `equivalence`（具体档） | 两份 IR 在同一组具体输入上是否逐 op 一致 | 注入 `vadd→vmul` → **MISMATCH** 并给首个发散点 |
| `equivalence`（符号档） | 有界内**所有输入**上是否存在反例（需 z3） | 同上，并给出反例赋值 |

结论是封闭枚举：`OK` / `OVERFLOW` / `DEADLOCK` / `MISMATCH` / `COVERAGE_GAP` /
`UNTRUSTED_DESCRIPTION`。退出码区分"验证失败"（1）与"没能验成"（4=覆盖缺口、
5=描述不可信），脚本不会把"没验"误当成"验过"。

## 查不到什么（边界，务必先读）

工具**如实报告自己的覆盖边界**，这些不是 bug，但用它做判断前必须知道：

1. **信任级别全部是 `provisional`**——描述库尚未与主仓 C++ 链系统对拍，每条结论都带
   此脚注。适合**开发回路自检**；要做**合入门禁拦截**，需先补对拍证据升级信任级
   （由 spec-gate 机械强制）。
2. **等价验证只覆盖语料的 5/41**。36 份报 `COVERAGE_GAP`，主因是 `mmadL1` 等逃生舱 op
   （分形布局代数无法声明式表达）一断链，下游整批 op 失去可比性；其次是动态 shape 不猜
   具体值。符号档**没有**绕开这个限制。
3. **7 个语料中出现的 op 未建模**（`set_mask_norm`/`varange`/`gather_load`/`matmul`/
   `nz2nd`/`scatter_store`/`custom_macro`）。op 出现次数口径下覆盖率约 94%。
4. **符号档用 Real 语义（无限精度有理数），不覆盖浮点精度**：`(a+b)+c == a+(b+c)` 在
   符号档下等价，在 f32 下因结合律缺失却不一定。数值等价归具体档负责，两档结论并列、
   不可互相替代。
5. **覆盖是按 D13 分层引入的语料**（L0 手写 12 / L1 干净 UT 10 / L2 剥离的 e2e 19），
   不是真实 pipeline 的全量覆盖率。

> 覆盖率数字是特定语料快照上的实测值，会随语料与描述演进变化；复现命令见
> [docs/tasks/M4.md](docs/tasks/M4.md) §8。

## 核心机制：验证器生成器

**MLIR 千变万化，正确性类别有限。** 因此不逐个建造验证工具，而是：

```
agent 写描述（Python 内嵌 DSL）
        │  ops: op 值语义 + 效应（内存/pipe/event）
        │  vm : 状态模型（地址空间/容量/调度）
        │  checks: 要生成的工具与判定口径
        ▼
DSL 解释器 / 生成器（静态检查 → 归一化 → 配置文档 → 元编程装配）
        ▼
spec 工具（统一契约：吃 MLIR，吐结构化结论/视图）
  ├── 等价验证：两份 MLIR → success/fail + 发散定位（具体档 / 符号档）
  ├── UB 占用图：MLIR → 片上内存占用曲线与峰值
  ├── 时序图：MLIR → pipe/event 时间线（结构性死锁判定）
  ├── 同步配对：set/wait 配对完整性
  ├── 未初始化读：tainted 数据流
  └── （按需生长）
```

关键设计：**模型即数据**（描述进 git、可评审、带信任等级 provisional→crosschecked→anchored）、
**引擎少而固定**（与主仓 C++ 链 / GraphSyncSolver / PlanMemory 交叉验证）、
**缺口是合法结论**（`COVERAGE_GAP`/`UNTRUSTED_DESCRIPTION`，禁止静默误判）。

扩展语义靠**写描述**而非改引擎：新增一个 op 时，已交付的全部检查（含符号档）零引擎
改动即可用——这是本项目的立命之本，详见 [AGENTS.md](AGENTS.md) §3.5 与
[M4 复盘](docs/tasks/M4.md) §10。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/requirements.md](docs/requirements.md) | 需求基线（FR/NFR/AC 编号体系；§6 未决问题处置状态） |
| [docs/design-framework.md](docs/design-framework.md) | 方案框架与已定决策（D1–D13）、VIR 契约（§6.1）、工具输入契约（§7）、漂移流程（§8.1）、D10 性能预算 |
| [docs/industry-research.md](docs/industry-research.md) | 业界调研与决策依据（Sail/EDA/系统化并发测试/Rosette） |
| [docs/milestone-plan.md](docs/milestone-plan.md) | 里程碑执行计划（M0–M4 任务分解）与工程约定 |
| [AGENTS.md](AGENTS.md) | agent 工作规约：核心命题、spec-gate 门禁、描述治理纪律、架构不可违约束、跨 pass 定位编排 |
| [docs/tasks/](docs/tasks/) | 里程碑任务卡（执行视图：顺序、完成信号、验收命令、止损条件、实录） |

## 路线图

| 里程碑 | 交付 | 状态 |
|---|---|---|
| 前置门禁 | V1 hivm 方言解析 / V2 环境依赖 | ✅ 已通过 |
| 工程基线 | CI 五检查 + spec-gate + 双层版本策略（D5/D7） | ✅ 已落地 |
| M0 | VIR 契约 + DSL 骨架：描述→静态检查→配置文档→装配 | ✅ 已完成 |
| M1 | UB 占用图（首个 spec 工具） | ✅ 已完成 |
| M2 | 时序图 + 结构性死锁判定 | ✅ 已完成 |
| M3 | 等价验证（具体执行档）— 含不可裁剪内核（D8） | ✅ 已完成 |
| M4 | 符号档（z3）+ 未初始化读 + 同步配对 | ✅ 已完成 |

下一步方向（尚未立项）：与主仓 C++ 链系统对拍以升级信任级、扩展 `mmadL1` 等逃生舱
op 的可执行值语义、按需引入 L3 e2e dump 语料。

## 开发环境

```bash
# 核心层（py3.10+；lint/types 用系统解释器亦可）
export UV_CACHE_DIR=/tmp/uvcache
uv pip install -e ".[test,dev]"

# IR 接口层（必须 py3.10 + bindings，跑 tool/run 与 requires_bindings 测试）
# 解释器安装目录须持久——落 /tmp 会随重启丢失并使 .venv310 断链
export UV_PYTHON_INSTALL_DIR="$PWD/.uvpython"   # ~/.local/share/uv 只读时必设
uv venv .venv310 --python 3.10
uv pip install -e ".[test]" --python .venv310/bin/python
bash scripts/setup_bindings.sh                  # 绑定树落位 .bindings/（约 246M）

# 可选能力
uv pip install -e '.[symbolic]' --python .venv310/bin/python   # z3：符号档
uv pip install -e '.[numeric]'  --python .venv310/bin/python   # ml_dtypes：bf16/fp8

# 提交前本地预检（与 CI 等价）
ruff check . && ruff format --check . && mypy src
pytest -m "not requires_bindings" -q
python scripts/merge_gate.py
```

核心层为纯 Python（`>=3.10`），不依赖主仓 bindings；IR 接口层需 bishengir bindings
（**cp310 ABI**），其测试标记 `requires_bindings`，在无绑定环境下跳过并登记为覆盖缺口（D7）。
`hivm-spec doctor` 可随时检查当前环境哪些能力就绪。

## 与 AscendNPU-IR 的关系

本项目面向 [AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR) 的 HIVM 方言（Hybrid Intelligence Virtual Machine），
作为**外部工具**通过其 Python bindings 对接，不侵入主仓；语义模型的交叉验证依赖主仓既有实现
（ConvertHIVMToUpstream / GraphSyncSolver / PlanMemory）。

## License

[Apache License 2.0](LICENSE)
