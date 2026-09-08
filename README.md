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
  ├── 等价验证：两份 MLIR → success/fail + 发散定位
  ├── UB 占用图：MLIR → 片上内存占用曲线与峰值
  ├── 时序图：MLIR → pipe/event 时间线（结构性死锁判定）
  └── （按需生长）
```

关键设计：**模型即数据**（描述进 git、可评审、带信任等级 provisional→cross-validated→anchored）、
**引擎少而固定**（互检交叉验证：与主仓 C++ 链 / GraphSyncSolver / PlanMemory）、
**缺口是合法结论**（`COVERAGE_GAP`/`UNTRUSTED_DESCRIPTION`，禁止静默误判）。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/requirements.md](docs/requirements.md) | 需求基线（FR/NFR/AC 编号体系；§6 未决问题处置状态） |
| [docs/design-framework.md](docs/design-framework.md) | 方案框架与已定决策（D1–D12）、VIR 契约（§6.1）、工具输入契约（§7）、漂移流程（§8.1）、审查台账（§13） |
| [docs/industry-research.md](docs/industry-research.md) | 业界调研与决策依据（Sail/EDA/系统化并发测试/Rosette） |
| [docs/milestone-plan.md](docs/milestone-plan.md) | 里程碑执行计划（M0–M4 任务分解）与工程约定 |
| [AGENTS.md](AGENTS.md) | agent 工作规约：spec-gate 门禁、描述治理纪律、架构不可违约束、跨 pass 定位编排 |

## 路线图

| 里程碑 | 交付 | 状态 |
|---|---|---|
| 前置门禁 | V1 hivm 方言解析 / V2 环境依赖 | ✅ 已通过 |
| 工程基线 | CI 四门禁 + spec-gate + 双层版本策略（D5/D7） | ✅ 已落地 |
| M0 | VIR 契约（T0.0）+ 语法 spike（T0.0b）+ DSL 骨架：描述→静态检查→配置文档→装配 | 🚧 进行中 |
| M1 | UB 占用图（首个 spec 工具） | 未开始 |
| M2 | 时序图 + 结构性死锁判定 | 未开始 |
| M3 | 等价验证（具体执行档）— 含不可裁剪内核（D8） | 未开始 |
| M4 | 符号档（z3）+ 候选性质 | 未开始 |

## 开发

```bash
# 环境（uv；~/.cache 只读时需重定向缓存）
export UV_CACHE_DIR=/tmp/uvcache
uv pip install -e ".[test,dev]"

# 提交前本地预检（与 CI 等价）
ruff check . && ruff format --check . && mypy src
pytest -m "not requires_bindings" -q
python scripts/spec_gate.py --base origin/main --head HEAD
```

核心层为纯 Python（`>=3.10`），不依赖主仓 bindings；IR 接口层需 bishengir bindings
（**cp310 ABI**），其测试标记 `requires_bindings`，在无绑定环境下跳过并登记为覆盖缺口（D7）。

## 与 AscendNPU-IR 的关系

本项目面向 [AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR) 的 HIVM 方言（Hybrid Intelligence Virtual Machine），
作为**外部工具**通过其 Python bindings 对接，不侵入主仓；语义模型的交叉验证依赖主仓既有实现
（ConvertHIVMToUpstream / GraphSyncSolver / PlanMemory）。

## License

[Apache License 2.0](LICENSE)
