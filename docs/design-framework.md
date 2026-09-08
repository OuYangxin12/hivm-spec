# HIVM 语义验证：验证器生成器方案框架（草案 v0.1）

> **状态**：方案框架讨论稿。配套文档：`requirements.md`（需求基线，FR/NFR/AC 编号沿用）。
> **范围声明**：本文档给出组件划分、机制方向与里程碑骨架；标注【已定】的决策不再重复讨论，标注【待定】的条目汇总于 §10，属于后续讨论议程。语法细节一律以"示意"标注，不作为定稿。

---

## 1. 目标回顾

为 vibecoding 的 pass 开发提供快速语义验证：agent 能够在开发回路内获得 **success/fail 判定**与**运行状态视图**，据此快速失败并纠正方向。核心命题：**MLIR 千变万化，正确性类别有限**——因此不逐个建造验证工具，而是建设一个**验证器生成器**：agent 用 DSL 建模语义域，生成器产出面向特定正确性类别的 spec 工具。

## 2. 总体架构

```
                 ┌────────────────────────────────────────┐
  agent 写描述 ──►│ DSL 解释器 / 生成器（纯 Python，唯一重资产）│
                 │  · 描述静态检查(引用完备/一致性/覆盖)       │
                 │  · 归一化 → 配置文档                       │
                 │  · 元编程装配引擎 → spec 工具实例           │
                 └──────┬──────────────────┬───────────────┘
                        ▼                  ▼
             ┌─────────────────┐  ┌─────────────────────────┐
             │ spec 工具(生成物) │  │ spec 工具(生成物)         │
             │ 等价验证         │  │ 状态视图                  │
             │ MLIR×2 →        │  │ MLIR → 时序图/UB占用图    │
             │ success/fail    │  │ （本框架首批，见 §7）      │
             └─────────────────┘  └─────────────────────────┘
                  统一契约：吃 MLIR(+描述引用)，吐结构化结论/视图
```

| 组件 | 职责 | 期次 |
|---|---|---|
| DSL 描述库 | agent 编写的语义域描述（Python 模块），入库版本管理 | 随任务生长 |
| 解释器/生成器 | 加载、静态检查、归一化、产出配置文档、装配工具实例 | M0 起 |
| 固定引擎组 | IR 接口 / 状态机调度 / 值执行 / 视图渲染 / 结论框架 | 分期，§6 |
| 配置文档 | 描述的编译产物：可 diff、可评审、可缓存的规整化语义表 | M0 起 |
| 账本与绊线 | op↔描述覆盖状态、方言演进构建期检查 | M0 起骨架 |

## 3. 已定决策记录

### D1【已定】纯 Python 实现全栈
- **内容**：DSL 本体、解释器/生成器、固定引擎、spec 工具均为纯 Python。
- **理由**：vibecoding 主语言即 Python，agent 编写/消费零摩擦；描述、检查、生成、执行同语言，元编程路径最短；项目已有 Python bindings（`bishengir/python`，含 hivm/hfusion/hacc/annotation 方言绑定）可复用。
- **代价与复查条件**：数值执行与符号执行的吞吐受 Python 限制——首期以 numpy 向量化 + 小规模输入（缩小 tiling）满足 NFR1；若后续不达标，复查条件是"预算内跑不完典型 kernel"，届时仅替换引擎执行核，DSL 与生成器不动。

### D2【已定】生成物 = 配置文档 + 元编程装配，非独立二进制
- **内容**："生成 spec 工具"的含义是：解释器把描述归一化为**一份配置文档**（描述的编译产物），运行时由固定引擎按配置文档装配出工具实例；元编程用于把描述中的声明/函数转换为引擎可执行的对象（执行计划、双模可调用体、检查器参数）。
- **理由**：省去二进制 codegen 的构建链成本；配置文档可 diff、可评审、可缓存、可版本化，天然承载信任等级与 provenance；Python 元编程足以完成装配。
- **影响**：spec 工具的用户体验仍是"一个命令"（薄 CLI 包装：`spec-tool <name> <inputs>`），但其实现是"引擎 + 配置文档"。

### D3【已定】首批生成物 = 状态视图工具，从简单开始
- **内容**：首批工具为 **UB 占用图（M1）**与**时序图（M2）**；等价验证工具后置（M3 具体执行档，M4 符号档）。
- **理由**：状态视图不需要符号引擎与浮点语义（只消费尺寸/效应/调度），是全案中工程最薄、价值最即时的垂直切片；同时 `vm` 状态模型是等价工具的前置地基，先建它不产生返工；直接服务 preload/CV 类特性的高频问题（溢出、流水化失效）。
- **顺序依据**：UB 占用图只需顺序执行语义（无需并发调度），比时序图更简单，故 M1 先行。

### D4【已定】OD1–OD12 全部采纳
采纳结果摘要（业界依据与备选项见 `industry-research.md`）：
- OD1 双模值 → 方案 A：轻量符号句柄（受限表达式子集），z3 延后至 M4；
- OD2 调度口径 → 两层：wait-for 图确定性层（结构性死锁，确定性结论）+ 有界策略探索层（顺序/轮转/pipe 优先/K 随机种子），报告措辞模板"在{策略集}×{展开界}内未发现"；
- OD3 容差 → rtol+atol 全局+per-op 覆盖；round_mode 显式表达；f16/bf16 用 ml_dtypes；
- OD4 描述治理 → 贴方言放置 + CODEOWNERS + trust 升级必须附对拍证据；
- OD5 接线 → AGENTS.md 新增 spec-gate（精确命令 + 强制时机）；
- OD6 锚点 → 三级分层（变换前 IR / HFusion 源 / 硬件 golden）；符号等价限值语义子集（设计约束，非可选项）；
- OD7 双架构 → 单模型 + arch 差异段参数化；禁止两份全量描述；
- OD8 逃生舱 → 注册制 + 效应强制显式 + 信任封顶 provisional + 必配性质测试；
- OD9 schema → spec_version + JSON Schema 校验 + 往返 golden 测试；不兼容变更=大版本；
- OD10 性能 → 定位声明"调试保真优先于吞吐"；numpy 向量化 + 缩小 tiling；numba/Cython 仅 D1 复查触发后引入；
- OD11 模型验证 → 每 op 三类用例（canonical / 与 C++ 链对拍 / Hypothesis 性质），与信任等级绑定；
- OD12 首切片 → 双 kernel：VecAdd（bring-up）+ cv-pipelining.mlir（目标场景）。

## 4. DSL 设计方向

### 4.1 承载形态【方向，语法细节待定】
D1 之下默认倾向 **Python 内嵌 DSL**：描述即 Python 模块（dataclass/装饰器/fluent API 任选其一为骨架），可获得类型检查、补全、单测、import 组织等既有工具链，agent 无需学习新语法。独立文本 DSL 不排除，但需要论证其相对内嵌式的增量收益。

### 4.2 三段结构（概念示意，非定稿语法）

```python
# descriptions/hivm_membase_basic.py —— 示意
spec = Spec(arch="a3", version=1)

# --- ops 段：op 语义 = 值函数 + 效应 ---
@spec.op("hivm.hir.vadd", pipe=PIPE_V, effects=rd("a","b")+wr("out"))
def vadd(a, b, out): out[...] = a[...] + b[...]

@spec.op("hivm.hir.load", effects=rd(src.space)+wr(dst.space))
def load(src, dst): dst[...] = src[...]

# --- vm 段：状态模型 ---
spec.space("gm", capacity=None)
spec.space("ub", capacity=chip.a3.ub_size, align=chip.a3.ub_align)
spec.pipe(PIPE_V, PIPE_MTE2, PIPE_MTE3, PIPE_M, PIPE_FIX)
spec.event("EVENT_ID0".."EVENT_ID7")

# --- checks 段：要生成的工具与口径 ---
spec.check("ub_occupancy", spaces=["ub","cbuf"])
spec.check("timeline", scheduling=conservative_strategies(n=4))
```

- 社区方言（scf/arith/memref/tensor）的语义**不进描述**，由引擎内置（对齐"官方 interpreter 内置 builtin 方言"的惯例）；描述只负责 HIVM 特有 op 与 vm 状态。
- 描述覆盖不到的 op：不写即 `UNMODELED`，工具触发时输出 `COVERAGE_GAP`（FR4），禁止静默。

### 4.3 双模值设计【已定（OD1·方案 A）：轻量符号句柄，z3 后端延后至 M4】
op 值函数写一次，两种模式执行：**具体模式**（操作数是 numpy 数组，直接算）与**符号模式**（操作数是符号表达式句柄，构建表达式图，供 M4 符号等价使用）。这是"描述写一次、后端多样"的技术支点，也是 D2 元编程的主要用武之地（把普通 numpy 风格函数转为双模可调用体）。符号句柄仅覆盖受限表达式子集（算术/比较/select），与 §4.4 概念清单一致；API 细则在 M3 前置设计中定稿（见里程碑计划 T3.1）。

### 4.4 表达力边界与逃生舱
声明式描述优先；无法表达者（如 custom 内置库、复杂布局代数）走显式逃生舱（注册 host Python 函数），逃生舱条目在账本中单独标注信任状态（§10 OD8）。DSL 概念清单（§4.2 所列）之外的表达能力**一律不加**，防止 DSL 范围失控。

## 5. 生成器机制（D2 展开）

```
加载描述 → 静态检查 → 归一化 → 配置文档 → 元编程装配 → spec 工具实例
```

1. **静态检查**（生成前置门）：引用完备（op/space/event/pipe 均已定义）、值函数签名与效应声明一致、check 声明的空间存在、覆盖状态合法。不通过 → 拒绝生成并给出定位（FR4/FR7 在生成层的落点）。
2. **归一化**：把 Python 描述折叠为稳定顺序的规整化语义表（消除书写顺序/风格差异），保证同一描述产出逐字节稳定的配置文档（FR8）。
3. **配置文档**（schema 草案要点，版本化见 §10 OD9）：
```json
{ "spec_version": 1, "arch": "a3",
  "ops": { "hivm.hir.vadd": { "value_ref": "...", "effects": [...], "trust": "provisional", "provenance": {...} } },
  "vm": { "spaces": [...], "pipes": [...], "scheduling": {...} },
  "checks": [...], "coverage": { "modeled": [...], "unmodeled": [...] } }
```
4. **装配**：引擎按 checks 选择与配置文档构建工具实例；薄 CLI 提供统一入口。

## 6. 固定引擎组合

| 引擎 | 职责 | 依赖的描述段 | 期次 |
|---|---|---|---|
| IR 接口引擎 | 解析 MLIR 文本（Python bindings），抽取 op 序列/循环结构/alloc/sync 结构；**前置核实：hivm 方言可从文本解析** | — | M0 |
| 状态机调度引擎 | 按 vm 段模拟执行：pipe/event 状态推进、保守交错策略集、迭代展开 | vm + ops 效应 | M2 |
| 值执行引擎 | 双模值的具体模式执行（numpy），沿 IR 控制流解释 | ops 值函数 | M3 |
| 占用/生存期引擎 | alloc 生存期与 per-space 峰值统计 | ops 效应 + vm 空间 | M1 |
| 视图渲染器 | 甘特式时序图、占用曲线：JSON（agent 消费）+ SVG/文本（人消费） | — | M1/M2 |
| 结论与诊断框架 | verdict 枚举（含 `COVERAGE_GAP`、`UNTRUSTED_DESCRIPTION`）、结构化诊断、描述哈希回写 | 全部 | M0 |
| 账本与绊线 | 覆盖状态机（provisional→cross-validated→anchored）；方言新增 op 的构建期检查 | — | M0 骨架 |

## 7. 首批 spec 工具规格

### 7.1 工具一：UB 占用图（M1）
- **输入**：目标 MLIR + 描述引用 + 架构配置（A3/A5）。
- **行为**：沿 IR 顺序语义计算各 `memref.alloc`（按地址空间）的生存期，输出 per-space 占用曲线与峰值，对照容量线；超限给出贡献者排序（alloc 名、尺寸、活跃区间）。
- **输出**：JSON 曲线数据 + 占用图。
- **已声明的简化口径**：采用编译器标注的 buffer 尺寸（`inferAndSetBufferSize` 等若存在），同时输出"内在最小峰值"作对照；判定边界随输出附带（Q2 钩子）。

### 7.2 工具二：时序图（M2）
- **输入**：同上 + 调度策略参数。
- **行为**：按 vm 段模拟 pipe/event 状态推进，输出 iteration × pipe 的甘特式时间线、wait/set 依赖链标注；多调度策略分别呈现。
- **价值场景**：CV/preload 类特性——两 pipe 从不重叠（流水化失效）、wait 无可满足 set（死锁风险）在图上直接可读。
- **已声明的简化口径**：保守交错策略集 + 有界迭代展开（参数进 §10 OD2）。

### 7.3 统一输出契约
```json
{ "tool": "ub_occupancy", "verdict": "OVERFLOW|OK|COVERAGE_GAP|UNTRUSTED_DESCRIPTION",
  "spec_hash": "...", "trust": "provisional", "details": {...}, "diagnostics": [...] }
```
verdict 为封闭枚举；`COVERAGE_GAP`/`UNTRUSTED_DESCRIPTION` 是合法结论而非错误（FR4/FR7）。

## 8. 信任与验证工作流

- **信任等级**：`provisional`（仅静态检查通过）→ `cross-validated`（与既有实现/分析对拍一致：C++ 链 41 op、GSS、PlanMemory）→ `anchored`（硬件 golden 锚定）。结论携带所用描述的**最低**等级。
- **防自欺（FR6 的 DSL 形态）**：描述进 git，语义篡改=显式 diff；spec 哈希写入结论，可复现可审计；未达 cross-validated 的描述产出的失败结论自动降级标注。
- **模型自身的验证工作流**：对拍用例与描述同库存放（§10 OD11）；新增/修改描述必须跑对拍集。

## 9. 里程碑

| 里程碑 | 内容 | 退出标准 |
|---|---|---|
| M0 | DSL 骨架 + 静态检查 + 配置文档产出 + 账本骨架；3-op toy 描述（alloc/load/vadd/store 子集）跑通 | toy 描述→配置文档→装配出可调用实例 |
| M1 | UB 占用图端到端（首个垂直切片；双 kernel：VecAdd + cv-pipelining） | 真实 HIVM 测试 kernel 出图；注入超限缺陷可检出 |
| M2 | 时序图 + pipe/event 模型 + 保守交错 | cv-pipelining 测试 kernel 出时序图；注入配对缺陷可在图中/结论中暴露 |
| M3 | 等价验证（具体执行档；三级锚点分层 + rtol/atol 容差策略） | 两份结构不同 IR 同输入对拍出 verdict；发散定位可用 |
| M4 | 符号档 + 候选性质（未初始化读、同步静态配对） | 有界范围内符号等价可用 |

## 10. 决策点处置：OD1–OD12 已全部采纳

全部采纳结果与业界依据见 `industry-research.md`；对框架正文的回写见 §3(D4)、§4.3、§9。无遗留待定决策；执行期新增的子决策在里程碑计划中随做随记。

## 11. 风险

| 风险 | 缓解 |
|---|---|
| 表达力悬崖（mmad 布局代数/custom 库等进不了声明式描述） | 逃生舱 + 账本如实标注；DSL 概念清单封闭（§4.4） |
| 双源真理（DSL 语义 vs 既有 C++ 链隐式语义漂移） | 漂移=bug 信号的对拍机制；长期收敛策略待定（可挂 OD6 讨论） |
| Python 吞吐不满足 NFR1 | D1 复查条件：仅替换引擎执行核，DSL/生成器不动 |
| 描述本身错误（元问题） | 静态检查 + 对拍工作流 + 信任等级降级标注 |
| Python bindings 对 hivm 文本解析能力不足 | M0 前置核实；备选：经 bishengir-opt 规范化后解析 |

## 12. 需求追踪

| 需求 | 落点 |
|---|---|
| FR1 语义等效 | M3/M4 等价引擎（锚点 OD6） |
| FR2 死锁 | M2 时序图（观测）→ 后续判定档 |
| FR3 溢出 | M1 UB 占用图 → 后续判定档 |
| FR4 按需覆盖 | 描述即扩展；静态检查；`COVERAGE_GAP` |
| FR5 可诊断 | 诊断框架 + 状态视图 |
| FR6 防自欺 | git 化描述 + spec 哈希 + 信任等级 |
| FR7 元可信 | 静态检查 + `UNTRUSTED_DESCRIPTION` + 元自检 |
| FR8 确定性 | 归一化 + 固定种子/策略集 |
| FR9 程序化 | 薄 CLI + JSON 契约 |
| FR10 候选性质 | M4 |
| NFR1/3 | D1 + 小规模输入；纯 Python 无硬件 |
| NFR2 | 互检 + 信任等级 + 误判优先压制 |
| NFR4/5 | 描述增量生长 + 账本绊线 |
