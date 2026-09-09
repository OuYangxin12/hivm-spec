# 里程碑执行计划（M0–M4）

> **状态**：v0.1。配套：`design-framework.md`（已定决策 D1–D4）、`requirements.md`（验收基线）。
> **预估说明**：人时为粗估（单人 + AI 协作），随做随校准；任务编号 T<m>.<n> 供跟踪。

## 0. 总览

| 里程碑 | 目标 | 预估 | 退出标准 | 前置 |
|---|---|---|---|---|
| M0 | VIR 契约 + 语法 spike + DSL 骨架垂直打通：描述→静态检查→配置文档→可装配实例 | 1.5 周 | toy 描述全链路跑通，两次生成逐字节一致；VIR 不变量单测过；CI 四门禁全绿 | V1/V2 核实 ✅ |
| M1 | UB 占用图端到端（首个 spec 工具） | 1–2 周 | 双 kernel 出图；注入缺陷检出；≤10s（D10）；VIR 已预留解释执行钩子 | M0 |
| M2 | 时序图 + 同步语义（观测 + 结构性死锁判定） | 2 周 | cv kernel 出图；注入配对缺陷暴露；≤30s（D10） | M1 |
| M3 | 等价验证（具体执行档）——**含 D8 不可裁剪内核** | 2–3 周 | before/after 对拍出 verdict + 发散定位；≤5min（D10） | M2 |
| M4 | 符号档 + 候选性质 | 3 周 | 受限子集符号等价可用 | M3 |

## 1. 前置核实（M0 门禁，不通过则先走备选）

| 编号 | 核实项 | 通过标准 | 失败备选 |
|---|---|---|---|
| V1 | Python bindings 从文本解析 hivm 方言（`mlir.ir.Context` 注册 hivm 后 `Module.parse`） | `test/Dialect/HIVM` 抽样 10 个 .mlir 可解析、op 可遍历 | 经 `bishengir-opt` 规范化输出后解析；或走 C API |
| V2 | 环境与依赖：Python 版本、numpy/hypothesis/jsonschema 可安装（含离线约束确认） | 依赖声明入 requirements，可复现安装 | 与构建系统集成（cmake FindPython） |
| V3 | 代码落位【已定】：本仓库为独立项目——引擎/DSL 入 `src/hivm_spec/`，描述库 `specs/`，对拍用例 `specs/cases/`；与 AscendNPU-IR 以 Python bindings 环境依赖对接，不侵入主仓 | 目录已建 | — |

### 门禁核实结果（首个执行记录）

- **V2 ✅ 通过**：Python 3.14.4（系统解释器）；依赖经 uv 装入 `.venv`：numpy 2.5.3 / jsonschema 4.26.0 / hypothesis 6.167.1 / pytest 9.1.1（cp314 轮子可用，venv 基于系统解释器可长期使用）。环境备注：系统无 pip/ensurepip、`~/.cache` 只读，已用 uv 用户态安装并将缓存重定向 `/tmp`。
- **V1 ✅ 通过**（经编译服务器 ssh `build`）：在 `~/proj/AscendNPU-IR` 开启 `MLIR_ENABLE_BINDINGS_PYTHON=ON` 增量构建（为其 python3.10 补装 numpy、pybind11==2.13.6——3.x 与 MLIR 19 绑定不兼容）。绑定包位于 `build/tools/bishengir/bishengir/python_packages/bishengir`（完整 MLIR API 以 `bishengir.` 前缀重根化，`from bishengir import ir`）。验证脚本 `scripts/v1_hivm_parse_test.py`：`test/Dialect/HIVM` 172 个正例严格解析 134 个；38 个失败均为测试夹具属性（26 个故意用未注册假 op、11 个单文件多用例重定义、1 个 test-only 方言），**无一为 hivm 方言能力缺口**；allow-unregistered 下 157/172。**发现主仓 bindings 缺口（建议上游回报）**：`_mlir_libs` 站点初始化只探测 `_mlirRegisterEverything`，不探测带前缀的 `_bishengirRegisterEverything`，导致 hivm/hfusion/hacc 不被自动注册；引擎须在 Context 创建后显式调用 `_bishengirRegisterEverything.register_dialects(ctx)`。注意：绑定 .so 为 cp310，本地沙箱 Python 3.14 不能直接加载——T1.1 起 IR 接口层将以"本地 uv 管理 py3.10 + 拉取绑定树"或"远端执行"方式对接。

## 2. M0：DSL 骨架

| 任务 | 内容 | 验收 |
|---|---|---|
| T0.1 | Spec API 骨架：`Spec`/`op`/`space`/`pipe`/`event`/`check` 构造器（dataclass + fluent），按 §4.2 三段结构 | toy 描述可构造 |
| T0.2 | 静态检查器：引用完备性、签名-效应一致性、check 合法性；错误带描述内定位 | 3 类违规样例均被拒绝且可读 |
| T0.3 | 归一化 + 配置文档：稳定排序、JSON 输出、jsonschema 校验 | 同一描述两次生成逐字节一致（FR8）；schema 往返 golden 转实测（CI job 3） |
| T0.4 | 信任账本骨架：trust/provenance/`spec_hash`（sha256，定义见框架 §7.3）；覆盖报告（modeled/unmodeled）；**drift ledger 条目结构（D9）** | 账本随配置文档输出；漂移条目可登记 |
| T0.5 | 生成 CLI：`hivm-spec gen <desc.py> -o config.json`（入口已在 pyproject 注册） | 端到端命令可用，替换当前 PENDING 返回 |
| T0.6 | toy 描述：load/vadd/store + gm/ub + ub_occupancy check | 为 M1 提供输入；**spec-gate R1 转硬失败** |
| T0.7 ✅ | 性质测试骨架：Hypothesis 注册表 + 示例（vadd 交换律） | 框架可运行、可扩展 |
| T0.8 ✅ | 账本绊线首版：pytest 遍历 hivm 已注册 op vs 账本，未声明 op 报告（cmake/tablegen 集成后置） | 对全量已注册 op 输出覆盖报告（实测 **114** 个，已建模 9） |

> **M0 新增前置任务（架构审查 D6/D11/D13）**——须在 T0.1 之前完成：
>
> | 任务 | 内容 | 验收 |
> |---|---|---|
> | **T0.0** | **VIR 契约（D6）**：定义 `VNode`/`VRegion`/`VLoop`/`VAlloc`/`VSync`/`VTrace` 钩子/`coverage` 的 dataclass 与四条不变量（框架 §6.1）；含值槽位与执行轨迹钩子（D8 为 M3 预留） | 契约模块 + 不变量单测；后续引擎一律消费 VIR，无绕过 |
> | **T0.0b** | **语法 spike（D11）**：以示意语法试写三 op——vadd（平凡）/ 跨迭代多 Wait-Set 事件对（效应复杂度，取自 `sync_related_args`）/ mmad 类（预期逃生舱）；产出"可表达 or 逃生舱"结论 | 三 op 结论入档；T0.1 API 据此定稿 |
> | **T0.0c** | **工程门禁落地（D5/D7）**：CI 四 job + spec-gate + lint/类型 + 双层版本策略 | ✅ 已完成（审查整改批次） |
| **T0.0d** | **测试语料 L0/L1（D13）**：`specs/cases/corpus/l0/` 手写微例 5–8 个（<30 行，覆盖 VIR 四不变量所需结构）+ `manifest.json` 契约；L1 从主仓 21 个干净 UT 中选 8–12 个入库并锚定来源 commit（规格见框架 §14） | L0 全部可转 VIR 且不变量单测通过；L1 每份无 `COVERAGE_GAP`；manifest 完备 |

## 3. M1：UB 占用图（首个 spec 工具）

| 任务 | 内容 | 验收 |
|---|---|---|
| T1.0 ✅ | **目标 kernel op 清点与 gap 分析**：cv-pipelining.mlir 的 hivm op 集合 vs 账本 → 建模工作清单（提前暴露 M2 的建模量） | 已完成，见 `docs/tasks/M1.md` §1：待建模 **9 个 op**，仅 `mmadL1` 需逃生舱 |
| T1.1 ✅ | IR 接口引擎：解析 kernel、遍历函数体，**产出 VIR**（T0.0 契约；alloc/效应节点 + scf 结构标注）；不认识的结构落入 VIR `coverage` 并上报 `COVERAGE_GAP`；含 bindings bootstrap 模块（框架 §6 强制条款）+ `requires_bindings` 回归 | L2 全部 19 份严格解析并降级成功；bootstrap 回归可跑（PR #6/#8） |
| T1.2 ✅ | 生存期/峰值引擎：alloc→last use 区间（顺序语义）；per-space 曲线与峰值 | 区间语义三性质已验证（不重叠不累加/重叠必累加/跨循环计入） |
| T1.3 ✅ | 尺寸策略：静态 shape 直读；动态维度降级 UNKNOWN_SIZE；编译器标注优先并记录来源（`hivm.multi_buffer` 实测生效） | 动态样例可用；multi_buffer 在 preload kernel 命中 |
| T1.4 ✅ | 容量判定：对照 arch 段容量（D4/OD7 参数化）；OVERFLOW/OK + 贡献者排序（alloc 名/尺寸/活跃区间）；空洞 OK 防线（L1 实测补强） | 注入缺陷被检出（AC1 溢出维）✅ |
| T1.5 ✅ | 渲染器：JSON 曲线 + 文本占用图（sparkline/比例条/贡献者；SVG 缓行） | 人/agent 双消费可用 |
| T1.6 ✅ | 结论框架落地：verdict 封闭枚举 + diagnostics + spec_hash（由配置文档字节串重算）+ trust 降级标注；缺口/问题退出码分离 | §7.3 契约字段齐备 |
| T1.7 ✅ | 双 kernel 验收：L0 bring-up + cv-pipelining 目标场景（18 分节 + preload） | 19 份全部出图，单 kernel 最大 **0.20s**（余量 51×）；实测值已回填 D10 表 |
| T1.8 ✅ | AGENTS.md spec-gate 首版（§6.5）：按改动性质选最小执行集 + 基线对照 | 指令入库 |

## 4. M2：时序图与同步语义 ✅（2026-09-09 完成，实测见 `docs/tasks/M2.md` §8）

| 任务 | 内容 | 验收 | 状态 |
|---|---|---|---|
| T2.1 | pipe/event 状态机（消费 vm 段 + **VIR**，不另建遍历核 D6）：op→pipe 归属、set/wait/pipe_barrier/sync_block 语义执行 | 单核顺序策略下时间线正确；无绕过 VIR 的 MLIR 访问 | ✅ |
| T2.2 | 交错策略集 + 有界迭代展开参数（顺序/轮转/pipe 优先/K 随机种子） | 策略可配置、可复现（种子化） | ✅ |
| T2.3 | wait-for 图确定性层：结构性死锁（不可满足 wait/环）判定，确定性结论 | 注入无配对 wait 被确定性判定 | ✅ |
| T2.4 | 时序渲染：iteration×pipe 甘特 + wait/set 依赖标注 + Chrome Trace Event Format 输出（perfetto 可视化） | cv kernel 图可读；trace 可导入 perfetto | ✅ |
| T2.5 | 报告口径：探索层结论统一"在{策略集}×{展开界}内未发现"（D4/OD2） | 结论字段落 schema | ✅ |
| T2.6 | cv kernel 验收：注入配对缺陷（跨迭代 wait/set 错位）在图或结论中暴露 | AC1 死锁维（观测层） | ✅ |

## 5. M3：等价验证（具体执行档）

| 任务 | 内容 | 验收 |
|---|---|---|
| T3.1 | 双模值具体模式引擎 + **符号句柄 API 定稿**（受限子集；D4/OD1 方案 A） | 同一 op 函数双模式输出一致（具体小例） |
| T3.2 | 控制流解释：**复用 VIR 的 `VRegion`/`VLoop` 与 T0.0 值槽位**（D6/D8，扩展而非重写）；scf.for/if/while + 社区方言语义内置（arith/memref/tensor 子集按需扩展） | 目标 kernel 可解释执行；遍历核仍唯一 |
| T3.3 | 数值基础设施：ml_dtypes（f16/bf16）；round_mode 显式表达 | 定点样例对拍一致 |
| T3.4 | 输入策略：固定种子随机 + 缩小 tiling 参数化 | 输入可复现 |
| T3.5 | 差分对拍：默认锚点=变换前 IR；rtol+atol 容差（全局+per-op） | 两份结构不同 IR 出 verdict |
| T3.6 | 发散定位：逐 op 值哈希 trace + 首发散点报告（op/位置/用例/前后值） | 注入语义缺陷被检出且定位可用（AC1 语义维） |
| T3.7 | D1 性能复查：实测 **D10 per-tool 预算**（等价验证 ≤5min、特性级 ≤10min）；超限则触发执行核替换评估（不动 DSL/生成器） | 预算达标或复查结论入档；实测值回填 D10 表 |

## 6. M4：符号档与候选性质

| 任务 | 内容 | 验收 |
|---|---|---|
| T4.1 | z3 后端接入符号句柄（受限子集提升为 SMT） | op 函数零改动切换符号模式 |
| T4.2 | 有界符号等价：值语义子集（限子集为设计约束，D4/OD6） | 子集内两 IR 等价出 SAT/UNSAT |
| T4.3 | 未初始化读检查：具体模式 poison 填充 + 读判定 | 注入用例检出 |
| T4.4 | 同步静态配对检查：set/wait 配对完整性（账本级，快速） | 注入用例检出 |

## 7. 横切事项

- **CI 接入**：✅ 已落地（`.github/workflows/ci.yml` 四 job：lint+types / core 矩阵 3.10+3.12 / schema golden / spec-gate）；M1 末 lit 接入评估已完成：**不接入** check-bishengir（纯下游外部工具定位，主仓 lit 负责结构回归；边界口径已落 requirements Q7）；
- **描述 PR 模板**：trust 升级 checklist（三类用例证据，D4/OD4+OD11）；**已由 spec-gate R2 机械强制**；
- **对拍用例库**：`specs/cases/`，与描述同 PR 演进（OD11）；**语料按 D13 分层引入**（L0 手写 → L1 干净 UT → L2 按需剥离 → L3 e2e dump），入库并以 manifest 锚定来源 commit（框架 §14）；
- **性能复核点**：M3.7 是 D1 的唯一预设复查点；per-tool 预算见 D10；
- **文档联动**：每里程碑退出时更新框架文档 §9 状态与本计划勾选。

## 8. 工程约定

- **PR 流程**：main 分支禁直推（分支保护已启用），所有改动走 feature 分支 → PR → 合入；单人阶段 PR 免**人工**审批直接 merge，多协作者后再收紧审批。
- **机器门禁不可绕过（D5）**：单人阶段免人工审批，但 CI 四门禁必须全绿方可合入——这是 FR6 防自欺的实际载体，不得以"单人阶段"为由跳过或放松。放松门禁的改动本身受 CODEOWNERS 看护。
- **本地预检命令**（与 CI 等价，agent 提交前须跑）：

  ```bash
  ruff check . && ruff format --check . && mypy src
  pytest -m "not requires_bindings" -q
  python scripts/spec_gate.py --base origin/main --head HEAD
  ```

- **Python 版本（D7）**：核心层 `>=3.10`（cp310 绑定 ABI 地板），不 import `bishengir`；IR 接口层测试标 `requires_bindings`，本地/CI 跳过并登记为覆盖缺口，回归由编译服务器远端执行承担。
- **依赖管理**：主依赖钉兼容区间（FR8 可复现）；环境用 uv 管理（`~/.cache` 只读，需 `export UV_CACHE_DIR=/tmp/uvcache`）。
- **与 AscendNPU-IR 主仓的关系**：纯下游使用者（Python bindings 环境依赖，见 V1 记录）。发现的主仓 bindings 注册缺口及其修复方案已存档于 V1 执行记录与项目记忆备查；bootstrap workaround 已提升为框架 §6 引擎强制条款；不主动上游化，未来需要时可按主仓 AGENTS.md 规范重做补丁。
- **语义漂移**：一律先登记 drift ledger 并冻结该 op 的 trust 升级；默认权威为主仓 C++ 链，仅硬件 golden 可推翻（D9，流程见框架 §8.1）。

## 9. 执行风险与缓解

| 风险 | 触发信号 | 缓解 |
|---|---|---|
| V1 解析缺口 | M0 门禁失败 | ✅ 已通过；备选路径（规范化后解析/C API）保留，不影响架构 |
| cv kernel 建模量超预期 | T1.0 gap 清单过大 | 优先建模 check 所需效应；值语义缺口走 COVERAGE_GAP 诚实降级 |
| 动态 shape 尺寸来源不足 | T1.3 | 首版限静态/参数化并在结论中声明覆盖边界 |
| 解释执行吞吐不足 | T3.7 超预算 | D1 复查条款：仅替换执行核；判据为 D10 per-tool 预算 |
| 描述质量参差（agent 撰写） | 对拍失败率 | OD11 三类用例门槛 + trust 降级标注 + spec-gate R1/R2 |
| VIR 表达力不足 | M2/M3 需绕过 VIR 直接读 MLIR | 按框架 §6.1 版本化演进契约，不允许私有分支解释核（D6） |
| M3 延期导致 P0 未兑现 | M2 退出后排期缺口 | D8 不可裁剪条款：范围可缩、能力不可删；延期须在框架 §3 D8 显式记录 |
| 语法 spike 暴露表达力悬崖 | T0.0b 三 op 有 ≥2 落逃生舱 | 重新评估 DSL 概念清单边界（§4.4）再定 T0.1 API |
