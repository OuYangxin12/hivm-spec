# 里程碑执行计划（M0–M4）

> **状态**：v0.1。配套：`design-framework.md`（已定决策 D1–D4）、`requirements.md`（验收基线）。
> **预估说明**：人时为粗估（单人 + AI 协作），随做随校准；任务编号 T<m>.<n> 供跟踪。

## 0. 总览

| 里程碑 | 目标 | 预估 | 退出标准 | 前置 |
|---|---|---|---|---|
| M0 | DSL 骨架垂直打通：描述→静态检查→配置文档→可装配实例 | 1 周 | toy 描述全链路跑通，两次生成逐字节一致 | V1/V2 核实 |
| M1 | UB 占用图端到端（首个 spec 工具） | 1–2 周 | 双 kernel 出图；注入缺陷检出；≤30s | M0 |
| M2 | 时序图 + 同步语义（观测 + 结构性死锁判定） | 2 周 | cv kernel 出图；注入配对缺陷暴露 | M1 |
| M3 | 等价验证（具体执行档） | 2–3 周 | before/after 对拍出 verdict + 发散定位 | M2 |
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
| T0.3 | 归一化 + 配置文档：稳定排序、JSON 输出、jsonschema 校验 | 同一描述两次生成逐字节一致（FR8） |
| T0.4 | 信任账本骨架：trust/provenance/spec_hash（sha256）；覆盖报告（modeled/unmodeled） | 账本随配置文档输出 |
| T0.5 | 生成 CLI：`python -m hivm_spec gen <desc.py> -o config.json` | 端到端命令可用 |
| T0.6 | toy 描述：load/vadd/store + gm/ub + ub_occupancy check | 为 M1 提供输入 |
| T0.7 | 性质测试骨架：Hypothesis 注册表 + 示例（vadd 交换律） | 框架可运行、可扩展 |
| T0.8 | 账本绊线首版：pytest 遍历 hivm 已注册 op vs 账本，未声明 op 报告（cmake/tablegen 集成后置） | 对全 122 op 输出覆盖报告 |

## 3. M1：UB 占用图（首个 spec 工具）

| 任务 | 内容 | 验收 |
|---|---|---|
| T1.0 | **目标 kernel op 清点与 gap 分析**：cv-pipelining.mlir 的 hivm op 集合 vs 账本 → 建模工作清单（提前暴露 M2 的建模量） | 清单进账本，缺口有归属 |
| T1.1 | IR 接口引擎：解析 kernel、遍历函数体，产出结构化 skeleton（alloc/效应节点 + scf 结构标注）；不认识的结构显式上报 `COVERAGE_GAP` | 两个目标 kernel skeleton 正确 |
| T1.2 | 生存期/峰值引擎：alloc→last use 区间（顺序语义）；per-space 曲线与峰值 | 与手算一致（小型样例） |
| T1.3 | 尺寸策略：静态 shape 直读；动态维度经测试配置参数化；编译器尺寸标注存在则优先并记录来源 | 动态样例可用 |
| T1.4 | 容量判定：对照 arch 段容量（D4/OD7 参数化）；OVERFLOW/OK + 贡献者排序（alloc 名/尺寸/活跃区间） | 注入缺陷（扩 alloc）被检出（AC1 溢出维） |
| T1.5 | 渲染器：JSON 曲线 + SVG/文本占用图 | 人/agent 双消费可用 |
| T1.6 | 结论框架落地：verdict 封闭枚举 + diagnostics + spec_hash + trust 降级标注 | §7.3 契约字段齐备 |
| T1.7 | 双 kernel 验收：VecAdd bring-up + cv-pipelining 目标场景 | 两者出图，预算内（≤30s，AC2/NFR1） |
| T1.8 | AGENTS.md spec-gate 首版：改 HIVM pass → 必跑对应 spec 工具（D4/OD5） | 指令入库 |

## 4. M2：时序图与同步语义

| 任务 | 内容 | 验收 |
|---|---|---|
| T2.1 | pipe/event 状态机（消费 vm 段）：op→pipe 归属、set/wait/pipe_barrier/sync_block 语义执行 | 单核顺序策略下时间线正确 |
| T2.2 | 交错策略集 + 有界迭代展开参数（顺序/轮转/pipe 优先/K 随机种子） | 策略可配置、可复现（种子化） |
| T2.3 | wait-for 图确定性层：结构性死锁（不可满足 wait/环）判定，确定性结论 | 注入无配对 wait 被确定性判定 |
| T2.4 | 时序渲染：iteration×pipe 甘特 + wait/set 依赖标注 + Chrome Trace Event Format 输出（perfetto 可视化） | cv kernel 图可读；trace 可导入 perfetto |
| T2.5 | 报告口径：探索层结论统一"在{策略集}×{展开界}内未发现"（D4/OD2） | 结论字段落 schema |
| T2.6 | cv kernel 验收：注入配对缺陷（跨迭代 wait/set 错位）在图或结论中暴露 | AC1 死锁维（观测层） |

## 5. M3：等价验证（具体执行档）

| 任务 | 内容 | 验收 |
|---|---|---|
| T3.1 | 双模值具体模式引擎 + **符号句柄 API 定稿**（受限子集；D4/OD1 方案 A） | 同一 op 函数双模式输出一致（具体小例） |
| T3.2 | 控制流解释：scf.for/if/while + 社区方言语义内置（arith/memref/tensor 子集按需扩展） | 目标 kernel 可解释执行 |
| T3.3 | 数值基础设施：ml_dtypes（f16/bf16）；round_mode 显式表达 | 定点样例对拍一致 |
| T3.4 | 输入策略：固定种子随机 + 缩小 tiling 参数化 | 输入可复现 |
| T3.5 | 差分对拍：默认锚点=变换前 IR；rtol+atol 容差（全局+per-op） | 两份结构不同 IR 出 verdict |
| T3.6 | 发散定位：逐 op 值哈希 trace + 首发散点报告（op/位置/用例/前后值） | 注入语义缺陷被检出且定位可用（AC1 语义维） |
| T3.7 | D1 性能复查：实测 NFR1 预算；超限则触发执行核替换评估（不动 DSL/生成器） | 预算达标或复查结论入档 |

## 6. M4：符号档与候选性质

| 任务 | 内容 | 验收 |
|---|---|---|
| T4.1 | z3 后端接入符号句柄（受限子集提升为 SMT） | op 函数零改动切换符号模式 |
| T4.2 | 有界符号等价：值语义子集（限子集为设计约束，D4/OD6） | 子集内两 IR 等价出 SAT/UNSAT |
| T4.3 | 未初始化读检查：具体模式 poison 填充 + 读判定 | 注入用例检出 |
| T4.4 | 同步静态配对检查：set/wait 配对完整性（账本级，快速） | 注入用例检出 |

## 7. 横切事项

- **CI 接入**：M0 起 pytest 随 PR；M1 末评估 lit 接入 `check-bishengir`；
- **描述 PR 模板**：trust 升级 checklist（三类用例证据，D4/OD4+OD11）；
- **对拍用例库**：`specs/cases/`，与描述同 PR 演进（OD11）；
- **性能复核点**：M3.7 是 D1 的唯一预设复查点；
- **文档联动**：每里程碑退出时更新框架文档 §9 状态与本计划勾选。

## 8. 执行风险与缓解

| 风险 | 触发信号 | 缓解 |
|---|---|---|
| V1 解析缺口 | M0 门禁失败 | 备选路径（规范化后解析/C API），不影响架构 |
| cv kernel 建模量超预期 | T1.0 gap 清单过大 | 优先建模 check 所需效应；值语义缺口走 COVERAGE_GAP 诚实降级 |
| 动态 shape 尺寸来源不足 | T1.3 | 首版限静态/参数化并在结论中声明覆盖边界 |
| 解释执行吞吐不足 | T3.7 超预算 | D1 复查条款：仅替换执行核 |
| 描述质量参差（agent 撰写） | 对拍失败率 | OD11 三类用例门槛 + trust 降级标注 |
