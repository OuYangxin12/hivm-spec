# 业界调研与决策建议（对应设计框架 §10 OD1–OD12）

> **状态**：调研纪要 v0.1，配套 `design-framework.md`。每个决策点给出：业界参照 → 可选项 → 建议（★）。

## 0. 四个参照系总览

| 参照系 | 代表 | 与本方案的同构关系 |
|---|---|---|
| ISA 规范语言 | [Sail](https://github.com/RaitoBezarius/sail)（RISC-V/CHERI 官方采用）、ARM ASL | 一份 ISA 描述 → 生成模拟器/SMT 工件；= "DSL 前端 + 多后端"的已验证范式 |
| EDA 等价验证与调试 | LEC（Cadence Conformal、Synopsys Formality）、[CIRCT circt-lec（SMT）](https://github.com/cowardsa/circt/blob/a095e2f7b92ba9d93655d1ddf5759cbe3e38c930/docs/Tools/circt-lec.md)、[CIRCT Formal Verification](https://raw.githubusercontent.com/llvm/circt/e45721de540709ca74b382e43b92e05bee635a69/docs/FormalVerification.md)、waveform viewer（Verdi/GTKWave）、SVA 断言 | pass 后语义保持 = netlist 等价检查；状态视图工具 = 波形查看器。**业界日常调试依赖波形远多于 LEC**——印证 D3 |
| 系统化并发测试 | Microsoft Chess/Shuttle、[Loom（与 Shuttle/Turmoil 对比评估）](https://github.com/autumn-foundation/autumn-harvest/blob/64ceca95731221410324ec3ece74a00a84fab354/docs/testing/concurrency-model-checking.md)、ThreadSanitizer happens-before | 有界调度枚举 + 向量时钟 = OD2 调度口径的直接模板 |
| Solver-aided 编程 + ML 编译器测试 | [Rosette（同一程序具体/符号双模）](https://www.sciencedirect.com/science/article/pii/S2590118423000485)、Triton TRITON_INTERPRET（numpy，调试定位）、IREE/torch.testing 数值容差、[ml_dtypes](https://android.googlesource.com/platform/external/ml_dtypes/+/refs/tags/android-16.0.0_r1/README.md) | OD1/OD3/OD10/OD11 的模板 |

## 1. 逐决策点建议

### OD1 双模值 API
- **业界**：Rosette 证明"宿主语言 + solver 构造 = 具体与符号同一份代码"可行；Triton interpreter 用 `TensorHandle` 抽象 numpy 执行。
- **选项**：A. 自建轻量符号句柄（受限表达式子集：算术/比较/select），z3 依赖延后到 M4；B. 直接以 z3 对象做双模（一步到位，重）。
- **★建议 A**：M1–M3 零求解器依赖；表达式子集与 4.4 概念清单一致，天然可静态检查。M4 再评估是否直接换 z3 后端。

### OD2 调度抽象口径（死锁/时序）
- **业界**：系统化并发测试 = **有界**调度枚举（preemption bounding），不做完备证明；死锁判定经典算法 = wait-for 图环检测（调度无关，一次可判）；TSan 用 happens-before 向量时钟做序关系。
- **★建议两层**：(a) 确定性层：wait-for 图可达性分析，报告"结构性不可满足/成环"（此类结论是确定性的）；(b) 探索层：有界策略集（顺序、轮转、按 pipe 优先、K 个随机种子），报告措辞模板："在 {策略集} × {迭代展开上界} 内未发现"（Q2 口径）。

### OD3 浮点容差
- **业界**：`torch.testing.assert_close` 与 IREE `check.expect_almost_eq` 均为 rtol+atol 组合并按 dtype 缩放；硬件 golden 对拍常用 ULP 上界。
- **★建议**：checks 段支持全局默认 + per-op 覆盖的容差策略（rtol+atol+dtype 维度）；round_mode 不依赖 numpy 隐式舍入，在值语义中显式表达；f16/bf16 用 ml_dtypes 补 numpy 空缺。

### OD4 描述治理
- **业界**：Sail/RISC-V 模型的 spec-first 治理；protobuf/buf 的 breaking-change CI。
- **★建议**：描述库贴方言放置（`include/bishengir/Dialect/HIVM/specs/` 或独立 `specs/`，OD4a 二选一）；CODEOWNERS 标注；PR 门槛 = 静态检查通过 + trust 升级必须附对拍证据。

### OD5 AGENTS.md 接线
- **业界**：仓库已有 skill 的 post-build-ut-gate 惯例，agent 对"精确命令 + 强制时机"的门禁遵循度最高。
- **★建议**：并列新增 spec-gate：改 HIVM pass → 必跑对应 spec 工具；改描述 → 必跑对拍集；命令与预算写死进 AGENTS.md。

### OD6 等价锚点
- **业界**：EDA LEC 是"结构变换保语义"的工业标准，但**昂贵且对顺序结构（loops/内存）能力有限**，业界常态是"LEC 限子集 + 仿真差分为主力"；编译器界同构实践为 translation validation（CompCert）。
- **★建议**：三级锚点分层启用——默认差分锚点=变换前 IR；第二锚=上层源 IR（HFusion）；第三=稀疏硬件 golden。符号等价（M4）明确限定在值语义子集，与 EDA "LEC 限组合逻辑"同构。

### OD7 双架构（A3/A5）
- **业界**：Sail 用 profile/extension 参数化单一模型（RISC-V XLEN、ARM profile）。
- **★建议**：op 语义一份共享；`arch` 段只存差异（容量常量 + regbase 特有 op 分文件）；`Spec(arch=...)` 参数化生成。禁止 copy-paste 两份全量描述。

### OD8 逃生舱
- **业界**：K framework hooks、Sail builtins、uninterpreted function with contract。
- **★建议**：host Python 函数注册制；效应声明强制显式；信任封顶 `provisional`（不可直升）；必配性质测试；账本独立色标。

### OD9 配置文档 schema 版本化
- **业界**：protobuf field-number + buf breaking；JSON Schema 校验。
- **★建议**：`spec_version` 整数 + JSON Schema 校验器 + 不兼容变更=大版本 + schema 往返 golden 测试入 CI。

### OD10 Python 性能
- **业界**：Triton interpreter 即 numpy 实现，**定位就是调试、性能弱是已知且被接受的**——与我们的场景定位完全一致。
- **★建议**：明确声明"调试保真优先于吞吐"；numpy 向量化 + 缩小 tiling 输入；numba/Cython 仅在 D1 复查条件触发后引入。

### OD11 模型自身的验证
- **业界**：property-based testing（Hypothesis）+ metamorphic testing；EDA golden model 也依赖 reference-case 回归。
- **★建议**：每条 op 描述配三类用例：① canonical 手工小例；② 与既有 C++ 链对拍（41 op 现成对照）；③ 性质测试（交换/结合/恒等等，Hypothesis 生成）。信任等级升级与这三者绑定。

### OD12 首个垂直切片
- **业界**：芯片 golden model bring-up 惯例 = 先 trivial kernel 验证骨架，再上目标 workload。
- **★建议**：双 kernel——VecAdd（bring-up，仓库唯一 E2E 参考）+ `cv-pipelining.mlir`（目标价值场景）；M1 的 UB 图两者都跑。

## 2. 可直接复用的开源件

| 组件 | 用途 | 期次 |
|---|---|---|
| numpy | 具体模式值执行 | M1+ |
| [ml_dtypes](https://android.googlesource.com/platform/external/ml_dtypes/+/refs/tags/android-16.0.0_r1/README.md) | bf16/f16 numpy dtype | M3 |
| [Perfetto / Chrome Trace Event Format](https://perfetto.dev/docs/getting-started/other-formats) | 时序图标准化输出（免费获得高质量可视化 UI） | M2 |
| Hypothesis | op 语义性质测试 | M0+ |
| jsonschema | 配置文档校验 | M0 |
| z3-solver | 符号模式（延后） | M4 |
| numba（备选） | D1 复查触发后的执行核加速 | 视需要 |

## 3. 推荐组合汇总

| OD | 推荐 | 备选保留 |
|---|---|---|
| OD1 | 轻量符号句柄（A） | z3 双模（B，M4 复评） |
| OD2 | wait-for 图确定性层 + 有界策略探索层 | 完备模型检查（不做） |
| OD3 | rtol+atol 全局+per-op；round_mode 显式；ml_dtypes | ULP（硬件对拍时） |
| OD4 | 贴方言目录 + CODEOWNERS + PR 门槛 | 独立 specs/ 目录 |
| OD5 | AGENTS.md spec-gate | — |
| OD6 | 三级锚点分层；符号等价限子集 | — |
| OD7 | 单模型 + arch 差异段 | 双份全量描述（禁止） |
| OD8 | 注册制逃生舱 + 信任封顶 | — |
| OD9 | spec_version + JSON Schema + 往返 golden | — |
| OD10 | 调试定位声明 + numpy + 缩小输入 | numba/Cython（复查后） |
| OD11 | 三类用例绑定信任等级 | — |
| OD12 | VecAdd + cv-pipelining 双 kernel | — |
