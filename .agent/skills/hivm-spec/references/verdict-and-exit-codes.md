# 结论契约：verdict、退出码、JSON、审计坐标

> 权威实现：`src/hivm_spec/verdict.py`。本文件是它的操作视图。

## verdict 封闭枚举

| verdict | 退出码 | 含义 | 你该做什么 |
|---|---|---|---|
| `OK` | 0 | 验过了，在已覆盖范围内没发现问题 | 可以据此继续；但仍须转述信任级别 |
| `OVERFLOW` | 1 | 片上内存峰值超容量 | 按 diagnostics 的 `loc` + 贡献者列表改代码 |
| `DEADLOCK` | 1 | 结构性死锁（与展开界无关的确定性结论） | 按首现 wait 位置修同步 |
| `MISMATCH` | 1 | 等价验证发现发散 | 从**首个发散点**修，别去修下游受影响步 |
| `COVERAGE_GAP` | 4 | **没能验成**（未建模 op / 无锚点 / 输入不可推导 / 无可分析结构） | 补描述、补锚点、换语料；结论中如实写明未覆盖 |
| `UNTRUSTED_DESCRIPTION` | 5 | 判定依据（描述）本身不可信 | 先修描述 / 登记 drift，一切结论作废 |
| `PENDING` | 3 | 该阶段未实现 | 上报缺口，**禁止**静默跳过或报成"验证失败" |
| （环境不可用） | 2 | 仅 `verify`：前置门挡住，**未产出任何验证结论** | 按 doctor 的 remedy 修环境；这不是 IR 的问题 |

聚合优先级（`run` 综合结论取此序最高者）：
`UNTRUSTED_DESCRIPTION` > `COVERAGE_GAP` > `OVERFLOW`/`DEADLOCK`/`MISMATCH` > `OK`。

理由：描述不可信时后续分析无意义；覆盖不全时"没发现问题"不能读作"没有问题"。
空结果集也会被判 `COVERAGE_GAP` —— 一项都没跑就说"没问题"是自欺。

## 缺口类 verdict 不是问题，也不是通过

`verdict.is_problem` 只对 `OVERFLOW/DEADLOCK/MISMATCH` 为真；`is_gap` 对
`COVERAGE_GAP/UNTRUSTED_DESCRIPTION` 为真。二者在代码层就是分开的，转述时不得混用。

脚本里最常见的错误写法：

```bash
hivm-spec tool timeline k.mlir || echo "FAIL"     # ✗ 把 4/5「没验成」也当成失败
```

正确写法：

```bash
hivm-spec tool timeline k.mlir; rc=$?
case $rc in
  0) echo verified-clean ;;
  1) echo real-failure ;;
  4|5) echo not-verified ;;
  3) echo unimplemented ;;
esac
```

## 每条结论都带信任脚注

`trust != anchored` 时，渲染输出必然带这一行：

```
  ⚠️ 信任级别：provisional —— 本结论基于**未经对拍验证**的描述，不可作为最终依据
```

现网全部描述为 `provisional`。这是刻意的：图形化输出天然显得权威，脚注是对该认知陷阱的
对策。**你引用结论时必须把这行带上**，否则就是把 provisional 冒充成 anchored。

信任升级路径：`provisional` → `cross-validated` → `anchored`，须附对拍证据；逃生舱
（`host_fn`）条目封顶 `provisional`，永不可升级。

## 审计坐标（可复现三件套）

每条结论尾部：`spec_hash` + `engine_version` + `ir_fingerprint`。三者缺一，历史结论不可
复现。跨版本/跨描述对比前先核对这三个值——不一致时差异可能来自描述变更，而非 IR 变更。

## JSON 输出（供 agent 消费）

```bash
hivm-spec tool equivalence after.mlir --anchor before.mlir --json r.json
hivm-spec run kernel.mlir --json run.json
```

单工具 JSON 字段：

```jsonc
{
  "tool": "equivalence",
  "verdict": "MISMATCH",
  "spec_hash": "sha256:…",            // 描述指纹
  "engine_version": "0.1.0",          // 引擎版本
  "ir_fingerprint": "…",              // 输入 IR 的 VIR 指纹
  "trust": "provisional",
  "details": { "outcome": …, "compared_steps": …, "first_divergence": … },
  "diagnostics": [
    { "severity": "error", "rule": "equivalence/first-divergence",
      "message": "…", "loc": "after.mlir:12:5", "extra": { … } }
  ]
}
```

`run --json` 多一层：`{"verdict", "exit_code", "orchestrator_version", "checks": [上面结构…]}`。

- 序列化是确定性的（`sort_keys=True`、固定缩进）——同输入两次生成逐字节一致，可直接入
  golden 比对。
- `diagnostics[].loc` 恒应存在（FR5：不可定位的诊断对 agent 没有行动价值）。缺 `loc` 时
  把它当缺陷上报，不要自行猜位置。
- `rule` 是回溯到实现的锚：`timeline/unpaired-wait`、`occupancy/overflow`、
  `equivalence/undeducible-input`、`pairing/vacuous`、`uninit/vacuous`、
  `run/*`（编排层留下的缺口记录）。

## 一条实测样本（可直接对照）

```
[ub_occupancy] OVERFLOW
  ⚠️ 信任级别：provisional —— 本结论基于**未经对拍验证**的描述，不可作为最终依据
  error [occupancy/overflow] @ specs/cases/corpus/l0/ub_overflow_injected.mlir:8:13:
    space 'ub' 峰值 524288B 超出容量 196608B（超 327680B）；主要贡献者：buf3(262144B), buf4(262144B)
  审计坐标：spec_hash=sha256:3fe62d40fc0c… engine=0.1.0 ir=5889665b5e5e0b39…
```

读法：溢出空间（ub）→ 峰值 vs 容量 → 超量 → **贡献者**（这才是能动手的地方）→ 行号。
