# 改了主仓 HIVM pass 之后必做的验证（T1.8 纪律）

**规则**：凡修改 HIVM pass 实现代码（`bishengir/**` 下的 pass/transform 源文件），必须对
**受影响 kernel 跑对应 spec 工具，并把 verdict 贴进 PR 描述**。

工具的价值随"pass 改了却没人验"归零。机器门禁只校验"描述与语料的一致性"；pass 侧行为是否
被复验，由本条纪律约束。

## 最小执行集（按改动性质选，拿不准就全跑）

| 改动性质 | 必跑 | 判据 |
|---|---|---|
| 内存/分配类（plan-memory、multi-buffer、workspace） | `ub_occupancy` 全部 L2 目标 kernel | verdict 不得从 `OK/OVERFLOW` 恶化为**新增** `OVERFLOW` |
| 调度/同步类（cv-pipelining、sync-solver、preload） | `timeline` | 死锁/时序 verdict 变化必须解释 |
| op 定义 / ODS 变更 | `hivm-spec check` 全部描述 + 注册表绊线重测 | 覆盖率变化需在 PR 中说明 |

## 基线在前，改后对照（不可省略）

判据基线是**变更前的 verdict**：

```bash
# 1) 改动前留存基线
git stash
hivm-spec run kernel.mlir --anchor base.mlir --json /tmp/base.json
git stash pop

# 2) 改动后重跑
hivm-spec run kernel.mlir --anchor base.mlir --json /tmp/head.json

# 3) 对照（逐项 verdict 差异）
python - <<'PY'
import json
b=json.load(open('/tmp/base.json')); h=json.load(open('/tmp/head.json'))
for x,y in zip(b['checks'], h['checks']):
    if x['verdict']!=y['verdict']:
        print(f"{x['tool']}: {x['verdict']} -> {y['verdict']}")
PY
```

或直接：`scripts/compare-verdicts.py /tmp/base.json /tmp/head.json`

**只贴"改后绿灯"不贴基线，等于没有验证。** PR 里两者都要有。

## 什么算恶化

- `OK → OVERFLOW` / `OK → DEADLOCK` / `OK → MISMATCH`：**必修**，失败即停止并修复。
- verdict 从问题类变成 `COVERAGE_GAP`：可疑——你可能把可判语料改成了不可判语料（例如引入
  了动态 shape 或未建模 op）。需解释。
- `COVERAGE_GAP → OK`：改进，值得说明为什么以前不可判。
- 审计坐标变化（`spec_hash` / `engine_version` 变了）：差异可能来自描述或引擎变更，而非
  你的 pass。先对齐坐标再下结论。

## 不得做的事

- **不得**通过放松断言、改期望值、加 skip 或降低门禁让结果变绿；
- **不得**用 `lit`/FileCheck 更新期望值来"合法化"错误输出——这正是本工具存在的理由；
- **不得**把"环境不可用"报告成"验证失败"（退出码 4/5 与 1 的区分就是为此）。

## 语料从哪来

`specs/cases/corpus/{l0,l1,l2}/` 是已入库语料，`manifest.json` 记录来源文件 + 主仓 commit +
剥离方式。要新增：

```bash
python scripts/harvest_corpus.py <主仓源文件（相对 AscendNPU-IR 根）> --sections all --dry-run
```

它做**可审计剥离**：剥 lit 指令与整行注释、按 `// -----` 分节、`"some_op"` 占位符按类型定向
替换（标量→`arith.constant`，tensor→`tensor.empty`，memref→**提升为函数参数**）、每步记进
manifest、入库前经真实 bindings 严格解析（不开 `allow_unregistered`）。

**禁止**整批镜像主仓测试目录：会把 `COVERAGE_GAP` 变成噪声，使唯一的诚实降级信号失效。
**不引入** `expected-error` 负例作解析语料：那是主仓 verifier 的职责。
