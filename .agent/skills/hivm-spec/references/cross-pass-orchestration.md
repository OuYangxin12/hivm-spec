# 跨 pass 定位：由你编排，不由工具负责

**工具只回答"这一份 IR 有没有这个毛病"。**"哪个 pass 引入了毛病"是**你的**工作，用既有
能力组合完成。不要为此扩展工具契约（D12）——工具不吃 pass 序列、不解析
`--print-ir-*` 转储、不建模 pass 顺序、不跨调用保持状态。

## 标准回路：二分/线性定位首恶 pass

```bash
# 1) 转储各 pass 后的 IR（MLIR 标准能力，不要指望工具帮你做这步）
bishengir-opt in.mlir --pass-a --pass-b --pass-c --mlir-print-ir-after-all 2> dumps.txt

# 2) 按 pass banner 切成单份 .mlir（banner 形如：// -----// IR Dump After <Pass> (<flag>)）
#    用你手上的任何切分手段；关键是每份都能被 bindings 独立解析

# 3) 逐份调用同一检查，第一个非 OK/非缺口 verdict 即首恶 pass
for f in dump_*.mlir; do
  hivm-spec tool timeline "$f" > "/tmp/$(basename "$f").log" 2>&1 || echo "HIT $f ($?)"
done
```

用 `scripts/locate-introducing-pass.sh` 可直接跑这个循环并把 verdict 汇总成表。

判定纪律：

- **第一个异常 verdict 即首恶 pass**（时间序上的最早者）。后面全红是传播后果。
- 中间某份报 `COVERAGE_GAP` → 该点**不可判**，不能当作 OK 跨过，也不能当作 MISMATCH。
  定位链在此断开，要么补描述要么换一个语料。
- 语料若已不可解析（pass 输出本身非法），那是解析/前置问题，先回 `bishengir-opt` 侧。

## 三类检查各自独立调用，结论正交

不要指望一次调用覆盖三类问题：

```bash
hivm-spec tool timeline      after.mlir                            # 死锁/时序：只看 after
hivm-spec tool ub_occupancy  after.mlir                            # 片上内存：只看 after
hivm-spec tool equivalence   after.mlir --anchor before.mlir       # 语义等效：after+before
```

## 基线在前，改后对照

判断"我这个 pass 有没有让事情变坏"，必须有**改动前的 verdict** 作对照：

```bash
git stash && hivm-spec run kernel.mlir --anchor base.mlir --json /tmp/base.json && git stash pop
hivm-spec run kernel.mlir --anchor base.mlir --json /tmp/head.json
```

对比 verdict 变化方向：`OK → OVERFLOW` 是恶化（必须修）；
`COVERAGE_GAP → OK` 是改进（值得说明为何以前不可判）。
只贴"改后绿灯"不贴基线，等于没有验证。

## 不要做的事

| 诱惑 | 为什么不行 |
|---|---|
| 把整份 all-pass dump 直接喂给工具 | 位置参数恒为一份 IR；且重复 IR 结构与 MLIR 既有能力重复 |
| 用工具报"pass X 有问题" | 工具没有 pass 概念；是"你转储 + 你切分 + 你逐份调用"得出的 |
| 在两次调用间期待状态继承 | 无跨调用状态，每次都是独立判定 |
| 为了定位而给工具加 pass 序列入参 | 契约随主仓 pipeline 膨胀（冲击 NFR4），这是明令禁止的扩展轴 |
