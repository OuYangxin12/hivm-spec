#!/usr/bin/env bash
# 重跑本 skill 引用的实测样本，校验 skill 里的断言是否仍然成立。
# 用法: verify-skill.sh [repo-root]        （默认从脚本位置向上找 specs/cases/corpus）
#
# 退出码: 0=全部断言成立；1=有断言失效（skill 需要更新）；2=环境不可用。
# 何时跑：改了 specs/ 描述、引擎版本、或要对外引用覆盖率数字之前。
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="${1:-$(cd "$here/../../../.." && pwd)}"
cd "$repo" || exit 2

: "${HIVM_SPEC_BINDINGS:?必须先 export HIVM_SPEC_BINDINGS=<bindings 目录>}"
HIVM_SPEC="${HIVM_SPEC:-hivm-spec}"
PY="${PYTHON:-python3}"

L=specs/cases/corpus/l0
fail=0
note() { printf '%-52s %s\n' "$1" "$2"; }

# --- 环境 ---
if ! out="$($HIVM_SPEC doctor 2>&1)"; then
  echo "$out"
  echo "环境不可用（doctor 报 MISSING），先按 references/environment.md 修环境。"
  exit 2
fi
echo "$out" | grep -q "ml_dtypes" || note "doctor 输出形态" "变了（期望含 ml_dtypes 探测项）"

# --- 断言 1：注入溢出语料必须 OVERFLOW(1) ---
$HIVM_SPEC tool ub_occupancy "$L/ub_overflow_injected.mlir" --no-chart >/tmp/vs1.txt 2>&1; rc=$?
[ $rc -eq 1 ] && note "ub_overflow_injected → OVERFLOW(1)" "OK" \
              || { note "ub_overflow_injected → OVERFLOW(1)" "失效 rc=$rc"; fail=1; }
grep -q "贡献者" /tmp/vs1.txt && note "OVERFLOW 诊断含贡献者列表" "OK" \
                             || { note "OVERFLOW 诊断含贡献者列表" "失效"; fail=1; }

# --- 断言 2：无 set 的 wait 必须 DEADLOCK(1) 且声明与展开界无关 ---
$HIVM_SPEC tool timeline "$L/waits_before_sets_deadlock.mlir" >/tmp/vs2.txt 2>&1; rc=$?
[ $rc -eq 1 ] && note "waits_before_sets_deadlock → DEADLOCK(1)" "OK" \
              || { note "waits_before_sets_deadlock → DEADLOCK(1)" "失效 rc=$rc"; fail=1; }
grep -q "与展开界无关" /tmp/vs2.txt && note "DEADLOCK 为结构性判定（与展开界无关）" "OK" \
                                   || { note "DEADLOCK 为结构性判定" "失效"; fail=1; }

# --- 断言 3：计数配平但同泳道错序 → timeline 死锁、sync_pairing 却过（正交性立论）---
$HIVM_SPEC tool timeline "$L/balanced_misordered_same_lane.mlir" >/dev/null 2>&1; rc_tl=$?
$HIVM_SPEC tool sync_pairing "$L/balanced_misordered_same_lane.mlir" >/dev/null 2>&1; rc_sp=$?
if [ $rc_tl -eq 1 ] && [ $rc_sp -eq 0 ]; then
  note "配平但错序：timeline 死锁 / sync_pairing 过" "OK"
else
  note "配平但错序：timeline 死锁 / sync_pairing 过" "失效 tl=$rc_tl sp=$rc_sp"; fail=1
fi

# --- 断言 4：等价验证的锚点契约 ---
cp "$L/loop_load_add_store.mlir" /tmp/vs_before.mlir
$HIVM_SPEC tool equivalence /tmp/vs_before.mlir --anchor /tmp/vs_before.mlir >/dev/null 2>&1
[ $? -eq 0 ] && note "equivalence 自比（显式同档锚点）→ OK(0)" "OK" \
             || { note "equivalence 自比（显式同档锚点）→ OK(0)" "失效"; fail=1; }
$HIVM_SPEC tool equivalence "$L/loop_load_add_store.mlir" >/dev/null 2>&1; rc=$?
[ $rc -eq 4 ] && note "equivalence 缺 --anchor → COVERAGE_GAP(4)" "OK" \
              || { note "equivalence 缺 --anchor → COVERAGE_GAP(4)" "失效 rc=$rc"; fail=1; }

sed 's/hivm\.hir\.vadd/hivm.hir.vmul/' /tmp/vs_before.mlir > /tmp/vs_after.mlir
if cmp -s /tmp/vs_before.mlir /tmp/vs_after.mlir; then
  note "语料含可替换的 vadd（MISMATCH 断言前提）" "失效（语料变了）"; fail=1
else
  $HIVM_SPEC tool equivalence /tmp/vs_after.mlir --anchor /tmp/vs_before.mlir >/tmp/vs3.txt 2>&1
  rc=$?
  [ $rc -eq 1 ] && grep -q "首个发散" /tmp/vs3.txt \
    && note "注入 vadd→vmul → MISMATCH(1) + 首个发散点" "OK" \
    || { note "注入 vadd→vmul → MISMATCH(1) + 首个发散点" "失效 rc=$rc"; fail=1; }
fi

# --- 断言 5：run 无锚点必须留一条 equivalence 缺口（不许静默消失）---
$HIVM_SPEC run "$L/../l1/hivm-pipeline.mlir" >/tmp/vs4.txt 2>&1; rc=$?
[ $rc -eq 4 ] && grep -q "equivalence" /tmp/vs4.txt && grep -q "未能完成验证" /tmp/vs4.txt \
  && note "run 无锚点 → 综合 COVERAGE_GAP(4) 且 equivalence 留痕" "OK" \
  || { note "run 无锚点 → 综合 COVERAGE_GAP(4) 且 equivalence 留痕" "失效 rc=$rc"; fail=1; }

# --- 断言 6：信任脚注恒在（provisional 不得被冒充）---
grep -q "信任级别" /tmp/vs1.txt && note "结论带信任脚注" "OK" || { note "结论带信任脚注" "失效"; fail=1; }

# --- 断言 7：verify 的前置门必须用独立退出码（2），不得借道 4 ---
out_noenv="$($HIVM_SPEC verify "$L/ub_overflow_injected.mlir" 2>&1)"; :
env -u HIVM_SPEC_BINDINGS $HIVM_SPEC verify "$L/ub_overflow_injected.mlir" >/tmp/vs5.txt 2>&1; rc=$?
[ $rc -eq 2 ] && grep -q "未产出任何验证结论" /tmp/vs5.txt \
  && note "verify 无 bindings → 2（而非 4），且不产出 verdict" "OK" \
  || { note "verify 无 bindings → 2（而非 4），且不产出 verdict" "失效 rc=$rc"; fail=1; }

# --- 断言 8：verify 的能力自述必须在场 ---
$HIVM_SPEC verify "$L/../l1/hivm-pipeline.mlir" --json /tmp/vs6.json >/tmp/vs6.txt 2>&1
grep -q "没验成" /tmp/vs6.txt && grep -q "self_description" /tmp/vs6.json \
  && note "verify 输出含「没验成」计数与 JSON self_description" "OK" \
  || { note "verify 输出含「没验成」计数与 JSON self_description" "失效"; fail=1; }

# --- 断言 9：bootstrap 必须真探到 bindings（真 import，不是看目录名）---
"$PY" "$here/bootstrap.py" >/tmp/vs7.txt 2>&1; rc=$?
[ $rc -eq 0 ] && grep -q "import 校验" /tmp/vs7.txt \
  && note "bootstrap.py 定位并通过 import 校验" "OK" \
  || { note "bootstrap.py 定位并通过 import 校验" "失效 rc=$rc（本机可能无 bindings——那是能力边界，非断言错误）"; }

# --- 快照数字：重新测量，不写死 ---
echo
echo "=== 等价可判率（现场重测，别引用 skill 里的静态数字）==="
tot=0; ok=0
for f in specs/cases/corpus/*/*.mlir; do
  tot=$((tot+1))
  $HIVM_SPEC tool equivalence "$f" --anchor "$f" >/dev/null 2>&1 && ok=$((ok+1))
done
echo "可比语料：$ok/$tot（自比口径，仅用于测量可判性，不作为等价证据）"

echo
[ $fail -eq 0 ] && echo "✅ skill 断言全部成立。" || echo "❌ 有断言失效：请更新 SKILL.md / references/。"
exit $fail
