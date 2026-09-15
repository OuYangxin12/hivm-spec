#!/usr/bin/env bash
# 定位"哪个 pass 引入了毛病"：对每份 dump 跑同一个检查，输出 verdict 表。
# 用法: locate-introducing-pass.sh <check> dump_01.mlir dump_02.mlir ...
#       locate-introducing-pass.sh <check> 'dump_*.mlir'   (glob 需加引号)
#
# 工具契约（D12）：工具只回答"这一份 IR 有没有这个毛病"。转储与切分是你的工作：
#   bishengir-opt in.mlir --pass-a --pass-b --mlir-print-ir-after-all 2> dumps.txt
# 然后用 banner "// -----// IR Dump After <Pass> (<flag>)" 切成单份 .mlir。
#
# 退出码：0=全部 OK；1=至少一个真问题；4=存在不可判定点（定位链断开）。
set -uo pipefail

HIVM_SPEC="${HIVM_SPEC:-hivm-spec}"
: "${HIVM_SPEC_BINDINGS:?必须先 export HIVM_SPEC_BINDINGS=<bindings 目录>}"

if [ $# -lt 2 ]; then
  sed -n '2,12p' "$0"
  exit 2
fi

check="$1"
shift

# 展开可能未展开的 glob
files=()
for a in "$@"; do
  if [ -f "$a" ]; then files+=("$a")
  else for g in $a; do [ -f "$g" ] && files+=("$g"); done
  fi
done
[ ${#files[@]} -eq 0 ] && { echo "没有匹配的 .mlir 文件：$*" >&2; exit 2; }

verdict_of() {  # verdict 词在渲染首行 "[<tool>] <VERDICT>"
  local v="$1"
  case "$v" in
    0) echo OK ;;
    1) echo PROBLEM ;;
    4) echo GAP ;;
    5) echo UNTRUSTED ;;
    3) echo PENDING ;;
    *) printf 'UNKNOWN(%s)\n' "$v" ;;
  esac
}

worst=0
first_hit=""
printf '%-58s %-8s %s\n' "IR" "VERDICT" "NOTE"
printf '%s\n' "--------------------------------------------------------------------------"

for f in "${files[@]}"; do
  out="$($HIVM_SPEC tool "$check" "$f" 2>&1)"; rc=$?
  note="$(printf '%s\n' "$out" | grep -m1 -E '^\s+(error|warning) ' | sed 's/^ *//' | cut -c1-72)"
  v="$(verdict_of "$rc")"
  printf '%-58s %-8s %s\n' "$(basename "$f")" "$v" "${note:--}"
  case "$rc" in
    1) [ -z "$first_hit" ] && first_hit="$f"; worst=1 ;;
    4|5) [ "$worst" -eq 0 ] && worst=4 ;;
  esac
done

echo
if [ -n "$first_hit" ]; then
  echo "首恶候选（时间序最早的异常 verdict）：$first_hit"
elif [ "$worst" -eq 4 ]; then
  echo "定位链断开：存在不可判定点（COVERAGE_GAP/UNTRUSTED_DESCRIPTION）。"
  echo "补描述或补锚点后重跑；不得把不可判定点跨过当作 OK。"
else
  echo "全部 OK。注意：OK 仍受信任级别限制（现网全部 provisional，不可当门禁）。"
fi
exit "$worst"
