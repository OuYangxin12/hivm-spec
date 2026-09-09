#!/usr/bin/env bash
# hivm-spec 一键导览：四步看完全部现有能力（约 5 秒）。
#
# 用法：
#   bash scripts/demo.sh
#
# 前置（首次）：
#   bash scripts/setup_bindings.sh     # 拉取 MLIR 绑定到 .bindings/（约 246M）
#   export UV_PYTHON_INSTALL_DIR="$PWD/.uvpython"   # 解释器须持久化，勿落 /tmp
#   uv venv .venv310 --python 3.10 && uv pip install -e '.[test]' --python .venv310/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv310/bin/python}"
export HIVM_SPEC_BINDINGS="${HIVM_SPEC_BINDINGS:-$PWD/.bindings}"
CFG=$(mktemp /tmp/hivm-demo-XXXXXX.json)
trap 'rm -f "$CFG"' EXIT

hr() { printf '\n\033[1;36m== %s ==\033[0m\n' "$1"; }

hr "第 1 步：描述静态检查（写错即拦，不生成带洞的工具）"
"$PY" -m hivm_spec check specs/toy.py specs/cv.py

hr "第 2 步：生成配置文档（描述 → 工具的单一事实来源）"
"$PY" -m hivm_spec gen specs/toy.py specs/cv.py -o "$CFG"
echo "→ 配置文档：$CFG（ops 共 $(python3 -c "import json;print(len(json.load(open('$CFG'))['ops']))" 2>/dev/null || echo '?') 个）"

hr "第 3 步：注入缺陷的样例 —— 工具抓出 UB 溢出（AC1）"
"$PY" -m hivm_spec tool ub_occupancy -c "$CFG" specs/cases/corpus/l0/ub_overflow_injected.mlir || true

hr "第 4 步：真实目标 kernel（主仓 cv-pipelining preload，19 份 L2 之一）"
"$PY" -m hivm_spec tool ub_occupancy -c "$CFG" \
  specs/cases/corpus/l2/cv-pipelining-preload-full-a5_preload_workspace.mlir --json /tmp/hivm-demo-result.json || true
echo
echo "完整结论（JSON，含占用曲线与逐条 diagnostics）已写 /tmp/hivm-demo-result.json"
echo "试试用 jq 看峰值：jq '.details.spaces.ub.peak_bytes' /tmp/hivm-demo-result.json"
