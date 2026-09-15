#!/usr/bin/env bash
# 安装 hivm-spec 的纯 Python 核心层，然后探测 bindings。
#
# 用法:
#   install.sh [--repo <hivm-spec 仓路径>] [--git <git+https://…>] [--venv <目录>]
#   环境变量 HIVM_SPEC_REPO / HIVM_SPEC_GIT 亦可。
#
# 为什么没有"一条 pip install hivm-spec"就完事：
#   核心层是纯 Python，装得上；但六类检查依赖的 bishengir bindings 是**主仓构建树
#   产物**（cp310 ABI、约 246M、上游未发布 wheel），装不了也带不走。所以本脚本
#   做两件事：装核心层 → 用 bootstrap.py 在这台机器上**找** bindings。
#   找不到就如实报告能力边界（doctor/check/gen 可用，run/tool/verify 不可用）。
#
# 退出码: 0=核心层就绪（bindings 可能缺，见输出）; 1=核心层安装失败
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(cd "$here/.." && pwd)"
venv="${VENV:-$PWD/.venv-hivm-spec}"
repo="${HIVM_SPEC_REPO:-}"
git_url="${HIVM_SPEC_GIT:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --git) git_url="$2"; shift 2 ;;
    --venv) venv="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 1 ;;
  esac
done

say() { printf '%s\n' "$*"; }

# --- 1. 找 3.10 解释器（核心层 >=3.10；要跑六类检查则必须恰好 3.10） ----------
py=""
for c in "${UV_PYTHON:-}" python3.10 python3 python; do
  [ -z "$c" ] && continue
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c 'import sys;raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)' 2>/dev/null; then
    py="$c"; break
  fi
done
if [ -z "$py" ]; then
  say "✗ 未找到 Python 3.10。"
  say "  bindings 是 cp310 ABI——这不是可选偏好而是硬约束（D7）。"
  say "  装一个（uv 用户）：uv python install 3.10"
  exit 1
fi
say "1/4 解释器：$py（$("$py" -c 'import platform;print(platform.python_version())')）"

# --- 2. 核心层安装 ----------------------------------------------------------
if [ -z "$repo" ] && [ -z "$git_url" ]; then
  # 从 skill 所在位置倒推：仓内 skill 就在 <repo>/.agent/skills/hivm-spec/
  guess="$(cd "$SKILL_DIR/../../.." 2>/dev/null && pwd || true)"
  if [ -f "$guess/pyproject.toml" ] && grep -q 'name = "hivm-spec"' "$guess/pyproject.toml" 2>/dev/null; then
    repo="$guess"
  fi
fi

"$py" -m venv "$venv" >/dev/null 2>&1 || "$py" -m virtualenv "$venv" >/dev/null 2>&1
vp="$venv/bin/python"
[ -x "$vp" ] || { say "✗ 虚拟环境创建失败：$venv"; exit 1; }
"$vp" -m pip install -q --upgrade pip >/dev/null 2>&1 || true

if [ -n "$repo" ]; then
  say "2/4 核心层：从本地仓安装 $repo"
  "$vp" -m pip install -q -e "$repo" || { say "✗ 核心层安装失败"; exit 1; }
elif [ -n "$git_url" ]; then
  say "2/4 核心层：从 git 安装 $git_url"
  "$vp" -m pip install -q "$git_url" || { say "✗ 核心层安装失败"; exit 1; }
else
  say "2/4 核心层：**未安装**——找不到 hivm-spec 仓，且 PyPI 上无 `hivm-spec` 包。"
  say "      给它一个来源再来：install.sh --repo <本地仓> 或 --git git+https://…"
fi

# --- 3/4 生成配置文档（脱离仓目录调用时必需） ---------------------------------
# 缺省配置路径 `build/config.json` 与缺省描述都是 **CWD 相对**，装好后从别的
# 目录调用就找不到。gen 是确定性的（同输入 byte 级一致），所以可以安全地预先
# 生成一份并用 HIVM_SPEC_CONFIG 指过去。
cfg="$venv/hivm-spec-config.json"
if [ -n "$repo" ] && [ -f "$repo/specs/toy.py" ] && [ -x "$vp" ]; then
  (cd "$repo" && "$vp" -m hivm_spec gen specs/toy.py specs/cv.py -o "$cfg") >/dev/null 2>&1 \
    && say "     配置文档：$cfg" \
    || say "     ⚠️ 配置文档生成失败——调用时用 -c 显式指定"
fi

# --- 3. 定位 bindings（不猜、不假装可用） ------------------------------------
say "4/4 bindings 探测："
"$vp" "$here/bootstrap.py"
brc=$?

say ""
case $brc in
  0) say "✅ 全功能就绪。一句话入口（任意目录可用）："
     say "     export HIVM_SPEC_CONFIG=$cfg"
     say "     $vp -m hivm_spec verify <kernel>.mlir --anchor <before>.mlir" ;;
  1) say "⚠️ 核心层就绪，但 bindings 校验未通过 → 六类检查（run/tool/verify）不可用。" ;;
  2) say "⚠️ 本机未找到 bindings → 只有核心层可用：doctor / check / gen。" ;;
esac
say "   能力边界的权威判定：$vp -m hivm_spec doctor"
exit 0
