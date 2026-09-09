#!/usr/bin/env bash
# 拉取 bishengir Python bindings 到本地，供 IR 接口层（T1.1+）开发与测试使用。
#
# 背景：绑定 .so 为 cp310 ABI，且构建树内含大量指向构建目录的符号链接。
# 直接 rsync 会得到 139 个悬空链接（本项目已踩过），故必须用 -L 解引用。
#
# 用法：
#   bash scripts/setup_bindings.sh [目标目录]
#     默认 <仓库>/.bindings（持久；家目录与 /var/tmp 常为只读，/tmp 会被清理）
#   export HIVM_SPEC_BINDINGS=<目标目录>           # 引擎据此定位
set -euo pipefail

DEST="${1:-${HIVM_SPEC_BINDINGS:-$(cd "$(dirname "$0")/.." && pwd)/.bindings}}"
REMOTE="${HIVM_SPEC_BUILD_HOST:-build}"
REMOTE_PATH="~/proj/AscendNPU-IR/build/tools/bishengir/bishengir/python_packages/bishengir/"

SSH_OPTS=(-F "$HOME/.ssh/config" -o ControlMaster=no -o ControlPath=none
          -o ConnectTimeout=10 -o BatchMode=yes)

echo "拉取 bindings：$REMOTE:$REMOTE_PATH → $DEST"
echo "（-L 解引用符号链接；约 246M，视网络需数十秒）"
rsync -azL -e "ssh ${SSH_OPTS[*]}" "$REMOTE:$REMOTE_PATH" "$DEST/"

echo
echo "校验："
DANGLING=$(find "$DEST" -type l | wc -l)
echo "  悬空链接：$DANGLING（应为 0）"
echo "  体积：$(du -sh "$DEST" | cut -f1)"
echo
echo "下一步："
echo "  export HIVM_SPEC_BINDINGS=$DEST"
echo "  <py3.10> -m pytest -m requires_bindings -q"
