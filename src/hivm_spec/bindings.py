"""bishengir bindings 引导（框架 §6 强制条款）。

**为什么必须有这个模块**：主仓 bindings 的 `_mlir_libs` 站点初始化只探测
`_mlirRegisterEverything`，不探测带前缀的 `_bishengirRegisterEverything`，
导致 hivm/hfusion/hacc 方言**不被自动注册**。后果是 `Module.parse` 把 hivm op
当作未注册方言而报错（或在 allow-unregistered 下静默降级为通用 op，那更糟——
会让工具在"看不懂 IR"的情况下给出貌似正常的结论）。

故引擎必须在 `Context` 创建后显式调用 `register_dialects(ctx)`。这是 V1 门禁
发现的主仓缺口，已在框架 §6 立为强制条款。

**本模块是核心层与 bindings 的唯一接触点之一**（另一处是 `ir_engine`）。
核心层其余部分不得 import bishengir（D7）。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "BindingsError",
    "BindingsHandle",
    "bindings_available",
    "load_bindings",
]

#: 环境变量：bindings 树的位置（由 scripts/setup_bindings.sh 拉取）
ENV_BINDINGS_PATH = "HIVM_SPEC_BINDINGS"

#: cp310 ABI 硬约束——绑定 .so 只能被 CPython 3.10 加载。
REQUIRED_PYTHON = (3, 10)


class BindingsError(RuntimeError):
    """bindings 不可用。

    区别于 `VIRError`（契约违规）与 `SpecError`（描述错误）：这是**环境**问题，
    FR7 要求把它与"IR 有问题"清楚分开——否则 agent 会把环境故障误读为验证失败。
    """


@dataclass(frozen=True, slots=True)
class BindingsHandle:
    """已完成注册的 bindings 句柄。"""

    ir: Any
    context: Any
    path: str

    def parse_module(self, text: str) -> Any:
        """解析 MLIR 文本。

        刻意**不**开启 `allow_unregistered_dialects`：宁可解析失败，也不要把
        hivm op 降级为通用 op——后者会让工具在看不懂 IR 的情况下给出貌似
        正常的结论（FR6 反自欺）。
        """
        return self.ir.Module.parse(text, self.context)


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get(ENV_BINDINGS_PATH)
    if env:
        paths.append(Path(env))
    # 约定俗成的本地位置（scripts/setup_bindings.sh 的默认目标）
    paths.append(Path("/tmp/bindings"))
    return paths


def _find_bindings_root() -> Path | None:
    for p in _candidate_paths():
        if (p / "bishengir" / "ir.py").is_file():
            return p
    return None


def bindings_available() -> bool:
    """能否加载 bindings（不抛异常，供 pytest 跳过判定使用）。"""
    if sys.version_info[:2] != REQUIRED_PYTHON:
        return False
    return _find_bindings_root() is not None


def load_bindings() -> BindingsHandle:
    """加载并**注册**方言，返回可用句柄。

    失败时抛 `BindingsError` 且消息里说明是环境问题——不允许静默返回 None，
    否则调用方容易把"没有 bindings"当成"IR 没问题"。
    """
    if sys.version_info[:2] != REQUIRED_PYTHON:
        raise BindingsError(
            f"bindings 为 cp310 ABI，当前解释器 "
            f"{sys.version_info.major}.{sys.version_info.minor} 不兼容"
            "（环境问题，非 IR 问题）"
        )

    root = _find_bindings_root()
    if root is None:
        raise BindingsError(
            f"未找到 bindings 树。请设置 {ENV_BINDINGS_PATH} 或运行 "
            "scripts/setup_bindings.sh（环境问题，非 IR 问题）"
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from bishengir import ir
    except ImportError as exc:
        raise BindingsError(f"导入 bishengir.ir 失败：{exc}（环境问题）") from exc

    ctx = ir.Context()

    # —— 框架 §6 强制条款：显式注册带前缀的方言 ——
    try:
        from bishengir._mlir_libs import (
            _bishengirRegisterEverything as _reg,
        )
    except ImportError as exc:
        raise BindingsError(
            f"导入 _bishengirRegisterEverything 失败：{exc}。"
            "该模块是 hivm 方言注册的唯一入口（主仓站点初始化不会自动调用它）"
        ) from exc

    _reg.register_dialects(ctx)

    # 自检：注册没生效就必须立刻失败，而不是等到解析时才暴露
    if not ctx.is_registered_operation("hivm.hir.vadd"):
        raise BindingsError(
            "方言注册后 hivm.hir.vadd 仍未注册——bootstrap 失效。这正是框架 §6 强制条款要防的情形"
        )

    return BindingsHandle(ir=ir, context=ctx, path=root_str)
