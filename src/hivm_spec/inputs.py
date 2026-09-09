"""输入生成策略（T3.4，FR8 确定性）。

## 为什么要固定种子

差分对拍的结论必须可复现：报告里说"用例 #3 在 vadd@i2 发散"，读者得能用同一
组输入重跑出同一结果。所以输入由 `(seed, dtype, shape, 名字)` 完全决定——
同种子逐字节一致（FR8），且**跨进程、跨平台一致**。

`np.random.default_rng(seed)`（PCG64）满足这点；`np.random.seed` 那套全局状态
不满足——任何一处别的代码抽一次随机数就会移动全局游标，让"同种子"不再同结果。
故本模块只用显式 Generator，不碰全局随机状态。

## 为什么按名字派生子种子

同一个 kernel 有多个输入（%a、%b、%gm…）。若都从一个 Generator 顺序抽取，
那么"输入的生成顺序"就成了结论的隐藏依赖：换个遍历顺序、多一个参数，所有输入
的值都变了，历史结论无法复现。故每个输入按 `名字` 派生独立子种子——
新增/删除一个输入不会扰动其余输入的值。

## 缩小 tiling

真实 kernel 的 shape 常是 256/512 起步，逐元素跑完既慢又对定位无益（D10 预算：
等价 ≤5min/kernel）。缩小 tiling 指**按比例压缩 shape**，保留结构（维数、
广播关系、是否对齐）而减少元素数。

关键约束：缩小后必须**如实标注**。"在缩小 tiling 下未发现问题"和"已验证"是两
回事——前者是 M3 能给的结论，后者不是（FR6）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from hivm_spec.numeric import NumericSupportError, dtype_of

__all__ = [
    "DEFAULT_SEED",
    "InputSpec",
    "InputStrategy",
    "TilingPolicy",
    "derive_seed",
    "parse_shape_text",
    "specs_from_module",
]

#: 默认主种子。写死而非取当前时间：时间戳会让"同一份描述 + 同一份 IR"每次跑出
#: 不同输入，报告里的用例编号也就失去意义。
DEFAULT_SEED = 20260101


def derive_seed(master: int, name: str) -> int:
    """按输入名派生稳定子种子。

    用 sha256 而非 `hash()`：Python 的 str hash 带进程级随机盐（PYTHONHASHSEED），
    同一名字在两次运行里会得到不同值——那会直接破坏 FR8。
    """
    digest = hashlib.sha256(f"{master}:{name}".encode()).digest()
    # 取 8 字节转无符号整数；numpy 的 SeedSequence 接受任意大小的非负整数
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True, slots=True)
class TilingPolicy:
    """缩小 tiling 的策略。

    `max_elements` 是**每个输入**的元素上限。压缩按最后一维优先（内层维通常是
    向量化维，压它最不改变结构语义），必要时再压外层。
    """

    max_elements: int = 256
    #: 每一维至少保留多少元素——压到 1 会让广播/边界语义失真
    min_dim: int = 2
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_elements < 1:
            raise NumericSupportError("max_elements 必须为正")
        if self.min_dim < 1:
            raise NumericSupportError("min_dim 必须为正")

    def shrink(self, shape: tuple[int, ...]) -> tuple[tuple[int, ...], str]:
        """返回 `(压缩后 shape, 说明)`；未压缩时说明为空串。

        说明非空即意味着结论必须带"在缩小 tiling 下"的限定语——这是 FR6 的要求，
        不是可选的礼貌用语。
        """
        if not self.enabled or not shape:
            return shape, ""
        total = 1
        for d in shape:
            total *= d
        if total <= self.max_elements:
            return shape, ""

        dims = list(shape)
        # 从最内层维往外压：内层通常是向量化维，压它对结构语义影响最小
        for i in range(len(dims) - 1, -1, -1):
            if total <= self.max_elements:
                break
            keep = max(self.min_dim, 1)
            while dims[i] > keep and total > self.max_elements:
                # 折半而非直接设成上限：保留原 shape 的 2 的幂特性，
                # 避免把对齐的 shape 压成非对齐从而改变 padding 语义
                total //= dims[i]
                dims[i] = max(keep, dims[i] // 2)
                total *= dims[i]
        shrunk = tuple(dims)
        if shrunk == shape:
            return shape, ""
        return (
            shrunk,
            f"shape {shape} → {shrunk}（缩小 tiling，元素上限 {self.max_elements}）",
        )


@dataclass(frozen=True, slots=True)
class InputSpec:
    """一个输入的声明：名字 + dtype + shape。"""

    name: str
    dtype: str = "f32"
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise NumericSupportError("输入必须有名字——匿名输入无法在报告里指认")


@dataclass(slots=True)
class InputStrategy:
    """固定种子输入生成器。

    可变（非 frozen）仅为累积 `notes`；值的生成本身不依赖调用顺序。
    """

    seed: int = DEFAULT_SEED
    tiling: TilingPolicy = field(default_factory=TilingPolicy)
    #: 缩小 tiling 等如实标注（随结论输出）
    notes: list[str] = field(default_factory=list)

    def generate(self, spec: InputSpec) -> Any:
        """生成一个具体输入（numpy 数组）。

        取值范围刻意避开"全是小整数"：那类输入会让很多真实缺陷（舍入、溢出、
        累加顺序）碰巧算对，从而放过缺陷（NFR2 反向的风险——假阴性）。
        """
        import numpy as np

        shape, note = self.tiling.shrink(spec.shape)
        if note and note not in self.notes:
            self.notes.append(f"{spec.name}: {note}")

        rng = np.random.default_rng(derive_seed(self.seed, spec.name))
        dt = dtype_of(spec.dtype)

        if spec.dtype == "i1":
            data = rng.integers(0, 2, size=shape)
        elif spec.dtype.startswith("i"):
            # 整数取小范围：避免乘法在窄 dtype 上必然溢出，那会把每个用例都变成
            # 溢出用例，反而掩盖真正要查的语义差异
            info = np.iinfo(dt)
            lo = max(int(info.min), -128)
            hi = min(int(info.max), 127)
            data = rng.integers(lo, hi + 1, size=shape)
        else:
            # 浮点：非整数、跨量级、含负值——最容易暴露舍入与容差问题
            data = rng.uniform(-2.0, 2.0, size=shape)
        return data.astype(dt)

    def generate_all(self, specs: tuple[InputSpec, ...]) -> dict[str, Any]:
        """批量生成。名字派生子种子，故与遍历顺序无关。"""
        return {s.name: self.generate(s) for s in specs}

    @property
    def shrunk(self) -> bool:
        """是否发生过缩小——结论必须据此加限定语（FR6）。"""
        return bool(self.notes)

    def provenance(self) -> dict[str, Any]:
        """输入来源摘要，随结论输出（FR5 可诊断性：读者能重放）。"""
        return {
            "seed": self.seed,
            "tiling_max_elements": self.tiling.max_elements if self.tiling.enabled else None,
            "shrunk": self.shrunk,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# 从 VIR 推导入口输入（T3.4 与 T3.2 的接缝）
# ---------------------------------------------------------------------------

#: shape 文本里的 dtype 后缀 → 规范 dtype 名。
#:
#: 只收**已知**映射：遇到没列出的 dtype 就登记覆盖缺口，而不是默认成 f32。
#: 默认成 f32 会让 bf16/f8 kernel 用更高精度算出"通过"，那是假阴性（FR6）。
_DTYPE_SUFFIX = {
    "f32": "f32",
    "f16": "f16",
    "bf16": "bf16",
    "f64": "f64",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "i1": "i1",
}


def parse_shape_text(text: str) -> tuple[tuple[int, ...] | None, str, str]:
    """解析 `VAlloc.shape_text`（如 `"256xf32"`、`"4x16xbf16"`、`"?x16xf16"`）。

    返回 `(shape, dtype, 无法解析的原因)`。任一环节不确定就返回原因，让调用方
    降级——**动态 shape 不猜具体值**：猜 1 会让边界语义失真，猜 256 会让预算
    失真，而两者都会把"不知道 shape"伪装成"验证过了"。
    """
    parts = text.split("x")
    if len(parts) < 1 or not parts[-1]:
        return None, "", f"shape 文本无法解析：{text!r}"
    dtype_raw = parts[-1]
    dtype = _DTYPE_SUFFIX.get(dtype_raw, "")
    if not dtype:
        return None, "", f"未知 dtype 后缀 {dtype_raw!r}（不默认成 f32：会造成假阴性）"
    dims: list[int] = []
    for d in parts[:-1]:
        if not d.isdigit():
            return None, dtype, f"动态维 {d!r} 无法确定具体值（{text}）"
        dims.append(int(d))
    return tuple(dims), dtype, ""


def specs_from_module(module: Any) -> tuple[tuple[InputSpec, ...], tuple[str, ...]]:
    """从 VIR 推导入口输入规格。

    入口输入 = `AllocOrigin.FUNC_ARG` 的 buffer：它们由调用方持有，是 kernel
    的真实输入面。本地 `memref.alloc` 不算——那是中间缓冲，其内容由 kernel
    自己算出，预置随机值反而会掩盖"忘了初始化"这类缺陷。

    返回 `(输入规格, 无法推导的原因清单)`。
    """
    from hivm_spec.vir import AllocOrigin

    specs: list[InputSpec] = []
    problems: list[str] = []
    for alloc in getattr(module, "allocs", ()):
        if alloc.origin is not AllocOrigin.FUNC_ARG:
            continue
        shape, dtype, why = parse_shape_text(alloc.shape_text)
        key = alloc.value or alloc.name
        if shape is None:
            problems.append(f"{alloc.name}: {why}")
            continue
        specs.append(InputSpec(name=key, dtype=dtype, shape=shape))
    return tuple(specs), tuple(problems)
