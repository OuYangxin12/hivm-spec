"""T1.5 渲染器 + T1.0b 规格合并 + L2 剥离脚本的测试。

渲染器测试锚定三个性质：确定性、定宽、溢出可见。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from hivm_spec.render import render_occupancy_chart

REPO_ROOT = Path(__file__).resolve().parent.parent
HARVEST = REPO_ROOT / "scripts" / "harvest_corpus.py"


def _details(ub_peak: int, capacity: int | None = 192 * 1024) -> dict:
    return {
        "spaces": {
            "ub": {
                "capacity": capacity,
                "peak_bytes": ub_peak,
                "utilization": (ub_peak / capacity) if capacity else None,
                "curve": [[0, ub_peak], [1, ub_peak]],
                "contributors": [
                    {"name": "bufA", "nbytes": ub_peak, "interval": [0, 1], "shape": "x"}
                ],
                "unsized_buffers": 0,
            }
        }
    }


def test_chart_is_deterministic() -> None:
    a = render_occupancy_chart(_details(1024))
    b = render_occupancy_chart(_details(1024))
    assert a == b


def test_chart_lines_do_not_exceed_80_columns() -> None:
    text = render_occupancy_chart(_details(1024))
    for line in text.splitlines():
        assert len(line) <= 80, f"行超宽（{len(line)} 列）：{line!r}"


def test_overflow_is_visibly_marked() -> None:
    """溢出必须出现 `!` 标记——渲染层不得把坏消息画得不显眼。"""
    text = render_occupancy_chart(_details(300 * 1024))
    assert "!" in text, "溢出时应有满格 `!` 标记"


def test_unknown_capacity_is_stated_not_faked() -> None:
    text = render_occupancy_chart(_details(1024, capacity=None))
    assert "容量未知" in text


def test_unsized_buffers_are_disclosed() -> None:
    d = _details(1024)
    d["spaces"]["ub"]["unsized_buffers"] = 3
    text = render_occupancy_chart(d)
    assert "3 个 buffer 尺寸未知" in text
    assert "下界" in text


def test_sparkline_buckets_keep_peaks() -> None:
    """超宽曲线分桶时必须取**桶内峰值**——取均值会抹平真实的瞬时高峰。"""
    curve = [[i, 100 if i == 500 else 0] for i in range(1000)]
    text = render_occupancy_chart(
        {
            "spaces": {
                "ub": {
                    "capacity": 1024,
                    "peak_bytes": 100,
                    "utilization": 0.1,
                    "curve": curve,
                    "contributors": [],
                    "unsized_buffers": 0,
                }
            }
        }
    )
    spark = [ln for ln in text.splitlines() if ln.startswith("    ") and "█" in ln]
    assert spark, "峰值桶应渲染为实心块"


def test_multi_buffer_note_appears_for_contributors() -> None:
    d = _details(4096)
    d["spaces"]["ub"]["contributors"] = [
        {"name": "ws", "nbytes": 1024, "interval": [0, 1], "shape": "x", "multi_buffer": 4}
    ]
    text = render_occupancy_chart(d)
    assert "multi_buffer4" in text


def test_empty_details_render_nothing() -> None:
    assert render_occupancy_chart({"spaces": {}}) == ""


# ---------------------------------------------------------------------------
# Spec.merge（描述并集）
# ---------------------------------------------------------------------------


def _load_spec_module(name: str, path: Path) -> object:
    so = importlib.util.spec_from_file_location(name, path)
    assert so and so.loader
    mod = importlib.util.module_from_spec(so)
    sys.modules[name] = mod
    so.loader.exec_module(mod)
    return mod


def test_merge_produces_union() -> None:
    from hivm_spec.spec import SpecError  # noqa: F401

    toy = _load_spec_module("toy_m", REPO_ROOT / "specs" / "toy.py")
    cv = _load_spec_module("cv_m", REPO_ROOT / "specs" / "cv.py")
    assert isinstance(toy.spec, object) and isinstance(cv.spec, object)
    toy.spec.merge(cv.spec)  # type: ignore[attr-defined]
    ops = {o.op for o in toy.spec.ops}  # type: ignore[attr-defined]
    assert "hivm.hir.load" in ops and "hivm.hir.mmadL1" in ops
    assert len(ops) == 18


def test_merge_rejects_duplicate_op() -> None:
    import pytest

    from hivm_spec.spec import Spec, SpecError

    a, b = Spec(name="a"), Spec(name="b")
    a.op("hivm.hir.x", params={}, effects=(), value="v")
    b.op("hivm.hir.x", params={}, effects=(), value="v2")
    with pytest.raises(SpecError, match="op 重复声明"):
        a.merge(b)


def test_merge_rejects_arch_mismatch() -> None:
    import pytest

    from hivm_spec.spec import Spec, SpecError

    a, b = Spec(name="a", arch="a3"), Spec(name="b", arch="a5")
    with pytest.raises(SpecError, match="arch 不一致"):
        a.merge(b)


# ---------------------------------------------------------------------------
# harvest_corpus.py 纯函数
# ---------------------------------------------------------------------------


def _load_harvest():
    so = importlib.util.spec_from_file_location("harvest_corpus", HARVEST)
    assert so and so.loader
    mod = importlib.util.module_from_spec(so)
    sys.modules["harvest_corpus"] = mod
    so.loader.exec_module(mod)
    return mod


def test_strip_fixture_removes_lit_and_comments() -> None:
    h = _load_harvest()
    body, records = h.strip_fixture(
        "// RUN: x -y %s | FileCheck %s\n// CHECK: scf.for\n// 普通注释\nmodule {}\n"
    )
    assert body == "module {}"
    assert any("lit" in r for r in records) and any("注释" in r for r in records)


def test_substitute_scalar_and_tensor_placeholders() -> None:
    h = _load_harvest()
    text, records, hoisted = h.substitute_placeholders(
        '%x = "some_op"() : () -> i32\n%y = "some_op"() : () -> tensor<16x16xf16>\n',
        [],
    )
    assert records
    assert "%x = arith.constant false" not in text
    assert "arith.constant 4 : i32" in text
    assert "tensor.empty() : tensor<16x16xf16>" in text
    assert not hoisted


def test_substitute_hoists_memref_placeholders() -> None:
    """memref 占位符必须提升为函数参数——不能用 memref.alloc 顶替（会凭空加分配）。"""
    h = _load_harvest()
    args: list[str] = []
    text, records, hoisted = h.substitute_placeholders(
        '%in = "some_op"() : () -> memref<16x16xf16>\n', args
    )
    assert "some_op" not in text
    assert args == ["%in: memref<16x16xf16>"]
    assert hoisted == {"in"}
    assert "避免引入分配" in records[0]


def test_hoist_args_preserves_body_and_indent() -> None:
    h = _load_harvest()
    text, name = h.hoist_args(
        'module {\n  func.func @f(%a: memref<2xf32>) attributes {k = "mix"} {\n    return\n  }\n}',
        ["%b: memref<4xf32>"],
    )
    assert name == "f"
    assert "%a: memref<2xf32>, %b: memref<4xf32>) attributes" in text
    assert text.count("(") == text.count(")"), "括号必须配平（回归：曾多出一个右括号）"
    assert 'attributes {k = "mix"} {' in text


def test_consumer_placeholder_is_dropped() -> None:
    h = _load_harvest()
    text, records, _ = h.substitute_placeholders(
        '"consume"(%v) : (tensor<16x16xf16>) -> ()\n"some_consume"(%v) : (tensor<4xf16>) -> ()\n',
        [],
    )
    assert "consume" not in text
    assert any("消费者占位符 2" in r for r in records)


def test_merged_gen_matches_two_spec_union(tmp_path: Path) -> None:
    """gen 多文件 = 语义并集；单文件生成的 hash 必然不同。"""
    from hivm_spec.__main__ import main

    merged = tmp_path / "merged.json"
    assert (
        main(
            [
                "gen",
                str(REPO_ROOT / "specs" / "toy.py"),
                str(REPO_ROOT / "specs" / "cv.py"),
                "-o",
                str(merged),
            ]
        )
        == 0
    )
    doc = json.loads(merged.read_text(encoding="utf-8"))
    assert len(doc["ops"]) == 18
