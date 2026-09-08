#!/usr/bin/env python3
"""V1 gate: parse HIVM-dialect MLIR text via bishengir Python bindings.

Verifies the hivm-spec engine's planned IR interface capability:
register hivm dialect in an MLIR Context, parse .mlir files written in
custom (non-generic) syntax, and traverse ops.

Usage:
    python3 v1_hivm_parse_test.py <bindings_path_root> <hivm_test_dir> [sample_n]

<bindings_path_root>: dir whose import root exposes `bishengir` (assembled
    python_packages/bishengir tree of an AscendNPU-IR build with
    MLIR_ENABLE_BINDINGS_PYTHON=ON).
"""
import glob
import os
import sys
from collections import Counter


def main() -> int:
    bindings_root = sys.argv[1]
    test_dir = sys.argv[2]
    sample_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    sys.path.insert(0, bindings_root)
    from bishengir import ir  # noqa: E402  (full MLIR python API, bishengir-prefixed)

    # --- hivm-spec 引擎引导步（bootstrap）---
    # 主仓 bindings 缺口：_mlir_libs 站点初始化只探测 `_mlirRegisterEverything`，
    # 不探测带前缀的 `_bishengirRegisterEverything`，导致 hivm/hfusion 等
    # 方言不会被自动注册。该扩展的 register_dialects 直接收 MlirContext，
    # 因此在 Context 创建后对其显式注册（MLIR 允许向活跃 Context 追加方言）。
    import bishengir._mlir_libs._bishengirRegisterEverything as _reg  # noqa: E402

    ctx = ir.Context()
    _reg.register_dialects(ctx)

    with ctx:
        # 注册证据 = 自定义语法解析成功（部分绑定版本无 get_registered_dialects）
        try:
            registered = ctx.get_registered_dialects()
            hivm_registered = [d for d in registered if "hivm" in d]
            print(f"[dialects] registered={len(registered)} hivm={hivm_registered}")
        except AttributeError:
            print("[dialects] registration check unavailable; parse result is the evidence")

        files = sorted(glob.glob(os.path.join(test_dir, "*.mlir")))
        # negative tests (expected-error) legitimately fail to parse: exclude
        positive = [
            f for f in files
            if "expected-error" not in open(f, encoding="utf-8", errors="replace").read()
        ]
        print(f"[files] total={len(files)} positive-parse-candidates={len(positive)}")
        if not positive:
            print("V1 FAIL: no positive test files found")
            return 1

        sample = positive[:sample_n]
        ops: Counter = Counter()
        ok = 0
        for f in sample:
            name = os.path.basename(f)
            try:
                module = ir.Module.parse(open(f, encoding="utf-8").read(), ctx)

                # 手写递归遍历（该绑定版本的 walk() 约定不同，不依赖它）
                def _iter_block_ops(block):
                    for op in block.operations:
                        yield op
                        for region in op.regions:
                            for b in region.blocks:
                                yield from _iter_block_ops(b)

                count = 0
                for op in _iter_block_ops(module.body):
                    count += 1
                    ops[op.operation.name] += 1
                ok += 1
                print(f"  OK   {name}  ops={count}")
            except Exception as e:  # noqa: BLE001  (report and continue)
                print(f"  FAIL {name}  {str(e)[:150]}")

        print(f"[result] parsed {ok}/{len(sample)}")
        print(f"[ops] distinct={len(ops)} top={ops.most_common(8)}")
        if ok == len(sample):
            print("V1 PASS: hivm dialect parses and traverses from Python")
            return 0
        print("V1 FAIL: some files unparseable")
        return 1


if __name__ == "__main__":
    sys.exit(main())
