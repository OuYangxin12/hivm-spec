#!/usr/bin/env python3
"""对照两次 hivm-spec 结论（`--json` 产物），报告 verdict 漂移与审计坐标变化。

用法:
  compare-verdicts.py base.json head.json      # 基线在前，改后在后
退出码:
  0 = 无恶化（含无变化 / 改进）
  1 = 至少一处恶化
  2 = 输入不合法

"只贴改后绿灯不贴基线等于没有验证" —— 本脚本存在的唯一理由是让基线对照成为一条命令。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROBLEM = ("OVERFLOW", "DEADLOCK", "MISMATCH")
GAP = ("COVERAGE_GAP", "UNTRUSTED_DESCRIPTION")


def load(path: str) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"读取失败 {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if not isinstance(doc, dict):
        print(f"{path}: 不是 JSON 对象", file=sys.stderr)
        raise SystemExit(2)
    if "checks" not in doc:  # 单工具 JSON：伪装成 run 形态
        doc = {"verdict": doc.get("verdict"), "checks": [doc]}
    return doc


def classify(base: str, head: str) -> tuple[str, bool]:
    """返回 (判定文案, 是否恶化)。"""
    if base == head:
        return "无变化", False
    if base == "OK" and head in PROBLEM:
        return "**恶化**（引入真问题）", True
    if base == "OK" and head in GAP:
        return "**恶化**（从可判变为不可判）", True
    if base in GAP and head == "OK":
        return "改进（覆盖恢复）", False
    if base in PROBLEM and head == "OK":
        return "改进（问题消除）", False
    if base in PROBLEM and head in GAP:
        return "**可疑**：可判语料变不可判，需解释是否此前漏判", True
    if base in GAP and head in PROBLEM:
        return "**可疑**：缺口处暴露真问题", True
    return "变化（自行判断方向）", False


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    base, head = load(argv[1]), load(argv[2])
    b = {c["tool"]: c for c in base["checks"]}
    h = {c["tool"]: c for c in head["checks"]}

    worsened: list[str] = []
    print(f"{'CHECK':<26} {'BASE':<24} {'HEAD':<24} 判定")
    print("-" * 86)
    for tool in sorted(b.keys() | h.keys()):
        bv = b.get(tool, {}).get("verdict", "(absent)")
        hv = h.get(tool, {}).get("verdict", "(absent)")
        call, bad = classify(bv, hv)
        print(f"{tool:<26} {bv:<24} {hv:<24} {call}")
        if bad:
            worsened.append(f"{tool}: {bv} -> {hv}")

    # 审计坐标：描述或引擎变了，结论差异就可能来自模型而非被测 pass
    for key in ("spec_hash", "engine_version"):
        bs = {c["tool"]: c.get(key) for c in base["checks"]}
        hs = {c["tool"]: c.get(key) for c in head["checks"]}
        diff = sorted(t for t in bs if t in hs and bs[t] and bs[t] != hs.get(t))
        if diff:
            print(f"\n⚠️ {key} 在 {len(diff)} 个检查上不一致：{', '.join(diff[:5])}")
            print("   → 结论差异可能来自描述/引擎变更而非被测 pass。先对齐坐标再下结论。")

    fp = {c["tool"]: c.get("ir_fingerprint") for c in base["checks"]}
    fp2 = {c["tool"]: c.get("ir_fingerprint") for c in head["checks"]}
    if any(t in fp2 and fp[t] and fp[t] != fp2[t] for t in fp):
        print(
            "\nℹ️ 被测 IR 指纹已变化（正常：你对 IR 做了改动）。跨 commit 复用历史结论前，"
            "先确认这是同一份被测输入。"
        )

    trust = {c.get("trust") for c in head["checks"] if c.get("trust")}
    if trust - {"anchored"}:
        print(
            "\n信任级别："
            + ", ".join(sorted(str(t) for t in trust))
            + " —— 未经对拍，不可作为合入门禁依据。"
        )

    if worsened:
        print(f"\n❌ 恶化 {len(worsened)} 处：")
        for w in worsened:
            print(f"   {w}")
        return 1
    print("\n✅ 无恶化。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
