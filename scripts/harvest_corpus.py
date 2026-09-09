#!/usr/bin/env python3
"""L2 语料剥离与入库（D13：强制夹具剥离，逐条留档）。

主仓 pass UT 不是干净的 kernel：含 lit 指令、`"some_op"` 占位符等**夹具细节**。
直接入库会把夹具细节污染进语料库（D13 拒绝批量导入的理由之一）。本脚本做
**可审计的剥离**：

1. 剥离 lit 指令行（// RUN: / // CHECK*: / // OTHER*）与整行注释；
2. 按 `// -----` 分节（split-input-file），每节独立成条目；
3. `"some_op"` 占位符按**类型定向替换**——占位符是 UT 作者为让 pass 有输入而
   造的"值发生器"，不是 kernel 语义：
   - 标量类型（i1/i32/index）→ `arith.constant`（不引入分配）；
   - tensor 类型 → `tensor.empty`（纯 SSA 值，无 buffer）；
   - memref 类型 → **提升为函数参数**（上游传入的 buffer；不能用 memref.alloc
     顶替——那会凭空增加分配，破坏占用语义）；
4. 每步替换记录进 manifest 的 `extraction` 字段；
5. 入库前经真实 bindings 严格解析（不开 allow_unregistered）——解析不过不入库。

用法：
    python scripts/harvest_corpus.py <主仓源文件路径> [--sections all|1,3-5] [--dry-run]

主仓源文件路径相对 AscendNPU-IR 根目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "specs" / "cases" / "corpus"
MANIFEST = CORPUS / "manifest.json"
PARENT = REPO.parent

#: lit 指令行前缀（RUN / CHECK* / OTHER*）
_LIT_LINE = re.compile(r"^\s*//\s*(RUN:|CHECK|OTHER)")
#: "some_op" 占位符：捕获类型文本
_PLACEHOLDER = re.compile(r'^(\s*)%"?([A-Za-z0-9_]+)"? = "some_op"\(\) : \(\) -> (.+)$', re.M)
#: func 签名行
_FUNC_LINE = re.compile(r"^(\s*)(func\.func @([A-Za-z0-9_]+)\()([^)]*)(\).*)$", re.M)


def _parent_commit() -> str:
    out = subprocess.run(
        ["git", "-C", str(PARENT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def split_sections(text: str) -> list[str]:
    """按 lit 的 split-input-file 分节符切分。无分节符则整体一节。"""
    parts = re.split(r"^// -+ *\n", text, flags=re.M)
    return [p for p in (seg.strip("\n") for seg in parts) if p.strip()]


def strip_fixture(section: str) -> tuple[str, list[str]]:
    """剥离 lit 指令与整行注释，返回（结果, 剥离记录）。"""
    records: list[str] = []
    kept: list[str] = []
    n_lit = n_comment = 0
    for line in section.splitlines():
        if _LIT_LINE.match(line):
            n_lit += 1
            continue
        if line.lstrip().startswith("//"):
            n_comment += 1
            continue
        kept.append(line)
    if n_lit:
        records.append(f"剥离 lit 指令行 {n_lit} 行")
    if n_comment:
        records.append(f"剥离注释行 {n_comment} 行")
    return "\n".join(kept).strip("\n"), records


#: 消费者占位符："some_consume"(%v) : (T) -> () —— UT 作者为消化结果值而造。
#: 无结果、无分配，剥离整行不影响占用语义；其消费的 SSA 值变为未使用，合法。
_CONSUMER = re.compile(r'^\s*"[^"]+"\([^)]*\) : \([^)]*\) -> \(\)\s*$', re.M)


def substitute_placeholders(text: str, func_args: list[str]) -> tuple[str, list[str], set[str]]:
    """类型定向替换 "some_op" 占位符。

    memref 型占位符无法在不引入分配的前提下产生值，提升为函数参数
    （追加到首个 func.func 签名尾部）。返回（新文本, 替换记录, 待提升的参数）。
    """
    records: list[str] = []
    to_hoist: list[tuple[str, str]] = []  # (名字, 类型)

    n_consumer = len(_CONSUMER.findall(text))
    if n_consumer:
        text = _CONSUMER.sub("", text)
        records.append(f"剥离消费者占位符 {n_consumer} 行（无结果无分配）")

    def _sub(m: re.Match[str]) -> str:
        indent, name, typ = m.group(1), m.group(2), m.group(3).strip()
        if typ.startswith("memref<"):
            to_hoist.append((name, typ))
            records.append(f"{name}: memref 占位符 → 函数参数（避免引入分配）")
            return f"{indent}// harvested-away: {name} 提升为函数参数"
        if typ in ("i1",):
            records.append(f"{name}: i1 占位符 → arith.constant false")
            return f"{indent}%{name} = arith.constant false"
        if typ in ("i32", "index"):
            v = "4" if typ == "i32" else "16"
            records.append(f"{name}: {typ} 占位符 → arith.constant {v}（值不影响占用分析）")
            return f"{indent}%{name} = arith.constant {v} : {typ}"
        if typ.startswith("tensor<"):
            records.append(f"{name}: tensor 占位符 → tensor.empty（纯 SSA 值，无 buffer）")
            return f"{indent}%{name} = tensor.empty() : {typ}"
        records.append(f"{name}: 无法识别的占位符类型 {typ!r}，保留原样（将解析失败）")
        return m.group(0)

    text = _PLACEHOLDER.sub(_sub, text)
    if to_hoist:
        func_args.extend(f"%{n}: {t}" for n, t in to_hoist)
    return text, records, {n for n, _ in to_hoist}


def hoist_args(text: str, new_args: list[str]) -> tuple[str, str | None]:
    """把待提升参数追加到首个 func.func 签名，返回（新文本, func 名）。"""
    m = _FUNC_LINE.search(text)
    if not m:
        return text, None
    indent, name = m.group(1), m.group(3)
    old_sig = m.group(0)
    args = m.group(4).rstrip()
    if args:
        args += ", "
    args += ", ".join(new_args)
    # group(5) 以 ")" 开头（闭合原参数表）并以函数体 "{" 结尾，勿再补括号
    new_sig = f"{indent}{m.group(2)}{args}{m.group(5)}"
    return text.replace(old_sig, new_sig, 1), name


def strict_parse(text: str) -> None:
    """严格解析校验（不开 allow_unregistered）。失败抛异常。"""
    from hivm_spec.bindings import BindingsError, load_bindings
    from hivm_spec.vir import VIRError

    try:
        load_bindings().parse_module(text)
    except BindingsError:
        raise
    except VIRError:
        raise
    except Exception as exc:
        raise VIRError(f"严格解析失败：{exc}") from exc


def harvest(source_rel: str, sections: str, dry_run: bool) -> int:
    src = PARENT / source_rel
    if not src.is_file():
        print(f"源文件不存在：{src}", file=sys.stderr)
        return 1
    commit = _parent_commit()
    raw = src.read_text(encoding="utf-8")

    all_sections = split_sections(raw)
    idx = _select(sections, len(all_sections))
    print(f"源：{source_rel} @ {commit[:12]}，共 {len(all_sections)} 节，选取 {idx}")

    entries: list[dict[str, object]] = []
    for n in idx:
        sec = all_sections[n - 1]
        body, records = strip_fixture(sec)
        new_args: list[str] = []
        body, sub_records, _hoisted = substitute_placeholders(body, new_args)
        records.extend(sub_records)
        if new_args:
            body, func_name = hoist_args(body, new_args)
            records.append(f"函数签名追加 {len(new_args)} 个提升参数")
        else:
            _, func_name = hoist_args(body, [])
        func_name = func_name or f"sec{n:02d}"

        label = f"sec{n:02d}" if len(all_sections) > 1 else "full"
        out_name = f"{Path(source_rel).stem}-{label}-{func_name}.mlir"
        out_path = CORPUS / "l2" / out_name

        if not dry_run:
            try:
                strict_parse(body)
            except Exception as exc:
                print(f"  ✗ {out_name}：{exc}", file=sys.stderr)
                return 1
            content = body + "\n"
            out_path.write_text(content, encoding="utf-8")

        entries.append(
            {
                "path": f"l2/{out_name}",
                "layer": "l2",
                "origin": f"harvested:{source_rel}#sec{n}/{len(all_sections)}",
                "source_path": source_rel,
                # 哈希对象必须是**写盘字节**（含尾换行），否则哈希对不上文件
                "sha256": "sha256:" + hashlib.sha256(content.encode()).hexdigest(),
                "source_commit": commit,
                "features": _features(body),
                "parse_verified": "hivm-spec bindings strict (no allow-unregistered)",
                "extraction": "；".join(records) if records else "无夹具细节，原文照录",
            }
        )
        print(f"  ✓ {out_name}（{len(body.splitlines())} 行，替换 {len(records)} 项）")

    if dry_run:
        print("（dry-run：未写入）")
        return 0

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    have = {e["path"] for e in manifest["entries"]}
    fresh = [e for e in entries if str(e["path"]) not in have]
    manifest["entries"].extend(fresh)
    manifest["entries"].sort(key=lambda e: (str(e["layer"]), str(e["path"])))
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest 已更新：新增 {len(fresh)} 条，共 {len(manifest['entries'])} 条")
    return 0


def _select(spec: str, total: int) -> list[int]:
    if spec == "all":
        return list(range(1, total + 1))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [n for n in out if not 1 <= n <= total]
    if bad:
        raise SystemExit(f"节号越界：{bad}（共 {total} 节）")
    return sorted(set(out))


def _features(body: str) -> list[str]:
    feats = []
    if "scf.for" in body:
        feats.append("scf_for")
    if "scf.if" in body:
        feats.append("scf_if")
    if "scope.scope" in body:
        feats.append("scope_scope")
    if "multi_buffer" in body:
        feats.append("hivm_multi_buffer")
    if "alloc_workspace" in body:
        feats.append("alloc_workspace")
    if "address_space" in body:
        feats.append("address_space")
    if "?x" in body or "memref<?" in body:
        feats.append("dynamic_shape")
    return feats or ["plain"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="主仓源文件（相对 AscendNPU-IR 根）")
    ap.add_argument("--sections", default="all", help="节号，如 all / 1 / 1,3-5")
    ap.add_argument("--dry-run", action="store_true", help="只报告不写入")
    a = ap.parse_args()
    return harvest(a.source, a.sections, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
