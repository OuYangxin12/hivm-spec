"""配置文档生成与信任账本（T0.3 / T0.4 / T0.5）。

产出物三件：
1. **配置文档**（归一化 JSON）—— 装配器的输入，FR8 要求两次生成逐字节一致；
2. **信任账本** —— 每个 op 的 trust/provenance，结论降级的依据（D4）；
3. **漂移账本** —— 语义漂移条目结构（D9）：漂移**总是先登记**，并冻结该 op 的
   trust 升级，直到处置完成。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hivm_spec.spec import HostFn, Spec, Trust

__all__ = [
    "CONFIG_SCHEMA",
    "DriftEntry",
    "GenResult",
    "TrustLedger",
    "generate",
    "write_outputs",
]

#: 引擎版本：与 `spec_hash` 分离（框架 §7.3）——描述未变而引擎变了，
#: 结论也可能变，故两者必须都能被审计。
ENGINE_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# 配置文档 schema（T0.3：jsonschema 校验）
# ---------------------------------------------------------------------------

CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "hivm-spec 配置文档",
    "type": "object",
    "required": ["schema_version", "spec_name", "arch", "ops", "vm", "checks"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": 1},
        "spec_name": {"type": "string", "minLength": 1},
        "arch": {"enum": ["a3", "a5"]},
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op", "pipe", "trust", "escape_hatch", "value", "params", "effects"],
                "additionalProperties": False,
                "properties": {
                    "op": {"type": "string", "minLength": 1},
                    "pipe": {"type": "string"},
                    "trust": {"enum": ["provisional", "cross-validated", "anchored"]},
                    "escape_hatch": {"type": "boolean"},
                    "value": {"type": "object"},
                    "params": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "kind", "type", "variadic", "arity"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "kind": {"enum": ["in", "out", "attr"]},
                                "type": {"type": "string"},
                                "variadic": {"type": "boolean"},
                                "arity": {"type": ["array", "null"]},
                            },
                        },
                    },
                    "effects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["kind", "target", "space", "when", "event"],
                            "properties": {
                                "kind": {
                                    "enum": [
                                        "read",
                                        "write",
                                        "cond_write",
                                        "sync_set",
                                        "sync_wait",
                                        "sync_barrier",
                                    ]
                                }
                            },
                        },
                    },
                },
            },
        },
        "vm": {
            "type": "object",
            "required": ["spaces", "pipes", "events"],
            "additionalProperties": False,
            "properties": {
                "spaces": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "capacity", "align"],
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "capacity": {"type": ["integer", "null"], "minimum": 1},
                            "align": {"type": ["integer", "null"], "minimum": 1},
                        },
                    },
                },
                "pipes": {"type": "array", "items": {"type": "string"}},
                "events": {"type": "array", "items": {"type": "string"}},
            },
        },
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "options"],
                "additionalProperties": False,
                "properties": {
                    "name": {"enum": ["ub_occupancy", "timeline", "equivalence", "sync_pairing"]},
                    "options": {"type": "object"},
                },
            },
        },
        # 未经对拍的语义假设（D9 前置登记，M2 审查发现 3）
        "assumptions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "subject",
                    "assumed",
                    "authority",
                    "rationale",
                    "risk_direction",
                    "resolve_by",
                ],
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string", "minLength": 1},
                    "assumed": {"type": "string", "minLength": 1},
                    "authority": {"enum": ["upstream-cpp", "hardware-golden", "doc"]},
                    "rationale": {"type": "string"},
                    "risk_direction": {"enum": ["false-positive", "false-negative", "both"]},
                    "resolve_by": {"type": "string"},
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# 信任账本（T0.4）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DriftEntry:
    """语义漂移条目（D9）。

    漂移**总是先登记**，且登记即冻结该 op 的 trust 升级——防止"发现漂移但
    因为赶进度而先升级信任"这类自欺。
    """

    op: str
    observed: str
    described: str
    #: "upstream-cpp" | "hardware-golden" | "unknown"
    authority: str = "upstream-cpp"
    #: 处置状态："open" | "description-fixed" | "upstream-suspected"
    status: str = "open"
    note: str = ""

    @property
    def freezes_trust_upgrade(self) -> bool:
        return self.status == "open"


@dataclass
class TrustLedger:
    """信任账本 + 覆盖报告 + 漂移账本。"""

    spec_name: str
    spec_hash: str
    engine_version: str = ENGINE_VERSION
    entries: dict[str, str] = field(default_factory=dict)
    escape_hatches: list[str] = field(default_factory=list)
    drift: list[DriftEntry] = field(default_factory=list)
    #: 未经对拍的语义假设（D9 前置；结构见 spec.SemanticAssumption）
    assumptions: list[Any] = field(default_factory=list)
    generated_at: str = ""

    def frozen_ops(self) -> tuple[str, ...]:
        """因未处置漂移而冻结 trust 升级的 op（D9）。"""
        return tuple(sorted({d.op for d in self.drift if d.freezes_trust_upgrade}))

    def unresolved_assumptions(self) -> tuple[str, ...]:
        """尚未对拍的语义假设主体——同样冻结 trust 升级（D9 前置）。

        理由与 drift 一致：语义还没和权威链对齐就升级信任，等于把猜测
        当成结论。假设登记在案且未销案时，信任封顶 provisional。
        """
        return tuple(sorted({a.subject for a in self.assumptions}))

    def max_trust(self) -> str:
        order = {"provisional": 0, "cross-validated": 1, "anchored": 2}
        if not self.entries:
            return "provisional"
        return max(self.entries.values(), key=lambda t: order[t])

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "spec_name": self.spec_name,
            "spec_hash": self.spec_hash,
            "engine_version": self.engine_version,
            "generated_at": self.generated_at,
            "coverage": {
                "modeled_op_count": len(self.entries),
                "escape_hatch_count": len(self.escape_hatches),
                "escape_hatches": sorted(self.escape_hatches),
            },
            "trust": dict(sorted(self.entries.items())),
            "frozen_by_drift": list(self.frozen_ops()),
            "frozen_by_assumption": list(self.unresolved_assumptions()),
            "assumptions": [
                {
                    "subject": a.subject,
                    "assumed": a.assumed,
                    "authority": a.authority,
                    "rationale": a.rationale,
                    "risk_direction": a.risk_direction,
                    "resolve_by": a.resolve_by,
                }
                for a in sorted(self.assumptions, key=lambda x: x.subject)
            ],
            "drift": [
                {
                    "op": d.op,
                    "observed": d.observed,
                    "described": d.described,
                    "authority": d.authority,
                    "status": d.status,
                    "note": d.note,
                }
                for d in sorted(self.drift, key=lambda x: (x.op, x.observed))
            ],
        }


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenResult:
    config: dict[str, Any]
    ledger: TrustLedger
    config_bytes: bytes

    @property
    def spec_hash(self) -> str:
        return self.ledger.spec_hash


def generate(spec: Spec, *, timestamp: str | None = None) -> GenResult:
    """从描述生成配置文档与信任账本。

    `timestamp` 显式可注入：账本含时间戳，若不可注入则测试无法验证确定性
    （FR8 只约束**配置文档**逐字节一致，账本的时间戳属元数据，不进 spec_hash）。
    """
    config = spec.normalize()
    config_bytes = spec.canonical_bytes()

    ledger = TrustLedger(
        spec_name=spec.name,
        spec_hash=spec.spec_hash(),
        generated_at=timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    ledger.assumptions = list(spec.assumptions)
    for op in spec.ops:
        ledger.entries[op.op] = op.effective_trust.value
        if isinstance(op.value, HostFn):
            ledger.escape_hatches.append(op.op)

    return GenResult(config=config, ledger=ledger, config_bytes=config_bytes)


def validate_config(config: dict[str, Any]) -> list[str]:
    """用 jsonschema 校验配置文档；返回错误信息列表。"""
    try:
        import jsonschema
    except ModuleNotFoundError:
        return ["jsonschema 不可用（环境问题，非配置错误）"]

    validator = jsonschema.Draft202012Validator(CONFIG_SCHEMA)
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in sorted(validator.iter_errors(config), key=lambda e: list(e.absolute_path))
    ]


def write_outputs(result: GenResult, config_path: Path) -> Path:
    """写配置文档与并列的账本文件；返回账本路径。

    配置文档以 `canonical_bytes` 原样写出——**不重新序列化**，否则 FR8 的
    逐字节一致性会依赖两处独立的序列化配置保持同步。
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(result.config_bytes)

    ledger_path = config_path.with_name(config_path.stem + ".ledger.json")
    ledger_path.write_text(
        json.dumps(result.ledger.to_json(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ledger_path


def trust_of(spec: Spec) -> dict[str, Trust]:
    return {op.op: op.effective_trust for op in spec.ops}
