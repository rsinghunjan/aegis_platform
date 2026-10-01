from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from aegis_policy.contracts import EngineDecisionRecord


def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class SignedBlob:
    payload_json: Mapping[str, Any]
    payload_sha256: str
    signature_primary: str
    key_ref_primary: str
    alg_primary: str
    signature_secondary: str | None = None
    key_ref_secondary: str | None = None
    alg_secondary: str | None = None


@dataclass(frozen=True)
class DecisionEvidence:
    request_id: str
    action: str
    resource_type: str
    resource_id: str
    bundle_sha256: str
    pin_scope_used: str
    final_allow: bool
    final_reason: str
    disagree: bool
    engine_records: Sequence[EngineDecisionRecord]

    def to_json(self) -> dict[str, Any]:
        def rec_to_json(r: EngineDecisionRecord) -> dict[str, Any]:
            return {
                "engine": r.engine,
                "bundle_sha256": r.bundle_sha256,
                "latency_ms": r.latency_ms,
                "decision": {
                    "allow": r.decision.allow,
                    "reason": r.decision.reason,
                    "policy_id": r.decision.policy_id,
                    "policy_version": r.decision.policy_version,
                    "obligations": [{"type": o.type, **dict(o.params)} for o in r.decision.obligations],
                    "labels": dict(r.decision.labels),
                },
            }

        return {
            "request_id": self.request_id,
            "action": self.action,
            "resource": {"type": self.resource_type, "id": self.resource_id},
            "policy": {"bundle_sha256": self.bundle_sha256, "pin_scope_used": self.pin_scope_used},
            "final": {"allow": self.final_allow, "reason": self.final_reason, "disagree": self.disagree},
            "engines": [rec_to_json(r) for r in self.engine_records],
        }
