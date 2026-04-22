from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Mapping
from urllib.request import Request, urlopen

from aegis_policy.contracts import EngineDecisionRecord, Obligation, PolicyDecision, PolicyInput


@dataclass(frozen=True)
class OpaHttpEngine:
    """
    Evaluates OPA Pattern-2 policy over HTTP.
    Expects query path to return: {"result": <decision-object>}
    """

    engine_name: str  # "opa-central-http" or "opa-sidecar-http"
    endpoint_url: str  # e.g. http://opa:8181/v1/data/aegis/decision
    bundle_sha256: str
    timeout_s: float = 2.0

    def evaluate(self, policy_input: PolicyInput) -> EngineDecisionRecord:
        payload = {"input": policy_input.to_json()}
        body = json.dumps(payload).encode("utf-8")

        req = Request(
            self.endpoint_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.time()
        with urlopen(req, timeout=self.timeout_s) as resp:
            raw_bytes = resp.read()
        latency_ms = int((time.time() - start) * 1000)

        raw = json.loads(raw_bytes.decode("utf-8"))
        result = raw.get("result")
        if result is None or not isinstance(result, Mapping):
            # Fail closed.
            decision = PolicyDecision(allow=False, reason="opa_invalid_result")
            return EngineDecisionRecord(
                engine=self.engine_name,  # type: ignore[arg-type]
                decision=decision,
                bundle_sha256=self.bundle_sha256,
                latency_ms=latency_ms,
                raw=raw,
            )

        obligations = []
        for o in result.get("obligations", []) or []:
            if isinstance(o, Mapping):
                t = str(o.get("type", ""))
                params = {k: v for k, v in o.items() if k != "type"}
                obligations.append(Obligation(type=t, params=params))

        decision = PolicyDecision(
            allow=bool(result.get("allow", False)),
            reason=str(result.get("reason", "")),
            policy_id=(str(result.get("policy_id")) if result.get("policy_id") is not None else None),
            policy_version=(str(result.get("policy_version")) if result.get("policy_version") is not None else None),
            obligations=tuple(obligations),
            labels=(dict(result.get("labels", {})) if isinstance(result.get("labels", {}), Mapping) else {}),
        )

        return EngineDecisionRecord(
            engine=self.engine_name,  # type: ignore[arg-type]
            decision=decision,
            bundle_sha256=self.bundle_sha256,
            policy_id=decision.policy_id,
            policy_version=decision.policy_version,
            latency_ms=latency_ms,
            raw=raw,
        )
