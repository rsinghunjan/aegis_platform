"""Advisory policy reasoning and explicitly approved policy changes."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from ai_core.observability import AIObservability
from ai_core.store import AICoreStore


class OpaPolicyClient:
    """Writes versioned tenant policy data through OPA's Data API."""

    def __init__(self, base_url: str, timeout_s: float = 2.0):
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("OPA policy URL must use HTTP or HTTPS")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def apply(self, tenant_id: str, version: str, policy: Mapping[str, Any]) -> None:
        url = (
            f"{self.base_url}/v1/data/aegis/ai_policies/"
            f"{quote(tenant_id, safe='')}"
        )
        body = json.dumps(
            {"active": {"version": version, "policy": dict(policy)}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urlopen(request, timeout=self.timeout_s):
            pass


class AIPolicyEngine:
    def __init__(
        self,
        observability: AIObservability,
        cost_budget_usd: float | None = None,
        min_samples: int = 5,
        replay_evaluator: Callable[
            [Mapping[str, Any], Sequence[Mapping[str, Any]]], Any
        ]
        | None = None,
        opa_client: Any = None,
        store: AICoreStore | None = None,
    ):
        self.observability = observability
        self.cost_budget_usd = cost_budget_usd
        self.min_samples = min_samples
        self.replay_evaluator = replay_evaluator
        self.opa_client = opa_client
        if self.opa_client is None and os.getenv("AEGIS_OPA_POLICY_URL"):
            self.opa_client = OpaPolicyClient(os.environ["AEGIS_OPA_POLICY_URL"])
        self.store = store
        self._lock = threading.RLock()
        self._events: dict[str, list[dict[str, Any]]] = {}

    def suggest(self, tenant_id: str) -> list[dict[str, Any]]:
        stats = self.observability.summary(tenant_id)
        if stats["answers"] < self.min_samples:
            return []
        suggestions: list[dict[str, Any]] = []
        if stats["failure_rate"] > 0.25:
            suggestions.append(
                {
                    "policy": "ai_generate",
                    "change": "require_approval",
                    "reason": "failure rate above 25%; gate generation until providers recover",
                    "requires_human_approval": True,
                }
            )
        if (
            self.cost_budget_usd is not None
            and stats["total_cost_usd"] > self.cost_budget_usd
        ):
            suggestions.append(
                {
                    "policy": "ai_generate",
                    "change": "lower_max_tokens",
                    "reason": "observed spend exceeds the configured cost budget",
                    "requires_human_approval": True,
                }
            )
        return suggestions

    def propose(
        self,
        tenant_id: str,
        suggestion: Mapping[str, Any],
        replay_traffic: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if suggestion.get("policy") != "ai_generate" or suggestion.get(
            "change"
        ) not in ("require_approval", "lower_max_tokens"):
            raise ValueError("unsupported policy suggestion")
        if suggestion.get("requires_human_approval") is not True:
            raise ValueError("policy suggestions must require human approval")
        if not replay_traffic or len(replay_traffic) > 500:
            raise ValueError("replay traffic must contain between 1 and 500 cases")
        if self.replay_evaluator is None:
            raise RuntimeError("policy replay evaluator is not configured")

        traffic = [dict(case) for case in replay_traffic]
        traffic_json = json.dumps(
            traffic, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if len(traffic_json.encode("utf-8")) > 1_000_000:
            raise ValueError("replay traffic exceeds the maximum size")
        policy = {
            "policy": "ai_generate",
            "change": suggestion["change"],
            "value": (
                True
                if suggestion["change"] == "require_approval"
                else 256
            ),
        }
        replay_hash = hashlib.sha256(traffic_json.encode("utf-8")).hexdigest()
        try:
            result = self.replay_evaluator(policy, traffic)
        except Exception as exc:
            self._record(
                tenant_id,
                {
                    "event": "replay_failed",
                    "policy": policy,
                    "traffic_sha256": replay_hash,
                    "case_count": len(traffic),
                    "error_type": type(exc).__name__,
                },
            )
            raise RuntimeError("policy replay evaluation failed") from exc
        passed = result is True or (
            isinstance(result, Mapping) and result.get("passed") is True
        )
        if not passed:
            self._record(
                tenant_id,
                {
                    "event": "replay_rejected",
                    "policy": policy,
                    "traffic_sha256": replay_hash,
                    "case_count": len(traffic),
                },
            )
            raise ValueError("policy suggestion did not pass replay validation")

        proposal_id = str(uuid.uuid4())
        event = {
            "event": "proposed",
            "proposal_id": proposal_id,
            "status": "pending",
            "policy": policy,
            "reason": str(suggestion.get("reason", ""))[:500],
            "traffic_sha256": replay_hash,
            "case_count": len(traffic),
        }
        self._record(tenant_id, event)
        return dict(event)

    def approve(
        self, tenant_id: str, proposal_id: str, actor: str, reason: str = ""
    ) -> dict[str, Any]:
        if not actor:
            raise ValueError("approving actor is required")
        with self._lock:
            proposal = self._proposal(tenant_id, proposal_id)
            if proposal is None:
                raise KeyError("policy proposal not found")
            if proposal["status"] != "pending":
                raise ValueError("policy proposal is not pending approval")
            if self.opa_client is None:
                raise RuntimeError("OPA policy client is not configured")
            return self._apply(
                tenant_id,
                proposal["policy"],
                actor,
                reason,
                proposal_id=proposal_id,
            )

    def list_proposals(self, tenant_id: str) -> list[dict[str, Any]]:
        events = self._read_events(tenant_id)
        applied = {
            event.get("proposal_id")
            for event in events
            if event.get("event") == "applied"
        }
        return [
            {
                **event,
                "status": (
                    "applied" if event["proposal_id"] in applied else "pending"
                ),
            }
            for event in events
            if event.get("event") == "proposed"
        ]

    def history(self, tenant_id: str) -> list[dict[str, Any]]:
        return [
            event
            for event in self._read_events(tenant_id)
            if event.get("event") in ("applied", "rolled_back")
        ]

    def rollback(
        self, tenant_id: str, version: str, actor: str, reason: str = ""
    ) -> dict[str, Any]:
        if not actor:
            raise ValueError("rollback actor is required")
        with self._lock:
            events = self._read_events(tenant_id)
            target = next(
                (
                    event
                    for event in events
                    if event.get("event") in ("applied", "rolled_back")
                    and event.get("version") == version
                ),
                None,
            )
            if target is None:
                raise KeyError("policy version not found")
            active = self._active(events)
            if active is None or active["version"] == version:
                raise ValueError("policy version is already active or unavailable")
            if self.opa_client is None:
                raise RuntimeError("OPA policy client is not configured")
            return self._apply(
                tenant_id,
                target["policy"],
                actor,
                reason,
                rollback_to=version,
            )

    def _apply(
        self,
        tenant_id: str,
        policy: Mapping[str, Any],
        actor: str,
        reason: str,
        proposal_id: str | None = None,
        rollback_to: str | None = None,
    ) -> dict[str, Any]:
        events = self._read_events(tenant_id)
        previous = self._active(events)
        policy_value = dict(policy)
        policy_hash = hashlib.sha256(
            json.dumps(policy_value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        audit = {
            "proposal_id": proposal_id,
            "previous_version": previous["version"] if previous else None,
            "rollback_to": rollback_to,
            "actor": actor,
            "reason": reason[:500],
            "policy": policy_value,
            "policy_sha256": policy_hash,
        }
        approval = self._record(
            tenant_id, {"event": "approval", **audit}, allocate_version=True
        )
        version = approval["version"]
        audit["version"] = version
        try:
            self.opa_client.apply(tenant_id, version, policy_value)
        except Exception as exc:
            self._record(
                tenant_id,
                {
                    "event": "apply_failed",
                    **audit,
                    "error_type": type(exc).__name__,
                },
            )
            raise RuntimeError("OPA policy update failed") from exc

        event = {
            "event": "rolled_back" if rollback_to else "applied",
            "status": "applied",
            **audit,
        }
        self._record(tenant_id, event)
        return dict(event)

    def _proposal(self, tenant_id: str, proposal_id: str) -> dict[str, Any] | None:
        events = self._read_events(tenant_id)
        proposal = next(
            (
                event
                for event in events
                if event.get("event") == "proposed"
                and event.get("proposal_id") == proposal_id
            ),
            None,
        )
        if proposal is None:
            return None
        if any(
            event.get("event") == "applied"
            and event.get("proposal_id") == proposal_id
            for event in events
        ):
            return {**proposal, "status": "applied"}
        return proposal

    @staticmethod
    def _active(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        return next(
            (
                event
                for event in reversed(events)
                if event.get("event") in ("applied", "rolled_back")
            ),
            None,
        )

    def _read_events(self, tenant_id: str) -> list[dict[str, Any]]:
        if self.store is not None:
            return self.store.policy_events(tenant_id)
        return list(self._events.get(tenant_id, ()))

    def _record(
        self,
        tenant_id: str,
        event: dict[str, Any],
        allocate_version: bool = False,
    ) -> dict[str, Any]:
        if self.store is not None:
            recorded = self.store.append_policy_event(
                tenant_id, event, allocate_version=allocate_version
            )
        else:
            recorded = dict(event)
            if allocate_version:
                used_versions = [
                    int(item["version"].removeprefix("ai-policy-v"))
                    for item in self._read_events(tenant_id)
                    if item.get("version", "").startswith("ai-policy-v")
                ]
                next_version = max(used_versions, default=0) + 1
                recorded["version"] = f"ai-policy-v{next_version:06d}"
        self._events.setdefault(tenant_id, []).append(recorded)
        self._events[tenant_id] = self._events[tenant_id][-1000:]
        return recorded
