"""Deterministic adapter from monitoring findings to governed agent runs."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, validator

from agentic.runtime import AgentRuntime, RiskLevel, ToolSpec


@dataclass
class RemediationAdapters:
    """Optional hooks; the defaults never contact model or cloud systems."""

    retrain: Optional[Callable[[dict[str, Any]], Any]] = None
    canary_deploy: Optional[Callable[[dict[str, Any]], Any]] = None
    promote: Optional[Callable[[dict[str, Any]], Any]] = None
    rollback: Optional[Callable[[dict[str, Any]], Any]] = None
    notify: Optional[Callable[[dict[str, Any]], Any]] = None

    def actions(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        return {
            name: callback
            for name, callback in (
                ("retrain", self.retrain),
                ("canary_deploy", self.canary_deploy),
                ("promote", self.promote),
                ("rollback", self.rollback),
            )
            if callback is not None
        }


class DriftFinding(BaseModel):
    tenant_id: str
    model_name: str
    metric: str
    value: float
    threshold: float
    observed_at: str
    context_ref: Optional[str] = None

    @validator("tenant_id", "model_name", "metric", "observed_at")
    def required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("finding fields must not be empty")
        return value

    @validator("value", "threshold")
    def finite_metric(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("drift metrics must be finite")
        return value

    @validator("context_ref")
    def hash_untrusted_reference(cls, value: Optional[str]) -> Optional[str]:
        if value and not value.startswith("sha256:"):
            return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
        return value


class DriftRemediationAdapter:
    """Creates idempotent runs; external governance actions remain injected adapters."""

    def __init__(
        self,
        runtime: AgentRuntime,
        diagnosis_tool: str = "diagnose_drift",
        promotion_adapter: Optional[Callable[[dict[str, Any]], Any]] = None,
        remediation_adapters: Optional[dict[str, Callable[[dict[str, Any]], Any]]] = None,
        adapters: Optional[RemediationAdapters] = None,
    ):
        self.runtime = runtime
        self.diagnosis_tool = diagnosis_tool
        self.promotion_adapter = promotion_adapter or (
            adapters.promote if adapters else None
        )
        self.notification_adapter = adapters.notify if adapters else None
        self.remediation_adapters = {
            **(adapters.actions() if adapters else {}),
            **(remediation_adapters or {}),
        }
        if diagnosis_tool not in runtime._tools:
            runtime.register_tool(
                ToolSpec(
                    name=diagnosis_tool,
                    idempotent=True,
                    output_schema={
                        "type": "object",
                        "required": ["diagnosis", "proposed_action"],
                    },
                ),
                self._diagnose_locally,
            )

    @classmethod
    def local(cls, runtime: AgentRuntime) -> "DriftRemediationAdapter":
        return cls(
            runtime,
            remediation_adapters={
                "canary_deploy": local_simulation_adapter,
                "rollback": local_simulation_adapter,
            },
        )

    @staticmethod
    def _diagnose_locally(payload: dict[str, Any]) -> dict[str, Any]:
        high_drift = payload["value"] >= payload["threshold"] * 2
        return {
            "diagnosis": (
                "severe_drift" if high_drift else "drift_detected"
            ),
            "proposed_action": "rollback" if high_drift else "canary_deploy",
        }

    async def consume(self, finding: DriftFinding) -> dict[str, Any]:
        fingerprint = hashlib.sha256(
            json.dumps(finding.dict(), sort_keys=True).encode()
        ).hexdigest()
        goal = json.dumps(
            {
                "tool": self.diagnosis_tool,
                "input": finding.dict(),
                "acceptance": {"required_keys": ["diagnosis"]},
            },
            sort_keys=True,
        )
        run = self.runtime.create_run(
            tenant_id=finding.tenant_id,
            goal=goal,
            budget=0.0,
            idempotency_key=f"drift:{fingerprint}",
        )
        self.runtime.record_evidence(
            run.run_id,
            finding.tenant_id,
            "drift_finding",
            {
                "finding_hash": fingerprint,
                "model_name": finding.model_name,
                "metric": finding.metric,
                "value": finding.value,
                "threshold": finding.threshold,
                "observed_at": finding.observed_at,
                "context_ref": finding.context_ref,
            },
        )
        if self.notification_adapter and not self.runtime.has_evidence(
            run.run_id,
            finding.tenant_id,
            "remediation_notification",
            "finding_hash",
            fingerprint,
        ):
            try:
                self.notification_adapter(
                    {
                        "tenant_id": finding.tenant_id,
                        "finding_hash": fingerprint,
                        "metric": finding.metric,
                    }
                )
                notification_status = "sent"
            except Exception:
                notification_status = "failed"
            self.runtime.record_evidence(
                run.run_id,
                finding.tenant_id,
                "remediation_notification",
                {"finding_hash": fingerprint, "status": notification_status},
            )
        outcome = await self.runtime.execute(run.run_id, finding.tenant_id)
        if (
            finding.metric in {"accuracy_drop", "performance_drift"}
            and "retrain" in self.remediation_adapters
        ):
            action = "retrain"
        else:
            action = (
                "rollback"
                if finding.value >= finding.threshold * 2
                else "canary_deploy"
            )
        high_risk = action in {"rollback", "retrain", "promote"}
        self.runtime.record_evidence(
            run.run_id,
            finding.tenant_id,
            "remediation_proposal",
            {
                "finding_hash": fingerprint,
                "action": action,
                "risk": "high" if high_risk else "low",
                "requires_approval": high_risk,
            },
        )
        remediation = None
        if outcome.get("status") == "SUCCEEDED" and action in self.remediation_adapters:
            tool_name = "apply_high_risk_remediation" if high_risk else "apply_low_risk_remediation"
            if tool_name not in self.runtime._tools:
                self.runtime.register_tool(
                    ToolSpec(
                        name=tool_name,
                        risk_level=RiskLevel.HIGH if high_risk else RiskLevel.LOW,
                        requires_approval=high_risk,
                        idempotent=True,
                        output_schema={"type": "object", "required": ["verified"]},
                    ),
                    self.remediation_adapters[action],
                )
            remediation_run = self.runtime.create_run(
                tenant_id=finding.tenant_id,
                goal=json.dumps(
                    {
                        "tool": tool_name,
                        "input": {
                            "model_name": finding.model_name,
                            "action": action,
                            "finding_hash": fingerprint,
                        },
                        "acceptance": {"required_keys": ["verified"]},
                    },
                    sort_keys=True,
                ),
                idempotency_key=f"drift-remediation:{fingerprint}",
            )
            remediation = await self.runtime.execute(
                remediation_run.run_id, finding.tenant_id
            )
            remediation = {"run_id": remediation_run.run_id, **remediation}
        if (
            remediation
            and remediation.get("status") == "SUCCEEDED"
            and self.promotion_adapter
            and not self.runtime.has_evidence(
                remediation["run_id"],
                finding.tenant_id,
                "remediation_promotion",
                "finding_hash",
                fingerprint,
            )
        ):
            try:
                self.promotion_adapter(
                    {
                        "tenant_id": finding.tenant_id,
                        "model_name": finding.model_name,
                        "run_id": remediation["run_id"],
                        "finding_hash": fingerprint,
                    }
                )
                promotion_status = "sent"
            except Exception:
                promotion_status = "failed"
            self.runtime.record_evidence(
                remediation["run_id"],
                finding.tenant_id,
                "remediation_promotion",
                {"finding_hash": fingerprint, "status": promotion_status},
            )
        return {
            "run_id": run.run_id,
            **outcome,
            "status": remediation["status"] if remediation else outcome["status"],
            "action": action,
            "remediation": remediation,
        }


def local_simulation_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Verify a local proposal without making external deployment changes."""
    return {"verified": True, "mode": "simulation", "action": payload["action"]}
