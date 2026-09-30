"""Deterministic adapter from monitoring findings to governed agent runs."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional

from pydantic import BaseModel

from agentic.runtime import AgentRuntime


class DriftFinding(BaseModel):
    tenant_id: str
    model_name: str
    metric: str
    value: float
    threshold: float
    observed_at: str
    context_ref: Optional[str] = None


class DriftRemediationAdapter:
    """Creates idempotent runs; external governance actions remain injected adapters."""

    def __init__(
        self,
        runtime: AgentRuntime,
        diagnosis_tool: str = "diagnose_drift",
        promotion_adapter: Optional[Callable[[dict[str, Any]], Any]] = None,
    ):
        self.runtime = runtime
        self.diagnosis_tool = diagnosis_tool
        self.promotion_adapter = promotion_adapter

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
        outcome = await self.runtime.execute(run.run_id, finding.tenant_id)
        if outcome.get("status") == "SUCCEEDED" and self.promotion_adapter:
            self.promotion_adapter(
                {
                    "tenant_id": finding.tenant_id,
                    "model_name": finding.model_name,
                    "run_id": run.run_id,
                    "finding_hash": fingerprint,
                }
            )
        return {"run_id": run.run_id, **outcome}
