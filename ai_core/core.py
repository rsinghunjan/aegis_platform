"""AICore: one orchestration layer wrapping AIWorkflow with memory and observability."""

from __future__ import annotations

import os
from typing import Any

from ai_core.limits import AIQuotaExceeded, AIQuotaGuard
from ai_core.memory import AIMemory, query_digest
from ai_core.observability import AIObservability
from ai_core.policy import AIPolicyEngine
from ai_core.store import AICoreStore
from services.ai_workflow import AIWorkflow, AIWorkflowError


class AICore:
    def __init__(
        self,
        workflow: AIWorkflow,
        memory: AIMemory | None = None,
        observability: AIObservability | None = None,
        policy: AIPolicyEngine | None = None,
        quota: AIQuotaGuard | None = None,
    ):
        self.workflow = workflow
        self.quota = quota
        self.memory = memory or AIMemory()
        self.observability = observability or AIObservability()
        self.policy = policy or AIPolicyEngine(self.observability)

    @classmethod
    def durable(cls, workflow: AIWorkflow) -> "AICore":
        """Build a core whose memory and telemetry persist in SQL.

        Uses AEGIS_AI_CORE_DATABASE_URL when set, otherwise the workflow's own
        AI database so replicas sharing that database share state.
        """
        url = os.getenv("AEGIS_AI_CORE_DATABASE_URL") or getattr(
            getattr(workflow, "data_store", None), "database_url", None
        )
        quota = AIQuotaGuard.from_env(workflow)
        if not url:
            return cls(workflow, quota=quota)
        store = AICoreStore(url)
        return cls(
            workflow,
            memory=AIMemory(store=store),
            observability=AIObservability(store=store),
            quota=quota,
        )

    def ingest(self, tenant_id: str, document: str) -> dict[str, Any]:
        result = self.workflow.ingest(tenant_id, document)
        self.memory.remember(tenant_id, "ingest", {})
        return result

    def answer(self, tenant_id: str, query: str, max_tokens: int = 512) -> dict[str, Any]:
        if self.quota is not None and self.quota.enabled:
            try:
                self.quota.check(tenant_id, max_tokens)
            except AIQuotaExceeded as exc:
                self.observability.record(
                    tenant_id, "answer_rejected", f"quota_{exc.kind}"
                )
                raise
        repeated = self.memory.seen_query(tenant_id, query)
        try:
            result = self.workflow.answer(tenant_id, query, max_tokens)
        except AIWorkflowError:
            self.observability.record(tenant_id, "answer", "failed")
            raise
        self.observability.record(
            tenant_id,
            "answer",
            "succeeded",
            latency_ms=result["latency_ms"],
            cost_usd=result["cost_usd"],
            citations=len(result["citations"]),
        )
        self.memory.remember(
            tenant_id,
            "answer",
            {
                "query_sha256": query_digest(query),
                "request_id": result["request_id"],
                "citations": len(result["citations"]),
            },
        )
        result["repeated_query"] = repeated
        return result

    def record_agent_event(
        self, tenant_id: str, run_id: str, action: str, outcome: str
    ) -> None:
        self.memory.remember(
            tenant_id,
            "agent_run",
            {
                "run_id_sha256": query_digest(run_id),
                "action": action,
                "outcome": outcome,
            },
        )
        self.observability.record(tenant_id, f"agent_run_{action}", outcome)

    def insights(self, tenant_id: str) -> dict[str, Any]:
        return {
            "summary": self.observability.summary(tenant_id),
            "recommendations": self.observability.recommendations(tenant_id),
            "policy_suggestions": self.policy.suggest(tenant_id),
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.workflow, name)
