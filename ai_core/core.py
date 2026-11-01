"""AICore: one orchestration layer wrapping AIWorkflow with memory and observability."""

from __future__ import annotations

from typing import Any

from ai_core.memory import AIMemory, query_digest
from ai_core.observability import AIObservability
from ai_core.policy import AIPolicyEngine
from services.ai_workflow import AIWorkflow, AIWorkflowError


class AICore:
    def __init__(
        self,
        workflow: AIWorkflow,
        memory: AIMemory | None = None,
        observability: AIObservability | None = None,
        policy: AIPolicyEngine | None = None,
    ):
        self.workflow = workflow
        self.memory = memory or AIMemory()
        self.observability = observability or AIObservability()
        self.policy = policy or AIPolicyEngine(self.observability)

    def ingest(self, tenant_id: str, document: str) -> dict[str, Any]:
        result = self.workflow.ingest(tenant_id, document)
        self.memory.remember(tenant_id, "ingest", {})
        return result

    def answer(self, tenant_id: str, query: str, max_tokens: int = 512) -> dict[str, Any]:
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

    def insights(self, tenant_id: str) -> dict[str, Any]:
        return {
            "summary": self.observability.summary(tenant_id),
            "recommendations": self.observability.recommendations(tenant_id),
            "policy_suggestions": self.policy.suggest(tenant_id),
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.workflow, name)
