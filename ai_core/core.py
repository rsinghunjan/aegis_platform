"""AICore: one orchestration layer wrapping AIWorkflow with memory and observability."""

from __future__ import annotations

import json
import os
import time
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

    def plan_agent(
        self,
        tenant_id: str,
        run_id: str,
        goal: str,
        registered_tools: set[str],
        planner: Any,
        recovery: bool = False,
    ) -> Any:
        started = time.monotonic()
        try:
            if hasattr(planner, "plan_with_inference"):
                if self.quota is not None and self.quota.enabled:
                    self.quota.check(tenant_id, 512)

                def generate(prompt: str) -> str:
                    try:
                        result = self.workflow.generate(tenant_id, prompt)
                    except AIWorkflowError:
                        self.observability.record(tenant_id, "inference", "failed")
                        self.memory.remember(
                            tenant_id,
                            "inference",
                            {"prompt_sha256": query_digest(prompt), "outcome": "failed"},
                        )
                        raise
                    self.observability.record(
                        tenant_id,
                        "inference",
                        "succeeded",
                        latency_ms=result["latency_ms"],
                        cost_usd=result["cost_usd"],
                    )
                    self.memory.remember(
                        tenant_id,
                        "inference",
                        {
                            "prompt_sha256": query_digest(prompt),
                            "request_id": result["request_id"],
                            "outcome": "succeeded",
                        },
                    )
                    return result["text"]

                plan = planner.plan_with_inference(
                    goal, registered_tools, generate, recovery
                )
            else:
                plan = planner.plan(goal, registered_tools, recovery)
        except AIQuotaExceeded as exc:
            self.observability.record(
                tenant_id, "inference_rejected", f"quota_{exc.kind}"
            )
            self.record_agent_event(
                tenant_id,
                run_id,
                "plan",
                "rejected",
                latency_ms=(time.monotonic() - started) * 1000,
            )
            raise
        except Exception:
            self.record_agent_event(
                tenant_id,
                run_id,
                "plan",
                "failed",
                latency_ms=(time.monotonic() - started) * 1000,
                metadata={"goal_sha256": query_digest(goal)},
            )
            raise

        self.record_agent_event(
            tenant_id,
            run_id,
            "plan",
            "succeeded",
            latency_ms=(time.monotonic() - started) * 1000,
            metadata={
                "goal_sha256": query_digest(goal),
                "plan_sha256": query_digest(
                    json.dumps(plan.dict(), sort_keys=True, separators=(",", ":"))
                ),
                "step_count": len(plan.steps),
            },
        )
        return plan

    def record_agent_event(
        self,
        tenant_id: str,
        run_id: str,
        action: str,
        outcome: str,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "run_id_sha256": query_digest(run_id),
            "action": action,
            "outcome": outcome,
        }
        if metadata:
            event.update(metadata)
        self.memory.remember(
            tenant_id,
            "agent_run",
            event,
        )
        self.observability.record(
            tenant_id,
            f"agent_run_{action}",
            outcome,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

    def insights(self, tenant_id: str) -> dict[str, Any]:
        return {
            "summary": self.observability.summary(tenant_id),
            "recommendations": self.observability.recommendations(tenant_id),
            "policy_suggestions": self.policy.suggest(tenant_id),
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.workflow, name)
