"""Self-observation: the platform records and analyzes its own AI operations."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any


class AIObservability:
    def __init__(self, window: int = 500):
        self._records: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._lock = threading.Lock()

    def record(
        self,
        tenant_id: str,
        operation: str,
        outcome: str,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        citations: int = 0,
    ) -> None:
        with self._lock:
            self._records[tenant_id].append(
                {
                    "operation": operation,
                    "outcome": outcome,
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "citations": citations,
                }
            )

    def summary(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            records = list(self._records.get(tenant_id, ()))
        answers = [r for r in records if r["operation"] == "answer"]
        agent_events = [
            r for r in records if r["operation"].startswith("agent_run_")
        ]
        ok = [r for r in answers if r["outcome"] == "succeeded"]
        latencies = sorted(r["latency_ms"] for r in ok)
        p95 = (
            latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
            if latencies
            else 0.0
        )
        return {
            "tenant_id": tenant_id,
            "answers": len(answers),
            "failure_rate": (len(answers) - len(ok)) / len(answers) if answers else 0.0,
            "uncited_rate": sum(1 for r in ok if r["citations"] == 0) / len(ok)
            if ok
            else 0.0,
            "p95_latency_ms": p95,
            "total_cost_usd": sum(r["cost_usd"] for r in ok),
            "agent_events": len(agent_events),
            "agent_failures": sum(1 for r in agent_events if r["outcome"] == "failed"),
        }

    def recommendations(
        self,
        tenant_id: str,
        max_failure_rate: float = 0.1,
        max_uncited_rate: float = 0.3,
        max_p95_latency_ms: float = 5000.0,
        min_samples: int = 5,
    ) -> list[dict[str, str]]:
        stats = self.summary(tenant_id)
        if stats["answers"] < min_samples:
            return []
        found: list[dict[str, str]] = []
        if stats["failure_rate"] > max_failure_rate:
            found.append(
                {
                    "code": "high_failure_rate",
                    "action": "Check provider health and add a fallback provider to the router.",
                }
            )
        if stats["uncited_rate"] > max_uncited_rate:
            found.append(
                {
                    "code": "poor_retrieval_coverage",
                    "action": "Ingest more tenant knowledge; many answers have no supporting citations.",
                }
            )
        if stats["p95_latency_ms"] > max_p95_latency_ms:
            found.append(
                {
                    "code": "high_latency",
                    "action": "Route to a faster model or reduce max_tokens.",
                }
            )
        return found
