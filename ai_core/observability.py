"""Self-observation: the platform records and analyzes its own AI operations."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from typing import Any

from ai_core.store import AICoreStore

logger = logging.getLogger(__name__)


class AIObservability:
    def __init__(self, window: int = 500, store: AICoreStore | None = None):
        self.window = window
        self.store = store
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
        record = {
            "operation": operation,
            "outcome": outcome,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "citations": citations,
        }
        with self._lock:
            self._records[tenant_id].append(record)
        if self.store is not None:
            try:
                self.store.append(tenant_id, self._kind(operation), record)
            except Exception as exc:
                logger.warning("AI telemetry persistence failed: %s", type(exc).__name__)

    @staticmethod
    def _kind(operation: str) -> str:
        return "obs:agent" if operation.startswith("agent_run_") else "obs:answer"

    def _load(self, tenant_id: str) -> list[dict[str, Any]]:
        if self.store is not None:
            try:
                return [
                    {k: v for k, v in event.items() if k not in ("kind", "at")}
                    for event in self.store.recent(
                        tenant_id, ["obs:answer", "obs:agent"], self.window
                    )
                ]
            except Exception as exc:
                logger.warning("AI telemetry read failed: %s", type(exc).__name__)
        with self._lock:
            return list(self._records.get(tenant_id, ()))

    def summary(self, tenant_id: str) -> dict[str, Any]:
        records = self._load(tenant_id)
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
