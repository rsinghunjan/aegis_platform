"""Advisory policy reasoning. Suggestions are never applied automatically."""

from __future__ import annotations

from typing import Any

from ai_core.observability import AIObservability


class AIPolicyEngine:
    def __init__(
        self,
        observability: AIObservability,
        cost_budget_usd: float | None = None,
        min_samples: int = 5,
    ):
        self.observability = observability
        self.cost_budget_usd = cost_budget_usd
        self.min_samples = min_samples

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
