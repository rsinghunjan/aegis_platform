"""Per-tenant spend and request-rate enforcement for direct AI calls."""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


class AIQuotaExceeded(RuntimeError):
    """Raised before any provider call when a tenant is over budget or rate."""

    def __init__(self, kind: str, retry_after_seconds: int):
        super().__init__(f"AI {kind} limit exceeded")
        self.kind = kind
        self.retry_after_seconds = max(1, retry_after_seconds)


def _env_number(name: str, cast: Any) -> Any:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value or None  # 0 disables the limit


class AIQuotaGuard:
    """Checks limits against the durable inference audit table.

    Spend is measured over the current UTC day and request rate over the last
    60 seconds, both from rows shared by every replica. Requests rejected here
    write no audit row, so rejection cannot extend a tenant's own window.
    Enforcement is best-effort under concurrency: simultaneous requests may
    briefly overshoot a limit by the number of in-flight calls. The projected
    cost of a call uses the caller's ``max_tokens`` as a worst case for output
    and ignores retrieved context, so a single call can still slightly exceed
    the budget. Limits are disabled (None) unless configured.
    """

    def __init__(
        self,
        data_store: Any,
        daily_budget_usd: float | None = None,
        requests_per_minute: int | None = None,
        output_cost_per_1k: float = 0.0,
    ):
        self.data_store = data_store
        self.daily_budget_usd = daily_budget_usd
        self.requests_per_minute = requests_per_minute
        self.output_cost_per_1k = output_cost_per_1k

    @classmethod
    def from_env(cls, workflow: Any) -> "AIQuotaGuard | None":
        store = getattr(workflow, "data_store", None)
        budget = _env_number("AEGIS_AI_TENANT_DAILY_BUDGET_USD", float)
        rpm = _env_number("AEGIS_AI_TENANT_REQUESTS_PER_MINUTE", int)
        if store is None or (budget is None and rpm is None):
            return None
        output_rate = float(getattr(workflow, "output_cost_per_1k", 0.0) or 0.0)
        input_rate = float(getattr(workflow, "input_cost_per_1k", 0.0) or 0.0)
        if budget is not None and output_rate == 0 and input_rate == 0:
            logger.warning(
                "AEGIS_AI_TENANT_DAILY_BUDGET_USD is set but token cost rates are "
                "zero; set AEGIS_LLM_INPUT_COST_PER_1K and "
                "AEGIS_LLM_OUTPUT_COST_PER_1K or the budget will never trigger"
            )
        return cls(store, budget, rpm, output_rate)

    @property
    def enabled(self) -> bool:
        return self.daily_budget_usd is not None or self.requests_per_minute is not None

    def check(
        self, tenant_id: str, max_tokens: int = 0, now: datetime | None = None
    ) -> None:
        now = now or datetime.now(timezone.utc)
        if self.requests_per_minute is not None:
            count, _ = self.data_store.usage_since(
                tenant_id, now - timedelta(seconds=60)
            )
            if count >= self.requests_per_minute:
                raise AIQuotaExceeded("rate", 60)
        if self.daily_budget_usd is not None:
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            _, spent = self.data_store.usage_since(tenant_id, day_start)
            projected = spent + (max_tokens / 1000.0) * self.output_cost_per_1k
            if spent >= self.daily_budget_usd or projected > self.daily_budget_usd:
                retry = int((day_start + timedelta(days=1) - now).total_seconds())
                raise AIQuotaExceeded("budget", retry)
