"""Cost tracking and budgeting for inference/cloud spend."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class BudgetExceededError(RuntimeError):
    def __init__(self, scope: str, spent: float, limit: float):
        super().__init__(f"budget exceeded for '{scope}': {spent:.4f} > {limit:.4f}")
        self.scope = scope
        self.spent = spent
        self.limit = limit


@dataclass
class Budget:
    scope: str
    limit: float
    period: str = "monthly"  # informational only; enforcement is caller-driven


@dataclass
class CostEntry:
    scope: str
    amount: float
    description: str = ""


class CostTracker:
    """Tracks cumulative spend per scope (tenant, project, model, etc.) and
    enforces configured budgets by raising :class:`BudgetExceededError`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._spend: Dict[str, float] = {}
        self._entries: Dict[str, List[CostEntry]] = {}
        self._budgets: Dict[str, Budget] = {}

    def set_budget(self, budget: Budget) -> None:
        with self._lock:
            self._budgets[budget.scope] = budget

    def record(self, scope: str, amount: float, description: str = "") -> float:
        """Record spend and return the new cumulative total for ``scope``.

        Raises :class:`BudgetExceededError` if a budget is configured for
        the scope and the cumulative spend now exceeds it.
        """
        with self._lock:
            total = self._spend.get(scope, 0.0) + amount
            self._spend[scope] = total
            self._entries.setdefault(scope, []).append(
                CostEntry(scope=scope, amount=amount, description=description)
            )
            budget = self._budgets.get(scope)
            if budget is not None and total > budget.limit:
                raise BudgetExceededError(scope, total, budget.limit)
            return total

    def spend(self, scope: str) -> float:
        with self._lock:
            return self._spend.get(scope, 0.0)

    def remaining_budget(self, scope: str) -> Optional[float]:
        with self._lock:
            budget = self._budgets.get(scope)
            if budget is None:
                return None
            return budget.limit - self._spend.get(scope, 0.0)

    def entries(self, scope: str) -> List[CostEntry]:
        with self._lock:
            return list(self._entries.get(scope, []))
