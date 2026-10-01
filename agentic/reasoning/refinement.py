"""Iterative plan refinement."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class RefinementResult:
    final_plan: str
    iterations: int
    history: List[str] = field(default_factory=list)
    converged: bool = False


class PlanRefiner:
    """Repeatedly applies a critique+revise loop until the plan stabilizes
    or a maximum number of iterations is reached.

    ``critique_fn(plan) -> Optional[str]`` should return ``None``/empty
    string when the plan is acceptable, otherwise a critique message.
    ``revise_fn(plan, critique) -> str`` should return an improved plan.
    """

    def __init__(
        self,
        critique_fn: Callable[[str], Optional[str]],
        revise_fn: Callable[[str, str], str],
        max_iterations: int = 5,
    ):
        self.critique_fn = critique_fn
        self.revise_fn = revise_fn
        self.max_iterations = max_iterations

    def refine(self, initial_plan: str) -> RefinementResult:
        history = [initial_plan]
        plan = initial_plan
        for iteration in range(1, self.max_iterations + 1):
            critique = self.critique_fn(plan)
            if not critique:
                return RefinementResult(
                    final_plan=plan, iterations=iteration - 1, history=history, converged=True
                )
            plan = self.revise_fn(plan, critique)
            history.append(plan)
        return RefinementResult(
            final_plan=plan, iterations=self.max_iterations, history=history, converged=False
        )
