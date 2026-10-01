"""Semantic goal decomposition.

Provides a deterministic, dependency-free decomposition strategy that
splits a goal into ordered subgoals using common delimiter heuristics
(numbered lists, "then"/"and then", semicolons). Callers that have an LLM
available can supply a ``decomposer_fn`` to replace the heuristic with a
model-backed semantic decomposition while reusing the same data shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Subgoal:
    text: str
    order: int
    depends_on: tuple = field(default_factory=tuple)


_SPLIT_PATTERN = re.compile(
    r"\s*(?:,?\s+and then\s+|\s*;\s*|\s*\n\s*\d+[\.\)]\s*|\s+then\s+)\s*",
    re.IGNORECASE,
)
_LEADING_NUMBER = re.compile(r"^\s*\d+[\.\)]\s*")
_LEADING_THEN = re.compile(r"^\s*(?:and\s+)?then\s+", re.IGNORECASE)


def decompose_goal(
    goal: str, decomposer_fn: Optional[Callable[[str], List[str]]] = None
) -> List[Subgoal]:
    """Decompose ``goal`` into an ordered list of :class:`Subgoal`.

    Each subgoal implicitly depends on the one before it (sequential
    decomposition), which is the conservative default for an agent runtime
    that executes plan steps in order.
    """
    if decomposer_fn is not None:
        parts = decomposer_fn(goal)
    else:
        goal = goal.strip()
        parts = [p.strip() for p in _SPLIT_PATTERN.split(goal) if p.strip()]
        parts = [_LEADING_NUMBER.sub("", p).strip() for p in parts]
        parts = [_LEADING_THEN.sub("", p).strip() for p in parts]

    if not parts:
        parts = [goal.strip()] if goal.strip() else []

    subgoals: List[Subgoal] = []
    for i, part in enumerate(parts):
        depends_on = (i - 1,) if i > 0 else ()
        subgoals.append(Subgoal(text=part, order=i, depends_on=depends_on))
    return subgoals
