"""Plan verification and backtracking support.

Operates on simple plan representations (a list of step dicts with
``tool`` and ``input`` keys), matching the shape produced by Aegis's
existing deterministic/LLM planners, without importing those modules
directly (keeping this package decoupled and independently testable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class VerificationIssue:
    step_index: int
    message: str
    severity: str = "error"  # "error" | "warning"


def verify_plan(
    steps: Sequence[Dict[str, Any]],
    allowed_tools: Optional[Sequence[str]] = None,
    custom_checks: Optional[Sequence[Callable[[int, Dict[str, Any]], Optional[str]]]] = None,
) -> List[VerificationIssue]:
    """Run structural verification checks over a plan's steps.

    Checks performed:
    - every step has a non-empty ``tool`` name
    - every step's ``tool`` is in ``allowed_tools`` (if provided)
    - no step is a no-op duplicate of the step immediately before it
    - any ``custom_checks`` callables are invoked per-step and may return
      an issue message (or ``None`` if the step passes)
    """
    issues: List[VerificationIssue] = []
    previous: Optional[Dict[str, Any]] = None

    for i, step in enumerate(steps):
        tool = step.get("tool")
        if not tool:
            issues.append(VerificationIssue(i, "step is missing a 'tool' name"))
        elif allowed_tools is not None and tool not in allowed_tools:
            issues.append(VerificationIssue(i, f"tool '{tool}' is not an allowed capability"))

        if previous is not None and step == previous:
            issues.append(
                VerificationIssue(i, "step is an exact duplicate of the previous step", severity="warning")
            )

        for check in custom_checks or []:
            message = check(i, step)
            if message:
                issues.append(VerificationIssue(i, message))

        previous = step

    return issues


def backtrack_to_last_valid_step(
    steps: Sequence[Dict[str, Any]], issues: Sequence[VerificationIssue]
) -> List[Dict[str, Any]]:
    """Truncate ``steps`` to drop everything from the first error onward.

    Warnings do not trigger truncation; only ``severity == "error"`` issues
    do, since warnings represent non-fatal plan quality concerns.
    """
    error_indices = [issue.step_index for issue in issues if issue.severity == "error"]
    if not error_indices:
        return list(steps)
    first_error = min(error_indices)
    return list(steps[:first_error])
