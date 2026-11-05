"""Plan-quality scoring for LLM agent planners against golden goals."""
from __future__ import annotations

from typing import Any, Callable

METRICS = ("llm_plan_rate", "tool_sequence_accuracy", "input_key_recall")


def score_plan(
    plan: Any, source: str, expected_tools: list[str], expected_input_keys: list[str]
) -> dict[str, float]:
    steps = list(getattr(plan, "steps", []) or [])
    tools = [step.tool_name for step in steps]
    keys = {key for step in steps for key in (step.input or {})}
    return {
        "llm_plan_rate": 1.0 if source == "llm" else 0.0,
        "tool_sequence_accuracy": 1.0 if tools == expected_tools else 0.0,
        "input_key_recall": (
            sum(key in keys for key in expected_input_keys) / len(expected_input_keys)
            if expected_input_keys
            else 1.0
        ),
    }


def evaluate_plans(
    cases: list[dict[str, Any]],
    plan_case: Callable[[dict[str, Any]], tuple[Any, str]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Run golden planning cases; ``plan_case`` returns ``(plan, source)``."""
    if not cases:
        raise ValueError("plan evaluation dataset must contain at least one case")
    if set(thresholds) != set(METRICS):
        raise ValueError("thresholds must define every plan metric")
    if any(not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("plan thresholds must be between 0 and 1")
    scores = []
    for case in cases:
        try:
            plan, source = plan_case(case)
            metrics = score_plan(
                plan,
                source,
                case["expected_tools"],
                case.get("expected_input_keys", []),
            )
        except Exception:
            metrics = {name: 0.0 for name in METRICS}
        scores.append({"case_id": case["id"], **metrics})
    aggregates = {
        name: sum(score[name] for score in scores) / len(scores) for name in METRICS
    }
    regressions = {
        name: {"score": value, "minimum": thresholds[name]}
        for name, value in aggregates.items()
        if value < thresholds[name]
    }
    return {
        "status": "passed" if not regressions else "failed",
        "case_count": len(scores),
        "metrics": aggregates,
        "thresholds": thresholds,
        "regressions": regressions,
        "cases": scores,
    }
