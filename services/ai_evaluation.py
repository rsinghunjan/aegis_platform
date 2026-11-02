"""Deterministic quality scoring for golden AI workflow evaluations."""
from __future__ import annotations

import re
from typing import Any, Callable

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "if", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "with",
}


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _content_words(value: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", value.lower())
        if len(word) > 2 and word not in _STOP_WORDS
    }


def score_answer(
    answer: str,
    context: str,
    expected_facts: list[str],
    citations: list[Any],
) -> dict[str, float]:
    """Score expected-fact coverage, source support, and citation presence."""
    normalized_answer = _normalized(answer)
    facts_found = sum(
        bool(_normalized(fact)) and _normalized(fact) in normalized_answer
        for fact in expected_facts
    )
    expected_fact_recall = facts_found / len(expected_facts) if expected_facts else 0.0
    answer_words = _content_words(answer)
    context_words = _content_words(context)
    groundedness = (
        len(answer_words & context_words) / len(answer_words) if answer_words else 0.0
    )
    citation_coverage = 1.0 if citations else 0.0
    return {
        "expected_fact_recall": expected_fact_recall,
        "groundedness": groundedness,
        "citation_coverage": citation_coverage,
    }


def tuning_recommendations(
    metrics: dict[str, float],
    feedback_ratings: list[int] | None = None,
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    recommendations = []
    metric_guidance = {
        "expected_fact_recall": (
            "retrieval",
            "Review retrieval relevance and prompt adherence against the missed facts.",
        ),
        "groundedness": (
            "prompt",
            "Review prompt grounding and the quality of retrieved source context.",
        ),
        "citation_coverage": (
            "retrieval",
            "Review retrieval citation coverage and answer citation behavior.",
        ),
    }
    for metric, score in metrics.items():
        if score < (thresholds or {}).get(metric, 1.0) and metric in metric_guidance:
            area, action = metric_guidance[metric]
            recommendations.append(
                {
                    "area": area,
                    "reason": metric,
                    "action": action,
                    "requires_human_approval": True,
                }
            )
    ratings = feedback_ratings or []
    if ratings and sum(ratings) / len(ratings) < 4.0:
        recommendations.append(
            {
                "area": "routing",
                "reason": "low_user_rating",
                "action": "Review provider and model routing against the golden dataset.",
                "requires_human_approval": True,
            }
        )
    return recommendations


def evaluate_dataset(
    cases: list[dict[str, Any]],
    answer_case: Callable[[dict[str, Any]], dict[str, Any]],
    thresholds: dict[str, float],
    feedback_ratings: list[int] | None = None,
) -> dict[str, Any]:
    """Run golden cases and return aggregate scores without retaining answers."""
    if not cases:
        raise ValueError("evaluation dataset must contain at least one case")
    metric_names = ("expected_fact_recall", "groundedness", "citation_coverage")
    if set(thresholds) != set(metric_names):
        raise ValueError("thresholds must define every evaluation metric")
    if any(not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("evaluation thresholds must be between 0 and 1")
    if feedback_ratings and any(
        isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5
        for rating in feedback_ratings
    ):
        raise ValueError("feedback ratings must be integers between 1 and 5")

    scores = []
    for case in cases:
        result = answer_case(case)
        if not isinstance(result, dict):
            raise ValueError("evaluation answer must be an object")
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        if not isinstance(answer, str) or not isinstance(citations, list):
            raise ValueError("evaluation answer and citations have invalid types")
        metrics = score_answer(answer, case["context"], case["expected_facts"], citations)
        scores.append({"case_id": case["id"], **metrics})

    aggregates = {
        name: sum(score[name] for score in scores) / len(scores)
        for name in metric_names
    }
    regressions = {
        name: {"score": score, "minimum": thresholds[name]}
        for name, score in aggregates.items()
        if score < thresholds[name]
    }
    return {
        "status": "passed" if not regressions else "failed",
        "case_count": len(scores),
        "metrics": aggregates,
        "thresholds": thresholds,
        "regressions": regressions,
        "feedback": {
            "rating_count": len(feedback_ratings or []),
            "average_rating": (
                sum(feedback_ratings) / len(feedback_ratings)
                if feedback_ratings
                else None
            ),
        },
        "tuning_recommendations": tuning_recommendations(
            aggregates, feedback_ratings, thresholds
        ),
        "cases": scores,
    }
