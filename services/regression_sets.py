"""Regression golden cases built from reviewed production failures.

Production audit rows are hash-only, so traces cannot become cases directly.
``failure_candidates`` lists failing requests by hash for operator review; an
operator then supplies reviewed (query, context, expected_facts) content that
``merge_cases`` validates, screens for PII/unsafe text, and de-duplicates.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from safety_checker import SafetyChecker

MAX_FIELD_CHARS = 8_000


def failure_candidates(usage: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": row["request_id"],
            "request_sha256": row["request_sha256"],
            "status": row["status"],
            "error_type": row.get("error_type"),
            "provider": row["provider"],
            "created_at": row["created_at"],
        }
        for row in usage
        if row["status"] != "succeeded"
    ]


def _case_id(case: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        (case["query"] + "\0" + case["context"]).encode("utf-8")
    ).hexdigest()
    return f"regression-{digest[:12]}"


def merge_cases(
    existing: list[dict[str, Any]],
    reviewed: list[Mapping[str, Any]],
    checker: SafetyChecker | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return (merged cases, rejected [{id, reason}])."""
    checker = checker or SafetyChecker()
    merged = list(existing)
    seen = {case["id"] for case in merged}
    rejected: list[dict[str, str]] = []
    for item in reviewed:
        try:
            query, context = item["query"], item["context"]
            facts = item["expected_facts"]
            source = item.get("source_request_sha256", "")
            if (
                not isinstance(query, str)
                or not isinstance(context, str)
                or not query.strip()
                or not context.strip()
                or not isinstance(facts, list)
                or not facts
                or any(not isinstance(f, str) or not f.strip() for f in facts)
                or len(query) > MAX_FIELD_CHARS
                or len(context) > MAX_FIELD_CHARS
            ):
                raise ValueError("invalid case shape")
            if any(checker.is_unsafe(text) for text in [query, context, *facts]):
                raise ValueError("contains PII or unsafe text")
            if any(fact.lower() not in context.lower() for fact in facts):
                raise ValueError("expected facts must appear in the context")
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"id": str(item.get("source_request_sha256", "?")), "reason": str(exc)})
            continue
        case = {
            "id": _case_id({"query": query, "context": context}),
            "query": query,
            "context": context,
            "expected_facts": facts,
            "source_request_sha256": source,
        }
        if case["id"] in seen:
            rejected.append({"id": case["id"], "reason": "duplicate"})
            continue
        seen.add(case["id"])
        merged.append(case)
    return merged, rejected
