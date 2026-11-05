"""Turn user ratings, verification outcomes and inference audit into improvement candidates.

Nothing here changes production behavior. Every candidate is advisory and
requires human approval; prompt candidates additionally require a passing
golden evaluation (``PromptRegistry.promote``) and retraining candidates are
exported as hash-only manifests for the signed, gated model pipeline.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping

LOW_RATING = 3.5
MIN_SAMPLES = 5
MIN_PROVIDER_SAMPLES = 10


def collect_run_signals(runtime: Any, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Per-run rating and verification signals; holds no goals, notes or outputs."""
    signals = []
    for run in runtime.list_runs(tenant_id, limit=limit):
        evidence = runtime.list_evidence(run.run_id, tenant_id)
        ratings = [
            item.metadata["rating"]
            for item in evidence
            if item.kind == "user_feedback"
            and isinstance(item.metadata.get("rating"), int)
            and not isinstance(item.metadata["rating"], bool)
            and 1 <= item.metadata["rating"] <= 5
        ]
        verified = [item for item in evidence if "verif" in item.kind]
        failed = [
            item
            for item in verified
            if item.metadata.get("passed") is False
            or item.metadata.get("status") == "failed"
        ]
        signals.append(
            {
                "run_id": run.run_id,
                "run_id_sha256": hashlib.sha256(run.run_id.encode()).hexdigest(),
                "status": str(run.status),
                "ratings": ratings,
                "verification_failures": len(failed),
                "planner_errors": sum(item.kind == "planner_error" for item in evidence),
                "run_sha256": run.final_result_hash,
            }
        )
    return signals


def provider_stats(usage: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usage:
        grouped[row["provider"]].append(row)
    stats = {}
    for provider, rows in grouped.items():
        ok = [row for row in rows if row["status"] == "succeeded"]
        stats[provider] = {
            "requests": len(rows),
            "success_rate": len(ok) / len(rows),
            "avg_latency_ms": sum(r["latency_ms"] for r in ok) / len(ok) if ok else 0.0,
            "avg_cost_usd": sum(r["cost_usd"] for r in ok) / len(ok) if ok else 0.0,
        }
    return stats


def recommend_provider_order(
    stats: Mapping[str, Mapping[str, float]], current: list[str]
) -> list[str] | None:
    """Rank providers by reliability, then latency; None when evidence is too thin."""
    known = [p for p in current if stats.get(p, {}).get("requests", 0) >= MIN_PROVIDER_SAMPLES]
    if len(known) < 2:
        return None
    ranked = sorted(
        known, key=lambda p: (-round(stats[p]["success_rate"], 2), stats[p]["avg_latency_ms"])
    )
    rest = [p for p in current if p not in ranked]
    order = ranked + rest
    return order if order != current else None


def build_candidates(
    signals: list[Mapping[str, Any]],
    usage: Iterable[Mapping[str, Any]],
    current_provider_order: list[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ratings = [r for s in signals for r in s["ratings"]]
    if len(ratings) >= MIN_SAMPLES:
        average = sum(ratings) / len(ratings)
        if average < LOW_RATING:
            candidates.append(
                {
                    "kind": "prompt_tuning",
                    "reason": f"average user rating {average:.2f} below {LOW_RATING}",
                    "action": "Propose a prompt candidate and promote it only via "
                    "PromptRegistry.promote (golden evaluation gate).",
                }
            )
    failing = [s for s in signals if s["verification_failures"] or s["planner_errors"]]
    if len(signals) >= MIN_SAMPLES and len(failing) / len(signals) > 0.2:
        candidates.append(
            {
                "kind": "planner_tuning",
                "reason": f"{len(failing)}/{len(signals)} runs had verification or planner failures",
                "action": "Extend data/golden_datasets/agent_plans.json with these "
                "goals and re-run scripts/evaluate_agent_plans.py.",
            }
        )
    order = recommend_provider_order(provider_stats(usage), current_provider_order)
    if order:
        candidates.append(
            {
                "kind": "routing",
                "reason": "provider reliability/latency differs from the configured order",
                "recommended_order": order,
                "action": "Apply with an explicit approval via the routing endpoint.",
            }
        )
    labeled = [
        s for s in signals if s["ratings"] and s["run_sha256"]
    ]
    if len(labeled) >= 50:
        candidates.append(
            {
                "kind": "retraining",
                "reason": f"{len(labeled)} rated runs are available as labeled outcomes",
                "action": "Export a manifest and run it through the signed, gated "
                "retraining/promotion workflow.",
            }
        )
    for candidate in candidates:
        candidate["requires_human_approval"] = True
    return candidates


def export_retraining_manifest(signals: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Hash-only labeled outcomes; the training job resolves content under access control."""
    rows = [
        {
            "run_id_sha256": s["run_id_sha256"],
            "result_sha256": s["run_sha256"],
            "mean_rating": sum(s["ratings"]) / len(s["ratings"]),
            "verification_failures": s["verification_failures"],
        }
        for s in signals
        if s["ratings"] and s["run_sha256"]
    ]
    body = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "aegis.feedback-manifest/v1",
        "count": len(rows),
        "manifest_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "rows": rows,
    }
