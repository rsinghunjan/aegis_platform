"""Anomaly detection over AI operations, with optional LLM triage.

Detection is statistical (robust median/MAD z-scores over a recent window versus
a baseline), so it needs no labeled data. LLM triage only *classifies* findings
into a closed set of actions; its output is validated and never executes anything.
"""
from __future__ import annotations

import json
import statistics
from typing import Any, Callable, Iterable, Mapping

from agentic.remediation import DriftFinding

MIN_BASELINE = 20
MIN_RECENT = 5
Z_THRESHOLD = 3.5
ACTIONS = ("none", "notify", "require_approval", "lower_max_tokens", "reorder_providers", "rollback")


def _robust_z(baseline: list[float], recent: list[float]) -> float:
    median = statistics.median(baseline)
    mad = statistics.median(abs(x - median) for x in baseline)
    # Floor the scale so a flat baseline does not turn tiny noise into a huge z.
    scale = max(1.4826 * mad, 0.05 * abs(median), 1e-9)
    return (statistics.fmean(recent) - median) / scale


def detect_anomalies(
    usage: Iterable[Mapping[str, Any]], recent_size: int = 10
) -> list[dict[str, Any]]:
    """``usage`` is newest-first, as returned by ``AIWorkflow.list_usage``."""
    rows = list(usage)
    recent, baseline = rows[:recent_size], rows[recent_size:]
    if len(baseline) < MIN_BASELINE or len(recent) < MIN_RECENT:
        return []
    anomalies = []

    def check(metric: str, values: Callable[[Mapping[str, Any]], float], only_ok: bool) -> None:
        pick = (lambda r: r["status"] == "succeeded") if only_ok else (lambda r: True)
        base = [values(r) for r in baseline if pick(r)]
        now = [values(r) for r in recent if pick(r)]
        if len(base) < MIN_BASELINE or len(now) < MIN_RECENT:
            return
        z = _robust_z(base, now)
        if z >= Z_THRESHOLD:
            anomalies.append(
                {
                    "metric": metric,
                    "value": statistics.fmean(now),
                    "baseline": statistics.median(base),
                    "z_score": round(z, 2),
                }
            )

    check("latency_ms", lambda r: float(r["latency_ms"]), True)
    check("cost_usd", lambda r: float(r["cost_usd"]), True)
    # Failure rate: a failure burst against a mostly healthy baseline.
    base_fail = sum(r["status"] != "succeeded" for r in baseline) / len(baseline)
    now_fail = sum(r["status"] != "succeeded" for r in recent) / len(recent)
    if now_fail >= 0.5 and now_fail - base_fail >= 0.3:
        anomalies.append(
            {"metric": "failure_rate", "value": now_fail, "baseline": base_fail, "z_score": None}
        )
    return anomalies


def deterministic_triage(anomaly: Mapping[str, Any]) -> str:
    return {
        "failure_rate": "require_approval",
        "cost_usd": "lower_max_tokens",
        "latency_ms": "reorder_providers",
    }.get(anomaly["metric"], "notify")


def triage(
    anomalies: list[dict[str, Any]], generate: Callable[[str], str] | None = None
) -> list[dict[str, Any]]:
    """Attach a validated ``recommended_action``; fall back to rules on any LLM issue."""
    results = []
    for anomaly in anomalies:
        action, source = deterministic_triage(anomaly), "rules"
        if generate is not None:
            prompt = (
                "You triage anomalies in an AI platform. Choose exactly one action from "
                f"{list(ACTIONS)}. Reply only with JSON {{\"action\": \"...\"}}.\n"
                f"Anomaly: {json.dumps(anomaly, sort_keys=True)}"
            )
            try:
                chosen = json.loads(generate(prompt)).get("action")
                # An LLM may only pick from the allowlist; "rollback" always needs a human.
                if chosen in ACTIONS:
                    action, source = chosen, "llm"
            except Exception:
                pass
        results.append(
            {
                **anomaly,
                "recommended_action": action,
                "source": source,
                "requires_human_approval": action not in ("none", "notify"),
            }
        )
    return results


def to_drift_findings(
    tenant_id: str, model_name: str, anomalies: list[Mapping[str, Any]], observed_at: str
) -> list[DriftFinding]:
    """Map anomalies to findings the governed ``DriftRemediationAdapter`` consumes."""
    findings = []
    for anomaly in anomalies:
        baseline = float(anomaly["baseline"])
        value = float(anomaly["value"])
        threshold = max(baseline * 1.5, 1e-6)
        findings.append(
            DriftFinding(
                tenant_id=tenant_id,
                model_name=model_name,
                metric="accuracy_drop" if anomaly["metric"] == "failure_rate" else "performance_drift",
                value=value,
                threshold=threshold,
                observed_at=observed_at,
                context_ref=f"ai-anomaly:{anomaly['metric']}",
            )
        )
    return findings
