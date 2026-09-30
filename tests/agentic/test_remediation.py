import asyncio

from agentic.remediation import DriftFinding, DriftRemediationAdapter
from agentic.runtime import AgentRuntime, AgentStore, ToolSpec


def test_drift_finding_creates_verified_idempotent_run(tmp_path):
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'drift.db'}"))
    calls = []
    runtime.register_tool(
        ToolSpec(name="diagnose_drift", idempotent=True),
        lambda payload: calls.append(payload) or {"diagnosis": "feature shift"},
    )
    adapter = DriftRemediationAdapter(runtime)
    finding = DriftFinding(
        tenant_id="tenant-a",
        model_name="classifier",
        metric="dataset_drift",
        value=0.4,
        threshold=0.1,
        observed_at="2026-09-30T00:00:00Z",
        context_ref="sha256:source",
    )

    first = asyncio.run(adapter.consume(finding))
    second = asyncio.run(adapter.consume(finding))
    assert first["status"] == second["status"] == "SUCCEEDED"
    assert first["run_id"] == second["run_id"]
    assert calls == [
        {
            "tenant_id": "tenant-a",
            "model_name": "classifier",
            "metric": "dataset_drift",
            "value": 0.4,
            "threshold": 0.1,
            "observed_at": "2026-09-30T00:00:00Z",
            "context_ref": "sha256:source",
        }
    ]
