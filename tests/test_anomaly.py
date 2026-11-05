import pytest
from fastapi.testclient import TestClient

from agentic.runtime import AgentRuntime, AgentStore
from ai_core import anomaly
from production import create_app
from services.ai_storage import AIDataStore
from services.ai_workflow import AIWorkflow
from services.embeddings import LocalHashEmbeddingProvider
from services.inference import EchoProvider, ModelRouter


def row(latency=100.0, cost=0.01, status="succeeded"):
    return {"latency_ms": latency, "cost_usd": cost, "status": status}


def healthy(n=40):
    return [row(100 + (i % 5), 0.01) for i in range(n)]


def test_no_anomaly_on_healthy_traffic_or_thin_data():
    assert anomaly.detect_anomalies(healthy()) == []
    assert anomaly.detect_anomalies(healthy(10)) == []


def test_latency_spike_and_failure_burst_detected():
    spike = [row(900)] * 10 + healthy()
    assert [a["metric"] for a in anomaly.detect_anomalies(spike)] == ["latency_ms"]
    burst = [row(status="failed")] * 8 + [row()] * 2 + healthy()
    assert "failure_rate" in [a["metric"] for a in anomaly.detect_anomalies(burst)]


def test_triage_is_allowlisted_and_falls_back():
    found = [{"metric": "cost_usd", "value": 1, "baseline": 0.1, "z_score": 9}]
    assert anomaly.triage(found)[0]["recommended_action"] == "lower_max_tokens"
    llm = anomaly.triage(found, lambda _p: '{"action": "none"}')[0]
    assert (llm["recommended_action"], llm["source"]) == ("none", "llm")
    for bad in ('{"action": "rm -rf /"}', "not json"):
        out = anomaly.triage(found, lambda _p, b=bad: b)[0]
        assert (out["recommended_action"], out["source"]) == ("lower_max_tokens", "rules")
    assert out["requires_human_approval"] is True


def test_findings_map_to_governed_drift_metrics():
    finding = anomaly.to_drift_findings(
        "t", "m", [{"metric": "failure_rate", "value": 0.8, "baseline": 0.0}], "now"
    )[0]
    assert finding.metric == "accuracy_drop" and finding.value > finding.threshold


@pytest.fixture
def client(tmp_path):
    workflow = AIWorkflow(
        inference_router=ModelRouter([EchoProvider()]),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'ai.db'}"),
    )
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'rt.db'}"))
    app = create_app(
        runtime,
        tenant_authorizer=lambda _r, tenant, _a, _actor: tenant == "t",
        principal_resolver=lambda _r: {"principal_id": "alice"},
        ai_workflow=workflow,
    )
    with TestClient(app) as c:
        yield c


def test_scan_endpoint_and_remediation_requires_configuration(client):
    ok = client.post("/operator/ai/anomalies/scan", json={"tenant_id": "t"})
    assert ok.status_code == 200 and ok.json()["anomalies"] == []
    assert client.post("/operator/ai/anomalies/scan", json={"tenant_id": "x"}).status_code in (401, 403)


def test_auto_propose_never_approves(client):
    response = client.post(
        "/operator/ai/policy/auto-propose",
        json={"tenant_id": "t", "replay_traffic": [{"q": 1}]},
    )
    assert response.status_code == 200
    assert response.json() == {"proposals": [], "rejected": []}
    assert client.get("/operator/ai/policy/versions", params={"tenant_id": "t"}).json()["versions"] == []
