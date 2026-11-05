import pytest
from fastapi.testclient import TestClient

import production
from agentic.runtime import AgentRuntime, AgentStore
from production import create_app
from services import feedback_loop
from services.ai_storage import AIDataStore
from services.ai_workflow import AIWorkflow
from services.embeddings import LocalHashEmbeddingProvider
from services.inference import EchoProvider, ModelRouter
from services.prompt_registry import PromptRegistry, render, validate_template

PASS = lambda _t: {"status": "passed", "metrics": {"groundedness": 1.0}}  # noqa: E731
FAIL = lambda _t: {"status": "failed", "regressions": {"groundedness": {}}}  # noqa: E731
TEMPLATE = "Be brief. Context: {context} Question: {query}"


def test_template_validation():
    validate_template(TEMPLATE)
    for bad in ("", "no placeholders", "{context} only", TEMPLATE + " {other}"):
        with pytest.raises(ValueError):
            validate_template(bad)


def test_render_does_not_interpret_braces_in_content():
    assert render(TEMPLATE, "{query}", "{context}") == "Be brief. Context: {query} Question: {context}"


def test_promotion_requires_passing_eval_and_approver(tmp_path):
    registry = PromptRegistry(tmp_path / "prompts.json")
    candidate = registry.propose(TEMPLATE)
    with pytest.raises(ValueError):
        registry.promote(candidate["version"], "", PASS)
    with pytest.raises(ValueError, match="failed evaluation"):
        registry.promote(candidate["version"], "alice", FAIL)
    assert registry.active()["version"] == 1

    second = registry.propose(TEMPLATE + " ")
    registry.promote(second["version"], "alice", PASS)
    assert registry.active()["template"] == TEMPLATE + " "
    assert PromptRegistry(tmp_path / "prompts.json").active()["version"] == second["version"]
    statuses = {v["version"]: v["status"] for v in registry.list()}
    assert statuses == {1: "retired", candidate["version"]: "rejected", second["version"]: "active"}


def test_router_reorder_keeps_breakers_with_providers():
    a, b = EchoProvider(), EchoProvider()
    a.name, b.name = "a", "b"
    router = ModelRouter([a, b], failure_threshold=1)
    router.breakers["0:a"].record_failure()
    router.reorder(["b", "a"])
    assert router.provider_order() == ["b", "a"]
    assert router.breakers["1:a"].is_open and not router.breakers["0:b"].is_open
    with pytest.raises(ValueError):
        router.reorder(["a"])


def _usage(provider, ok, n, latency):
    return [
        {"provider": provider, "status": "succeeded" if i < ok else "failed",
         "latency_ms": latency, "cost_usd": 0.0}
        for i in range(n)
    ]


def _signal(i, ratings, failures=0):
    return {"run_id": f"r{i}", "run_id_sha256": f"h{i}", "status": "completed",
            "ratings": ratings, "verification_failures": failures,
            "planner_errors": 0, "run_sha256": f"x{i}"}


def test_candidates_cover_prompt_planner_routing_and_retraining():
    signals = [_signal(i, [2], failures=1) for i in range(60)]
    usage = _usage("a", 5, 20, 100) + _usage("b", 20, 20, 90)
    kinds = {c["kind"]: c for c in feedback_loop.build_candidates(signals, usage, ["a", "b"])}
    assert set(kinds) == {"prompt_tuning", "planner_tuning", "routing", "retraining"}
    assert kinds["routing"]["recommended_order"] == ["b", "a"]
    assert all(c["requires_human_approval"] for c in kinds.values())


def test_healthy_signals_produce_no_candidates():
    signals = [_signal(i, [5]) for i in range(10)]
    assert feedback_loop.build_candidates(signals, _usage("a", 20, 20, 50), ["a"]) == []


def test_manifest_is_hash_only_and_stable():
    manifest = feedback_loop.export_retraining_manifest([_signal(1, [4, 5])])
    assert manifest["rows"] == [
        {"run_id_sha256": "h1", "result_sha256": "x1", "mean_rating": 4.5, "verification_failures": 0}
    ]
    assert manifest == feedback_loop.export_retraining_manifest([_signal(1, [4, 5])])


@pytest.fixture
def client(tmp_path, monkeypatch):
    names = []
    for name in ("a", "b"):
        provider = EchoProvider()
        provider.name = name
        names.append(provider)
    workflow = AIWorkflow(
        inference_router=ModelRouter(names),
        embedding_provider=LocalHashEmbeddingProvider(dimensions=64),
        data_store=AIDataStore(f"sqlite:///{tmp_path / 'ai.db'}"),
    )
    runtime = AgentRuntime(store=AgentStore(f"sqlite:///{tmp_path / 'rt.db'}"))
    monkeypatch.setattr(production, "build_prompt_evaluator", lambda _w: PASS)
    app = create_app(
        runtime,
        tenant_authorizer=lambda _r, tenant, _a, _actor: tenant == "t",
        principal_resolver=lambda _r: {"principal_id": "alice"},
        ai_workflow=workflow,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_prompt_and_routing_endpoints_require_authorization_and_gate(client):
    body = {"tenant_id": "other", "template": TEMPLATE}
    assert client.post("/operator/ai/prompts", json=body).status_code in (403, 401)

    proposed = client.post("/operator/ai/prompts", json={**body, "tenant_id": "t"})
    assert proposed.status_code == 200
    version = proposed.json()["version"]
    assert client.post(
        "/operator/ai/prompts", json={"tenant_id": "t", "template": "bad"}
    ).status_code == 422
    promoted = client.post(f"/operator/ai/prompts/{version}/promote", json={"tenant_id": "t"})
    assert promoted.status_code == 200 and promoted.json()["approved_by"] == "alice"

    applied = client.post("/operator/ai/routing/apply", json={"tenant_id": "t", "order": ["b", "a"]})
    assert applied.json()["order"] == ["b", "a"]
    assert client.post(
        "/operator/ai/routing/apply", json={"tenant_id": "t", "order": ["z"]}
    ).status_code == 422

    improvements = client.get("/operator/ai/improvements", params={"tenant_id": "t"})
    assert improvements.status_code == 200
    assert improvements.json()["candidates"] == []
