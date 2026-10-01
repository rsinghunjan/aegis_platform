from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aegis_db.models import AuditLog, Job, Run, Tenant
from governance import api
from governance import promotion
from governance.promotion import (
    PromotionError,
    initialize_promotion_schema,
    promote_run,
    resolve_promoted_model,
)


class FakeMlflowClient:
    def __init__(self, run):
        self.run = run

    def get_run(self, run_id):
        if run_id != self.run.info.run_id:
            raise KeyError(run_id)
        return self.run


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'promotion.db'}")
    initialize_promotion_schema(engine)
    yield sessionmaker(bind=engine)
    engine.dispose()


def make_run(status="FINISHED"):
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id="mlflow-run-1",
            status=status,
            artifact_uri="file:///models/model-1",
            start_time=1_700_000_000_000,
            end_time=1_700_000_001_000,
        ),
        data=SimpleNamespace(metrics={"accuracy": 0.97}, params={"seed": "7"}),
    )


def test_promote_persists_governance_and_resolves_model(session_factory):
    client = FakeMlflowClient(make_run())

    result = promote_run(
        "mlflow-run-1",
        client=client,
        model_name="fraud-detector",
        version="v2",
        tenant_id="tenant-a",
        actor="alice",
        notes="passed evaluation",
        session_factory=session_factory,
    )

    assert result["decision"] == "approved"
    assert resolve_promoted_model(
        "fraud-detector", tenant_id="tenant-a", session_factory=session_factory
    ) == {
        "model_name": "fraud-detector",
        "version": "v2",
        "run_id": "mlflow-run-1",
        "artifact_uri": "file:///models/model-1",
        "governance_evidence": {},
        "tenant_id": "tenant-a",
        "job_id": result["job_id"],
    }
    with session_factory() as session:
        assert session.get(Tenant, "tenant-a").name == "tenant-a"
        assert session.get(Run, "mlflow-run-1").metrics_json == {"accuracy": 0.97}
        job = session.get(Job, result["job_id"])
        assert job.payload_json["artifact_uri"] == "file:///models/model-1"
        audit = session.query(AuditLog).one()
        assert audit.decision == "approved"
        assert audit.actor == "alice"


def test_invalid_run_records_rejected_audit_and_is_not_resolvable(session_factory):
    client = FakeMlflowClient(make_run(status="RUNNING"))

    with pytest.raises(PromotionError, match="FINISHED"):
        promote_run(
            "mlflow-run-1",
            client=client,
            model_name="fraud-detector",
            session_factory=session_factory,
        )

    with session_factory() as session:
        assert session.query(Run).count() == 0
        assert session.query(Job).one().status == "rejected"
        assert session.query(AuditLog).one().decision == "rejected"
    assert (
        resolve_promoted_model("fraud-detector", session_factory=session_factory)
        is None
    )


def test_database_url_builds_local_promotion_sessionmaker(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'configured.db'}")
    monkeypatch.setattr(promotion, "_engine", None)
    monkeypatch.setattr(promotion, "_session_factory", None)

    factory = promotion.get_promotion_sessionmaker()

    with factory() as session:
        assert session.query(Tenant).count() == 0
    assert (tmp_path / "configured.db").exists()


def test_governance_routes_keep_promotion_and_run_listing(monkeypatch, session_factory):
    run = make_run()

    class ApiMlflowClient(FakeMlflowClient):
        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="experiment-1")

        def search_runs(self, experiment_ids, max_results):
            return [run]

    monkeypatch.setattr(api, "get_mlflow_client", lambda: ApiMlflowClient(run))
    monkeypatch.setattr(
        promotion, "get_promotion_sessionmaker", lambda: session_factory
    )
    monkeypatch.setitem(
        api.app.config,
        "AEGIS_GOVERNANCE_AUTHORIZER",
        lambda _request, action, tenant_id: {
            "actor_id": "trusted-operator",
            "approval_id": "approval-42",
            "policy_version": "policy-v3",
            "decision_evidence_sha256": "a" * 64,
            "artifact_sha256": "b" * 64,
            "signature_verified": True,
        }
        if tenant_id == "default" and action in {"model.read", "model.promote"}
        else None,
    )
    client = api.app.test_client()

    runs_response = client.get("/runs/demo")
    promotion_response = client.post(
        "/promote",
        json={
            "run_id": "mlflow-run-1",
            "model_name": "fraud-detector",
            "user": "alice",
        },
    )
    resolve_response = client.get("/models/fraud-detector/promoted")

    assert runs_response.status_code == 200
    assert runs_response.json[0]["run_id"] == "mlflow-run-1"
    assert promotion_response.status_code == 200
    assert promotion_response.json == {"ok": True, "run_id": "mlflow-run-1"}
    assert resolve_response.status_code == 200
    assert resolve_response.json["artifact_uri"] == "file:///models/model-1"
    assert resolve_response.json["governance_evidence"] == {
        "approval_id": "approval-42",
        "policy_version": "policy-v3",
        "decision_evidence_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
        "signature_verified": True,
    }
    with session_factory() as session:
        assert session.query(AuditLog).one().actor == "trusted-operator"
