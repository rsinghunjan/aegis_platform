# tests/test_api_e2e.py
import json
import uuid
from types import SimpleNamespace

import pytest

# Basic e2e tests: health, predict, jobs with mocked Celery enqueue

def test_health_and_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    # ready returns loaded_models count (int)
    assert "ready" in data and isinstance(data["ready"], bool)


def test_predict_json_allowed(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {"text": "Hello Aegis! This is a smoke test."}
    r = client.post("/v1/models/multimodal_demo/versions/v1/predict", json=payload, headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["model"] == "multimodal_demo"
    assert data["version"] == "v1"
    assert "request_id" in data
    assert "result" in data


def test_job_enqueue_and_status(client, admin_token, monkeypatch):
    # Mock Celery task apply_async to avoid requiring a real broker
    import api.tasks as tasks_module

    def fake_apply_async(args=None, queue=None, **kwargs):
        # return a small object with id attribute to mimic Celery AsyncResult
        return SimpleNamespace(id=f"fake-task-{uuid.uuid4()}")

    monkeypatch.setattr(tasks_module.process_job, "apply_async", fake_apply_async)

    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {"work_units": 2, "parameters": {"batch": True}}
    r = client.post("/v1/jobs", json=payload, headers=headers)
    assert r.status_code == 202, r.text
    j = r.json()
    assert "request_id" in j
    request_id = j["request_id"]

    # Immediately fetch job status; should be PENDING since worker didn't run
    r = client.get(f"/v1/jobs/{request_id}", headers=headers)
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["request_id"] == request_id
    assert st["status"] in ("PENDING", "RUNNING", "SUCCESS", "FAILED", "CANCELLED")


def test_registry_register_and_list(client, admin_token, tmp_path):
    # Create a tiny fake artifact file and upload it via registry API
    model_name = "unit_test_model"
    version = "v0"

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"dummy model artifact")

    files = {"artifact_file": ("artifact.bin", artifact.read_bytes())}
    data = {"model_name": model_name, "version": version, "stage": "staging", "description": "unit test model"}
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/v1/registry/register", data=data, files=files, headers=headers)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["model"] == model_name and out["version"] == version

    # list versions
    r = client.get(f"/v1/registry/{model_name}/versions", headers=headers)
    assert r.status_code == 200, r.text
    listing = r.json()
    assert listing["model"] == model_name
    assert any(v["version"] == version for v in listing["versions"])
