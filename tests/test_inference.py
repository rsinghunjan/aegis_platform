import json
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("aegis_multimodal_ai_system.inference.server")

from aegis_multimodal_ai_system.inference.server import app, model_wrapper

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_predict_safe_input(monkeypatch):
    # Ensure the model is loaded with dummy model
    model_wrapper.load()

    req = {"items": [{"id": "1", "text": "hello safe world"}]}
    res = client.post("/predict", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["flagged"] is False
    assert data["predictions"][0]["output"]["label"] in ("short", "long")


def test_predict_flagged_input(monkeypatch):
    # Provide input that should be flagged by default safety keywords (e.g., contains 'kill')
    model_wrapper.load()
    req = {"items": [{"id": "1", "text": "I will kill you"}]}
    res = client.post("/predict", json=req)
    assert res.status_code == 403
