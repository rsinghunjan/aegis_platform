import importlib
import sys
from types import ModuleType

from fastapi.testclient import TestClient


def load_app(monkeypatch):
    core_module = ModuleType("aegis_multimodal_ai_system.multimodal_ai_system")

    class FakeSystem:
        def generate_safe_response(self, query):
            return f"handled: {query}"

    core_module.MultimodalAISystem = FakeSystem
    monkeypatch.setitem(
        sys.modules, "aegis_multimodal_ai_system.multimodal_ai_system", core_module
    )
    sys.modules.pop("aegis_multimodal_ai_system.app", None)
    return importlib.import_module("aegis_multimodal_ai_system.app")


def test_generate_endpoint_and_existing_root(monkeypatch):
    app_module = load_app(monkeypatch)
    client = TestClient(app_module.app)

    assert client.get("/").json() == {
        "status": "Aegis Multimodal AI System is running."
    }
    response = client.post("/generate", json={"query": "hello"})
    assert response.status_code == 200
    assert response.json() == {"response": "handled: hello"}


def test_generate_endpoint_rejects_unsafe_input(monkeypatch):
    app_module = load_app(monkeypatch)
    monkeypatch.setattr(app_module.safety_checker, "is_unsafe", lambda text: True)

    response = TestClient(app_module.app).post("/generate", json={"query": "unsafe"})

    assert response.status_code == 422
    assert response.json()["detail"] == "query blocked by safety checker"


def test_generate_endpoint_rejects_unsafe_output(monkeypatch):
    app_module = load_app(monkeypatch)
    monkeypatch.setattr(
        app_module.safety_checker,
        "is_unsafe",
        lambda text: text.startswith("handled:"),
    )

    response = TestClient(app_module.app).post("/generate", json={"query": "hello"})

    assert response.status_code == 422
    assert response.json()["detail"] == "response blocked by safety checker"


def test_generate_endpoint_returns_generic_error(monkeypatch):
    app_module = load_app(monkeypatch)
    monkeypatch.setattr(
        app_module.ai_system,
        "generate_safe_response",
        lambda query: (_ for _ in ()).throw(RuntimeError("private details")),
    )

    response = TestClient(app_module.app).post("/generate", json={"query": "hello"})

    assert response.status_code == 500
    assert response.json()["detail"] == "generation failed"
