import pytest

from services.ai_workflow import AIWorkflow
from services.inference.gateway import build_inference_router, configured_providers

ENV = (
    "AEGIS_ALLOW_DEV_PROVIDERS",
    "AEGIS_LLM_API_KEY",
    "OPENAI_API_KEY",
    "AEGIS_INFERENCE_PROVIDERS",
    "AEGIS_EMBEDDING_PROVIDER",
)


@pytest.fixture
def strict(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_startup_fails_without_real_provider(strict):
    with pytest.raises(RuntimeError, match="no real model provider"):
        build_inference_router()


def test_echo_rejected_outside_dev(strict):
    strict.setenv("AEGIS_INFERENCE_PROVIDERS", "echo")
    with pytest.raises(ValueError, match="echo"):
        configured_providers()


def test_dev_flag_allows_echo(strict):
    strict.setenv("AEGIS_ALLOW_DEV_PROVIDERS", "true")
    assert configured_providers()[0].name == "local-echo"


def test_real_provider_accepted(strict):
    strict.setenv("AEGIS_LLM_API_KEY", "k")
    assert configured_providers()[0].name == "openai"


def test_local_hash_embeddings_rejected_outside_dev(strict):
    strict.setenv("AEGIS_EMBEDDING_PROVIDER", "local-hash")
    with pytest.raises(RuntimeError, match="local-hash"):
        AIWorkflow._default_embeddings()


def test_embeddings_default_to_real_model_outside_dev(strict):
    with pytest.raises(RuntimeError, match="unavailable"):
        AIWorkflow._default_embeddings()
