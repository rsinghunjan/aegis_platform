"""Tests for services/inference: registry, providers, router, tokens, batch."""
import time

import pytest

from services.inference import (
    ModelMetadata,
    ModelRegistry,
    EchoProvider,
    InferenceRequest,
    ModelRouter,
    NoProviderAvailableError,
    count_tokens,
    estimate_cost,
    BatchInferenceQueue,
)
from services.inference.providers import InferenceProvider, InferenceResult, ProviderError


def test_model_registry_register_and_get_latest():
    registry = ModelRegistry()
    v1 = registry.register(ModelMetadata(name="demo", version="1", source="custom", provider="local"))
    v2 = registry.register(ModelMetadata(name="demo", version="2", source="custom", provider="local"))

    assert registry.get("demo") == v2
    assert registry.get("demo", version="1") == v1
    assert [m.version for m in registry.list_versions("demo")] == ["1", "2"]
    assert v1.fingerprint != v2.fingerprint


def test_model_registry_missing_raises():
    registry = ModelRegistry()
    with pytest.raises(KeyError):
        registry.get("missing")


def test_model_registry_deregister_updates_latest():
    registry = ModelRegistry()
    registry.register(ModelMetadata(name="demo", version="1", source="custom", provider="local"))
    registry.register(ModelMetadata(name="demo", version="2", source="custom", provider="local"))
    registry.deregister("demo", "2")
    assert registry.get("demo").version == "1"


def test_echo_provider_generates_and_streams():
    provider = EchoProvider()
    request = InferenceRequest(prompt="hello world", max_tokens=1)
    result = provider.generate(request)
    assert result.text == "hello"
    assert result.provider == "local-echo"
    assert "".join(provider.stream(InferenceRequest(prompt="a b c"))) == "a b c "


class _FailingProvider(InferenceProvider):
    name = "failing"

    def generate(self, request: InferenceRequest) -> InferenceResult:
        raise ProviderError("boom")


class _UnavailableProvider(InferenceProvider):
    name = "unavailable"

    def is_available(self) -> bool:
        return False

    def generate(self, request: InferenceRequest) -> InferenceResult:  # pragma: no cover
        raise AssertionError("should not be called")


def test_router_falls_back_to_working_provider():
    router = ModelRouter([_UnavailableProvider(), _FailingProvider(), EchoProvider()])
    result = router.generate(InferenceRequest(prompt="hi"))
    assert result.provider == "local-echo"
    assert [a.provider for a in router.last_attempts] == ["unavailable", "failing", "local-echo"]


def test_router_raises_when_all_fail():
    router = ModelRouter([_UnavailableProvider(), _FailingProvider()])
    with pytest.raises(NoProviderAvailableError):
        router.generate(InferenceRequest(prompt="hi"))


def test_count_tokens_fallback_heuristic():
    assert count_tokens("") == 0
    assert count_tokens("hello world") >= 2


def test_estimate_cost():
    cost = estimate_cost(1000, 1000, cost_per_1k_input_tokens=0.01, cost_per_1k_output_tokens=0.02)
    assert cost == pytest.approx(0.03)


def test_batch_inference_queue_runs_jobs():
    router = ModelRouter([EchoProvider()])
    batch_queue = BatchInferenceQueue(router, workers=2)
    try:
        job_id = batch_queue.submit(InferenceRequest(prompt="batched request"))
        job = batch_queue.wait(job_id, timeout=5)
        assert job.error is None
        assert job.result is not None
        assert batch_queue.status(job_id) == "succeeded"
    finally:
        batch_queue.shutdown()
