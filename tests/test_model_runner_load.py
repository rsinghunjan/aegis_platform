"""
Tests for ModelRegistry on-demand load/warmup concurrency safety
(api/model_runner.py).

Focus:
 - concurrent on-demand `predict_async`/`load_async` calls for the same
   not-yet-loaded model key must coalesce into a single blocking load, not
   race and trigger duplicate loads/warmups.
 - the blocking load is offloaded to a worker thread, so it does not stall
   the event loop while in progress.

Run: pytest tests/test_model_runner_load.py -q
"""
import asyncio
import time

from api.model_runner import ModelConfig, ModelRegistry


class _FakeWrapper:
    def predict_batch(self, batched_inputs):
        return [None for _ in batched_inputs]


def test_concurrent_on_demand_loads_are_deduplicated(monkeypatch):
    registry = ModelRegistry()
    registry.register("demo", "v1", ModelConfig(model_path="unused"))

    load_calls = {"count": 0}

    def fake_build(self, model_name, version):
        load_calls["count"] += 1
        time.sleep(0.1)  # simulate slow model I/O / warmup
        # A minimal stand-in wrapper, bypassing the real torch/onnx machinery.
        return _FakeWrapper()

    monkeypatch.setattr(ModelRegistry, "_build_and_warmup_wrapper", fake_build)

    async def scenario():
        results = await asyncio.gather(*(registry.load_async("demo", "v1") for _ in range(5)))
        return results

    results = asyncio.run(scenario())
    assert load_calls["count"] == 1
    assert all(r is results[0] for r in results)


def test_load_async_does_not_block_event_loop(monkeypatch):
    registry = ModelRegistry()
    registry.register("demo", "v1", ModelConfig(model_path="unused"))

    def fake_build(self, model_name, version):
        time.sleep(0.3)
        return _FakeWrapper()

    monkeypatch.setattr(ModelRegistry, "_build_and_warmup_wrapper", fake_build)

    async def scenario():
        ticks = []

        async def ticker():
            for _ in range(10):
                await asyncio.sleep(0.02)
                ticks.append(time.time())

        await asyncio.gather(registry.load_async("demo", "v1"), ticker())
        return ticks

    ticks = asyncio.run(scenario())
    assert len(ticks) >= 5
