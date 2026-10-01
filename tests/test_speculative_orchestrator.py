"""
Tests for services/speculative/orchestrator.py in-memory queue bounding
(backpressure) and non-blocking event logging.

Note: orchestrator.py imports `services.excel.dlp_rules` and
`services.audit.audit_logger`, which do not exist in this checkout
(pre-existing, unrelated to this change). We install lightweight stand-in
modules in sys.modules before import so the orchestrator module (and the
behavior we're actually testing: the bounded in-memory scoring queue) can be
exercised in isolation.

Run: pytest tests/test_speculative_orchestrator.py -q
"""
import asyncio
import sys
import types

import pytest


def _install_stub_dependencies():
    if "services.excel" not in sys.modules:
        excel_pkg = types.ModuleType("services.excel")
        sys.modules["services.excel"] = excel_pkg
    dlp_mod = types.ModuleType("services.excel.dlp_rules")
    dlp_mod.contains_pii = lambda text: False
    dlp_mod.redact = lambda text: text
    sys.modules["services.excel.dlp_rules"] = dlp_mod

    if "services.audit" not in sys.modules:
        audit_pkg = types.ModuleType("services.audit")
        sys.modules["services.audit"] = audit_pkg
    audit_mod = types.ModuleType("services.audit.audit_logger")
    audit_mod.log_event = lambda *args, **kwargs: None
    sys.modules["services.audit.audit_logger"] = audit_mod


_install_stub_dependencies()

import services.speculative.orchestrator as orch  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_queue_state():
    orch._BATCH_QUEUE.clear()
    orch._PENDING_FUTURES.clear()
    yield
    orch._BATCH_QUEUE.clear()
    orch._PENDING_FUTURES.clear()


def test_enqueue_rejects_when_queue_is_full(monkeypatch):
    monkeypatch.setattr(orch, "MAX_QUEUE_SIZE", 2)

    async def scenario():
        # Fill the queue to capacity without a worker draining it.
        fut1 = asyncio.get_event_loop().create_future()
        fut2 = asyncio.get_event_loop().create_future()
        orch._BATCH_QUEUE.append({"job_id": "a", "context": "c", "draft": "d"})
        orch._BATCH_QUEUE.append({"job_id": "b", "context": "c", "draft": "d"})
        orch._PENDING_FUTURES["a"] = fut1
        orch._PENDING_FUTURES["b"] = fut2

        with pytest.raises(RuntimeError, match="scoring queue full"):
            await orch.enqueue_for_scoring("context", "draft")

    asyncio.run(scenario())
    assert len(orch._BATCH_QUEUE) == 2  # rejected job was not appended


def test_enqueue_succeeds_under_capacity():
    async def scenario():
        async def drain_one():
            # Simulate the batch worker completing the single queued job.
            await asyncio.sleep(0.01)
            job = orch._BATCH_QUEUE.pop(0)
            fut = orch._PENDING_FUTURES.pop(job["job_id"])
            fut.set_result({"tokens": ["x"], "token_scores": [0.0]})

        result, _ = await asyncio.gather(
            orch.enqueue_for_scoring("context", "draft"),
            drain_one(),
        )
        return result

    result = asyncio.run(scenario())
    assert result == {"tokens": ["x"], "token_scores": [0.0]}
    assert len(orch._BATCH_QUEUE) == 0
