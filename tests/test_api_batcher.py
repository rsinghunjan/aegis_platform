"""
Unit tests for AsyncBatcher (api/batcher.py) performance-sensitive behavior.

Focus:
 - the batching loop no longer busy-waits (polling get_nowait()+sleep(0));
   instead it should block efficiently via queue.get() with a timeout.
 - batching semantics (max_batch_size / max_latency_ms) are preserved.

Run: pytest tests/test_api_batcher.py -q
"""
import asyncio
import time

from api.batcher import AsyncBatcher


def _process_batch(items):
    return [item * 2 for item in items]


def test_batches_multiple_items_submitted_concurrently():
    async def scenario():
        loop = asyncio.get_event_loop()
        batcher = AsyncBatcher(process_batch=_process_batch, max_batch_size=4, max_latency_ms=50, loop=loop)
        try:
            results = await asyncio.gather(*(batcher.submit(i) for i in range(4)))
            assert sorted(results) == [0, 2, 4, 6]
        finally:
            await batcher.stop()

    asyncio.run(scenario())


def test_single_item_returns_within_latency_budget():
    async def scenario():
        loop = asyncio.get_event_loop()
        max_latency_ms = 30
        batcher = AsyncBatcher(process_batch=_process_batch, max_batch_size=8, max_latency_ms=max_latency_ms, loop=loop)
        try:
            start = time.time()
            result = await batcher.submit(5)
            elapsed_ms = (time.time() - start) * 1000.0
            assert result == 10
            # Should complete close to the latency budget, not hang.
            assert elapsed_ms < max_latency_ms + 500
        finally:
            await batcher.stop()

    asyncio.run(scenario())


def test_batcher_loop_does_not_busy_wait(monkeypatch):
    """
    Regression test for the CPU-burning busy-wait that previously polled
    `get_nowait()` in a loop with `await asyncio.sleep(0)`. That pattern
    would call asyncio.sleep(0) a very large number of times while waiting
    (idle) for the batch window to elapse. The fixed implementation should
    instead await a single `queue.get()` with a timeout, so sleep(0) (a
    zero-delay yield) should never be invoked by the batcher loop.
    """
    calls = {"sleep0": 0}
    orig_sleep = asyncio.sleep

    async def counting_sleep(delay, *args, **kwargs):
        if delay == 0:
            calls["sleep0"] += 1
        return await orig_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", counting_sleep)

    async def scenario():
        loop = asyncio.get_event_loop()
        batcher = AsyncBatcher(process_batch=_process_batch, max_batch_size=8, max_latency_ms=40, loop=loop)
        try:
            # Only one item is submitted, so the batcher has to wait out most
            # of the latency window with nothing else in the queue.
            result = await batcher.submit(3)
            assert result == 6
        finally:
            await batcher.stop()

    asyncio.run(scenario())
    assert calls["sleep0"] == 0
