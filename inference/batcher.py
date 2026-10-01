"""
Simple async batching worker for inference serving.

BatchWorker collects individual prediction requests (submitted via enqueue())
into batches bounded by max_batch_size / max_latency_s, then calls predict_fn
once per batch and fans results back out to each caller's future.
"""
import asyncio
import logging
import time
from typing import Any, List, Tuple

logger = logging.getLogger("aegis.inference.batcher")


class BatchWorker:
    def __init__(self, predict_fn, max_batch_size: int = 8, max_latency_s: float = 0.05):
        self.predict_fn = predict_fn
        self.max_batch_size = max_batch_size
        self.max_latency_s = max_latency_s
        self._queue: "asyncio.Queue[Tuple[str, asyncio.Future]]" = asyncio.Queue()
        self._stop = False
        self._task = None

    async def enqueue(self, text: str) -> Any:
        """Submit one item for batched prediction and await its result."""
        fut = asyncio.get_event_loop().create_future()
        await self._queue.put((text, fut))
        return await fut
    async def _drain_batch(self) -> List[Tuple[str, asyncio.Future]]:
        """
        Wait for at least one item, then collect up to max_batch_size items within max_latency_s.
        """
        items = []
        # block until at least one item is available
        item = await self._queue.get()
        items.append(item)
        start = time.time()
        # keep collecting until max_batch_size or timeout
        while len(items) < self.max_batch_size:
            elapsed = time.time() - start
            remaining = self.max_latency_s - elapsed
            if remaining <= 0:
                break
            try:
                # short wait for next item
                nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                items.append(nxt)
            except asyncio.TimeoutError:
                break
        return items

    async def _worker_loop(self):
        logger.info("BatchWorker started (max_batch_size=%d, max_latency_s=%.3f)", self.max_batch_size, self.max_latency_s)
        while not self._stop:
            try:
                batch = await self._drain_batch()
                texts = [t for (t, f) in batch]
                # Call model (sync or async). If predict_fn is coroutine, await it.
                t0 = time.time()
                try:
                    res = self.predict_fn(texts)
                    if asyncio.iscoroutine(res):
                        res = await res
                except Exception as e:
                    logger.exception("Batch predict failed: %s", e)
                    # set exception on futures
                    for _, fut in batch:
                        if not fut.done():
                            fut.set_exception(e)
                    continue
                # Expect res to be list-like with same length
                if not isinstance(res, (list, tuple)) or len(res) != len(batch):
                    err = RuntimeError("Batch predict returned invalid result length")
                    for _, fut in batch:
                        if not fut.done():
                            fut.set_exception(err)
                    continue

                # set results
                for (_, fut), out in zip(batch, res):
                    if not fut.done():
                        fut.set_result(out)
                latency = time.time() - t0
                logger.debug("Batch of %d processed in %.4f s", len(batch), latency)
            except Exception:
                logger.exception("BatchWorker loop error")
                await asyncio.sleep(0.1)

    def start(self):
        if self._task is None or self._task.done():
            self._stop = False
            self._task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        self._stop = True
        # allow running task to exit gracefully
        if self._task:
            await self._task
