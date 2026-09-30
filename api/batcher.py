"""
Simple async batching helper used by the model registry.

AsyncBatcher accepts arbitrary items via submit(), groups them into batches
bounded by max_batch_size / max_latency_ms, and invokes process_batch(items)
(a blocking callable) in a thread-pool executor, then distributes the
per-item results back to each caller's awaiting future.
"""
import asyncio
import logging
import time
from typing import Any, List

logger = logging.getLogger("aegis.batcher")


class AsyncBatcher:
    def __init__(self, process_batch, max_batch_size: int = 8, max_latency_ms: int = 50, loop=None):
        self.process_batch = process_batch
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.loop = loop or asyncio.get_event_loop()
        self._queue: "asyncio.Queue" = asyncio.Queue()
        self._stopped = False
        self._task = self.loop.create_task(self._batcher_loop())

    async def submit(self, item: Any):
        fut = self.loop.create_future()
        await self._queue.put((item, fut))
        return await fut

    async def _batcher_loop(self):
        while not self._stopped:
            try:
                first = await self._queue.get()
                items = [first[0]]
                futures = [first[1]]
                start = time.time()
                # Drain within max_latency_ms or until max_batch_size, without busy-waiting.
                # Instead of polling get_nowait()+sleep(0) (which burns CPU spinning the
                # event loop), block on queue.get() with a timeout for the remaining
                # latency budget so the loop is idle (not spinning) between arrivals.
                while len(items) < self.max_batch_size:
                    elapsed_ms = (time.time() - start) * 1000.0
                    remaining_s = (self.max_latency_ms - elapsed_ms) / 1000.0
                    if remaining_s <= 0:
                        break
                    try:
                        item, fut = await asyncio.wait_for(self._queue.get(), timeout=remaining_s)
                    except asyncio.TimeoutError:
                        break
                    items.append(item)
                    futures.append(fut)
                # Now process batch (call blocking function in threadpool)
                results = await self.loop.run_in_executor(None, self._safe_process, items)
                # results must be list-like with len == len(items)
                if not isinstance(results, (list, tuple)) or len(results) != len(items):
                    # best-effort: broadcast single result
                    for fut in futures:
                        if not fut.done():
                            fut.set_result(results)
                else:
                    for fut, res in zip(futures, results):
                        if not fut.done():
                            fut.set_result(res)
            except Exception:
                logger.exception("batcher loop error")
                await asyncio.sleep(0.1)

    def _safe_process(self, items: List[Any]):
        try:
            return self.process_batch(items)
        except Exception as exc:
            logger.exception("process_batch failed: %s", exc)
            # return list of errors
            return [{"error": str(exc)} for _ in items]

    async def stop(self):
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
