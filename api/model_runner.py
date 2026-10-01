"""
ModelRegistry replacement with batching, concurrency control, warmup, and basic OOM protection.

Provides:
 - ModelRegistry class with load_model / unload_model / list_models
 - Per-model AsyncBatcher and concurrency Semaphore
 - predict_sync(model, version, request_payload) -> blocking call
 - predict_async(...) -> awaitable

Tuning knobs per model:
 - max_concurrency: number of parallel in-flight requests
 - max_batch_size, batch_latency_ms
 - warmup sample & iters

This implementation is deliberately opinionated but small; extend it to:
 - integrate with Prometheus metrics and OpenTelemetry spans
 - add model eviction / LRU caching of loaded models
 - advanced OOM handling and preflight memory checks
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional

from .model_loader import TorchModelWrapper, ONNXModelWrapper, get_preferred_device, BaseModelWrapper
from .batcher import AsyncBatcher

logger = logging.getLogger("aegis.model_runner")


class ModelConfig:
    def __init__(self, model_path: str, runtime: str = "torch", device: Optional[str] = None,
                 max_concurrency: int = 4, max_batch_size: int = 8, batch_latency_ms: int = 50,
                 warmup_sample: Optional[dict] = None, warmup_iters: int = 1):
        self.model_path = model_path
        self.runtime = runtime
        self.device = device
        self.max_concurrency = max_concurrency
        self.max_batch_size = max_batch_size
        self.batch_latency_ms = batch_latency_ms
        self.warmup_sample = warmup_sample
        self.warmup_iters = warmup_iters


class ModelRegistry:
    def __init__(self):
        # key = (model_name, version)
        self._configs: Dict[str, ModelConfig] = {}
        self._models: Dict[str, BaseModelWrapper] = {}
        self._batchers: Dict[str, AsyncBatcher] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        # Per-model locks guarding on-demand load/warmup so concurrent requests
        # for the same not-yet-loaded model coalesce into a single load instead
        # of racing (which could otherwise trigger duplicate loads/warmups and
        # waste CPU/GPU/memory under concurrency).
        self._load_locks: Dict[str, asyncio.Lock] = {}

    def _key(self, model_name: str, version: str) -> str:
        return f"{model_name}:{version}"

    def _get_load_lock(self, k: str) -> asyncio.Lock:
        lock = self._load_locks.get(k)
        if lock is None:
            lock = asyncio.Lock()
            self._load_locks[k] = lock
        return lock

    def register(self, model_name: str, version: str, config: ModelConfig):
        k = self._key(model_name, version)
        self._configs[k] = config

    def _build_and_warmup_wrapper(self, model_name: str, version: str) -> BaseModelWrapper:
        """
        Blocking: construct the wrapper and perform model I/O / warmup.

        Contains no asyncio-affine calls (no task/future/semaphore creation),
        so it is safe to run off the event-loop thread (e.g. via
        run_in_executor) from `load_async`.
        """
        k = self._key(model_name, version)
        cfg = self._configs.get(k)
        if not cfg:
            raise KeyError("model config not registered")
        if cfg.runtime.lower().startswith("torch"):
            wrapper = TorchModelWrapper(cfg.model_path, model_name, version, device=cfg.device or get_preferred_device())
        elif cfg.runtime.lower().startswith("onnx"):
            wrapper = ONNXModelWrapper(cfg.model_path, model_name, version, use_gpu=(cfg.device == "cuda"))
        else:
            raise ValueError("unsupported runtime")
        try:
            wrapper.load()
        except Exception:
            # torch wrapper loads lazily in warmup too; ignore here
            logger.exception("model load failed for %s", k)
        if cfg.warmup_sample:
            try:
                wrapper.warmup(cfg.warmup_sample, iters=cfg.warmup_iters)
            except Exception:
                logger.exception("warmup failed for %s", k)
        return wrapper

    def _register_loaded_wrapper(self, model_name: str, version: str, wrapper: BaseModelWrapper):
        """
        Creates the batcher/semaphore for a freshly-loaded wrapper and
        publishes it in the registry. Must run on the event-loop thread that
        will drive the batcher's task (asyncio primitives aren't safe to
        create from an arbitrary worker thread).
        """
        k = self._key(model_name, version)
        cfg = self._configs[k]
        sem = asyncio.Semaphore(cfg.max_concurrency)
        batcher = AsyncBatcher(process_batch=wrapper.predict_batch, max_batch_size=cfg.max_batch_size, max_latency_ms=cfg.batch_latency_ms)
        self._models[k] = wrapper
        self._batchers[k] = batcher
        self._semaphores[k] = sem
        logger.info("Model %s loaded with concurrency=%d batch_size=%d", k, cfg.max_concurrency, cfg.max_batch_size)
        return wrapper

    def load(self, model_name: str, version: str):
        k = self._key(model_name, version)
        if k in self._models:
            return self._models[k]
        wrapper = self._build_and_warmup_wrapper(model_name, version)
        return self._register_loaded_wrapper(model_name, version, wrapper)

    async def load_async(self, model_name: str, version: str):
        """
        Async-safe on-demand load/warmup.

        `load()` runs potentially slow, blocking work (model I/O, warmup
        inference). Calling it directly from an async handler would stall the
        event loop for the whole duration. This wrapper:
          - runs the blocking model I/O/warmup in the default thread-pool
            executor so the event loop keeps serving other requests, then
            finishes registration (batcher/semaphore creation) back on the
            calling event-loop thread, and
          - serializes concurrent on-demand loads of the *same* model/version
            behind a per-key lock, so a burst of requests for a cold model
            triggers exactly one load instead of N redundant (and
            resource-contending) loads.
        """
        k = self._key(model_name, version)
        if k in self._models:
            return self._models[k]
        lock = self._get_load_lock(k)
        async with lock:
            # Re-check: another waiter may have completed the load while we
            # were blocked on the lock.
            if k in self._models:
                return self._models[k]
            loop = asyncio.get_running_loop()
            wrapper = await loop.run_in_executor(None, self._build_and_warmup_wrapper, model_name, version)
            return self._register_loaded_wrapper(model_name, version, wrapper)

    async def predict_async(self, model_name: str, version: str, input_payload: Any, timeout_s: float = 30.0):
        k = self._key(model_name, version)
        if k not in self._models:
            # try to load on demand (off the event loop, de-duplicated across
            # concurrent callers for the same model/version)
            try:
                await self.load_async(model_name, version)
            except Exception:
                raise KeyError("model not found")
        sem = self._semaphores[k]
        batcher = self._batchers[k]

        # Acquire concurrency permit (async)
        try:
            await asyncio.wait_for(sem.acquire(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise TimeoutError("concurrency limit acquire timeout")
        try:
            # Submit to batcher; batcher returns result sync per item
            fut = await batcher.submit(input_payload)
            return fut
        finally:
            try:
                sem.release()
            except Exception:
                logger.exception("semaphore release failed for %s", k)

    def predict_sync(self, model_name: str, version: str, input_payload: Any, timeout_s: float = 30.0):
        """
        Blocking wrapper for sync callers (e.g., FastAPI handlers).
        Runs the asyncio predict_async in the registry loop with timeout.
        """
        coro = self.predict_async(model_name, version, input_payload, timeout_s=timeout_s)
        return asyncio.get_event_loop().run_until_complete(coro)

    def unload(self, model_name: str, version: str):
        k = self._key(model_name, version)
        if k in self._batchers:
            # stop batcher gracefully
            try:
                coro = self._batchers[k].stop()
                asyncio.get_event_loop().run_until_complete(coro)
            except Exception:
                logger.exception("failed to stop batcher for %s", k)
            del self._batchers[k]
        if k in self._models:
            try:
                self._models[k].cpu_offload()
            except Exception:
                pass
            del self._models[k]
        if k in self._semaphores:
            del self._semaphores[k]
        if k in self._configs:
            del self._configs[k]
        if k in self._load_locks:
            del self._load_locks[k]
        logger.info("Unloaded model %s", k)

    def list_models(self):
        return list(self._configs.keys())


# Global registry instance for easy import
registry = ModelRegistry()
