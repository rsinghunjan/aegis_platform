  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
Glue to wire adaptive batching and model evictor into the existing registry.

Responsibilities:
 - provide touch_model_usage(model_name, version) helper used by wrappers after handling a request
 - wrap a batched predict call to record latency, update adaptive batcher and touch last_used
 - start background ModelEvictor and keep reference for diagnostics

Integration steps:
 - import and call bootstrap_model_runtime_hardening(registry, loop) from your app startup (api_server)
 - after each successful model prediction call record_predict_result(model_key, latency_ms, batch_size)
 - ensure registry._models is a dict and registry.unload(name, version) exists
"""
import time
import logging
from typing import Tuple

from api.adaptive_batcher import AdaptiveBatcher
from api.model_eviction import ModelEvictor

logger = logging.getLogger("aegis.model_runtime_hardening")

# simple per-model state maps
_ADAPTIVE_BATCHERS = {}
_EVICTOR = None
_REGISTRY = None


def bootstrap_model_runtime_hardening(registry, loop=None):
    """
    Start the model evictor and attach registry reference. Call at app startup.
    """
    global _EVICTOR, _REGISTRY
    _REGISTRY = registry
    _EVICTOR = ModelEvictor(registry, loop=loop)
    _EVICTOR.start()
    logger.info("Model runtime hardening bootstrap complete")


def touch_model_usage(model_name: str, version: str):
    """
    Mark model as recently used for eviction LRU bookkeeping.
    """
    key = f"{model_name}:{version}"
    wrapper = getattr(_REGISTRY, "_models", {}).get(key)
    if not wrapper:
        return
    try:
        setattr(wrapper, "_last_used", time.time())
    except Exception:
        logger.exception("failed to touch model usage for %s", key)


def get_adaptive_batcher(model_name: str, version: str, **kwargs) -> AdaptiveBatcher:
    """
    Return or create an AdaptiveBatcher for a model. Configure initial values via kwargs.
    """
    key = f"{model_name}:{version}"
    ab = _ADAPTIVE_BATCHERS.get(key)
    if not ab:
        ab = AdaptiveBatcher(key,
                             initial_max_batch=kwargs.get("initial_max_batch", 8),
                             min_batch=kwargs.get("min_batch", 1),
                             max_batch=kwargs.get("max_batch", 64),
                             target_p95_ms=kwargs.get("target_p95_ms", 500))
        _ADAPTIVE_BATCHERS[key] = ab
    return ab


def record_predict_result(model_name: str, version: str, latency_ms: float, batch_size: int):
    """
    Record a predict result (latency & batch_size). This will update adaptive batcher state
    and touch model usage for eviction.
    Call this after successful (or failed) batched predict.
    """
    try:
        ab = get_adaptive_batcher(model_name, version)
        ab.record_latency(latency_ms, batch_size)
    except Exception:
        logger.exception("adaptive batcher record failed")
    try:
        touch_model_usage(model_name, version)
    except Exception:
        logger.exception("touch_model_usage failed")

