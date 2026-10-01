"""
AdaptiveBatcher: dynamically adjusts per-model batch timeout and max_batch_size
Usage pattern:
 - create AdaptiveBatcher(name, initial_max_batch=8, min_batch=1, max_batch=64)
 - call record_latency(lat_ms, batch_size)
 - read current max_batch via get_max_batch()

This implementation is conservative and uses an EWMA of p95 latency approximations
(we don't compute exact p95 here to keep overhead small). It is intended to be
used by the model serving path: after each batched predict completes call
record_latency() with observed latency and batch_size.
"""
import time
import threading
import statistics
import logging

logger = logging.getLogger("aegis.adaptive_batcher")


class AdaptiveBatcher:
    def __init__(self, name: str, initial_max_batch: int = 8, min_batch: int = 1, max_batch: int = 64, target_p95_ms: int = 500):
        self.name = name
        self._max_batch = int(initial_max_batch)
        self.min_batch = int(min_batch)
        self.max_batch = int(max_batch)
        self.target_p95_ms = int(target_p95_ms)
        # sliding window of recent latencies (ms)
        self._latencies = []
        self._sizes = []
        self._lock = threading.Lock()
        # EWMA smoothing factor
        self.alpha = 0.2
        self._ewma_p95 = None

    def record_latency(self, lat_ms: float, batch_size: int):
        with self._lock:
            self._latencies.append(float(lat_ms))
            self._sizes.append(int(batch_size))
            # keep last 200 samples
            if len(self._latencies) > 200:
                self._latencies = self._latencies[-200:]
                self._sizes = self._sizes[-200:]
            # compute approximate p95 via percentile
            try:
                p95 = float(statistics.quantiles(self._latencies, n=100)[94]) if len(self._latencies) >= 10 else max(self._latencies)
            except Exception:
                p95 = max(self._latencies) if self._latencies else lat_ms
            if self._ewma_p95 is None:
                self._ewma_p95 = p95
            else:
                self._ewma_p95 = (self.alpha * p95) + (1 - self.alpha) * self._ewma_p95
            # adjust batch size conservatively
            self._adjust_batch()

    def _adjust_batch(self):
        if self._ewma_p95 is None:
            return
        # if p95 below target by margin, try increasing batch
        margin = 0.75  # 75% of target considered safe
        if self._ewma_p95 < (self.target_p95_ms * margin):
            # increase by 10% capped
            new_batch = min(self.max_batch, int(self._max_batch * 1.1) + 1)
            if new_batch != self._max_batch:
                logger.info("AdaptiveBatcher(%s): increasing max_batch %s -> %s (p95=%.1fms)", self.name, self._max_batch, new_batch, self._ewma_p95)
            self._max_batch = new_batch
        elif self._ewma_p95 > self.target_p95_ms:
            # decrease by 20%
            new_batch = max(self.min_batch, int(self._max_batch * 0.8))
            if new_batch != self._max_batch:
                logger.info("AdaptiveBatcher(%s): decreasing max_batch %s -> %s (p95=%.1fms)", self.name, self._max_batch, new_batch, self._ewma_p95)
            self._max_batch = new_batch

    def get_max_batch(self) -> int:
        with self._lock:
            return int(self._max_batch)
