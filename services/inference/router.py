"""Routing with fallback chains across inference providers."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .providers import InferenceProvider, InferenceRequest, InferenceResult, ProviderError

logger = logging.getLogger("aegis.inference.router")


class NoProviderAvailableError(RuntimeError):
    """Raised when every provider in the fallback chain fails or is unavailable."""

    def __init__(self, message: str, attempts: List[RouteAttempt]):
        super().__init__(message)
        self.attempts = tuple(attempts)


@dataclass
class RouteAttempt:
    provider: str
    succeeded: bool
    error: Optional[str] = None


class CircuitBreaker:
    """Opens after consecutive failures and allows one probe after a cooldown."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._probing = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.failure_threshold > 0

    def allow(self) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            if self._opened_at is None:
                return True
            if self._clock() - self._opened_at < self.cooldown_seconds:
                return False
            if self._probing:
                return False
            self._probing = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probing = False

    def record_failure(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._failures += 1
            if self._probing or self._failures >= self.failure_threshold:
                self._opened_at = self._clock()
            self._probing = False

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None


class ModelRouter:
    """Routes an :class:`InferenceRequest` through an ordered fallback chain.

    Providers are tried in order; the first available provider that
    successfully generates a result wins. All failures are recorded so
    callers can inspect why a fallback occurred (useful for observability).

    Each provider is retried up to ``max_retries`` times with exponential
    backoff, and guarded by a circuit breaker that skips it after
    ``failure_threshold`` consecutive failed requests until ``cooldown_seconds``
    elapse. Defaults preserve the previous behavior (no retries, no breaker).
    """

    def __init__(
        self,
        providers: List[InferenceProvider],
        max_retries: int = 0,
        backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 4.0,
        failure_threshold: int = 0,
        cooldown_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not providers:
            raise ValueError("ModelRouter requires at least one provider")
        self.providers = providers
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep
        self.breakers: Dict[str, CircuitBreaker] = {
            self._key(index, provider): CircuitBreaker(
                failure_threshold, cooldown_seconds, clock
            )
            for index, provider in enumerate(providers)
        }
        self.last_attempts: List[RouteAttempt] = []

    @staticmethod
    def _key(index: int, provider: InferenceProvider) -> str:
        return f"{index}:{provider.name}"

    def generate(self, request: InferenceRequest) -> InferenceResult:
        attempts: List[RouteAttempt] = []
        for index, provider in enumerate(self.providers):
            breaker = self.breakers[self._key(index, provider)]
            if not provider.is_available():
                attempts.append(RouteAttempt(provider.name, False, "unavailable"))
                continue
            if not breaker.allow():
                attempts.append(RouteAttempt(provider.name, False, "circuit_open"))
                continue
            result = self._call_with_retries(provider, request, attempts)
            if result is not None:
                breaker.record_success()
                self.last_attempts = attempts
                return result
            breaker.record_failure()
        self.last_attempts = attempts
        raise NoProviderAvailableError(
            f"all providers failed: {[(a.provider, a.error) for a in attempts]}",
            attempts,
        )

    def _call_with_retries(
        self,
        provider: InferenceProvider,
        request: InferenceRequest,
        attempts: List[RouteAttempt],
    ) -> Optional[InferenceResult]:
        for attempt in range(self.max_retries + 1):
            try:
                result = provider.generate(request)
                attempts.append(RouteAttempt(provider.name, True))
                return result
            except ProviderError as exc:
                attempts.append(RouteAttempt(provider.name, False, str(exc)))
                logger.warning("provider %s failed: %s", provider.name, exc)
            except Exception as exc:  # pragma: no cover - defensive catch-all
                attempts.append(RouteAttempt(provider.name, False, repr(exc)))
                logger.exception("provider %s raised unexpected error", provider.name)
            if attempt < self.max_retries:
                self._sleep(
                    min(self.max_backoff_seconds, self.backoff_seconds * (2**attempt))
                )
        return None
