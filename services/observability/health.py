"""Health checks and circuit breaker for resilient service calls."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


@dataclass
class HealthCheck:
    name: str
    check_fn: Callable[[], bool]
    critical: bool = True


@dataclass
class HealthResult:
    name: str
    healthy: bool
    critical: bool
    error: Optional[str] = None


class HealthRegistry:
    """Registers and runs named health checks (liveness/readiness probes)."""

    def __init__(self) -> None:
        self._checks: Dict[str, HealthCheck] = {}

    def register(self, check: HealthCheck) -> None:
        self._checks[check.name] = check

    def run_all(self) -> List[HealthResult]:
        results = []
        for check in self._checks.values():
            try:
                healthy = bool(check.check_fn())
                results.append(HealthResult(check.name, healthy, check.critical))
            except Exception as exc:  # pragma: no cover - defensive
                results.append(HealthResult(check.name, False, check.critical, error=str(exc)))
        return results

    def is_healthy(self) -> bool:
        return all(r.healthy for r in self.run_all() if r.critical)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    """Standard circuit breaker: closed -> open after N failures, then
    half-open after a cooldown to probe recovery.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN and self._opened_at is not None:
                if time.time() - self._opened_at >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def call(self, fn: Callable[[], "object"]):
        current_state = self.state
        if current_state == CircuitState.OPEN:
            raise CircuitOpenError("circuit breaker is open")
        try:
            result = fn()
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()

    def _record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._opened_at = None
