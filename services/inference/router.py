"""Routing with fallback chains across inference providers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

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


class ModelRouter:
    """Routes an :class:`InferenceRequest` through an ordered fallback chain.

    Providers are tried in order; the first available provider that
    successfully generates a result wins. All failures are recorded so
    callers can inspect why a fallback occurred (useful for observability).
    """

    def __init__(self, providers: List[InferenceProvider]):
        if not providers:
            raise ValueError("ModelRouter requires at least one provider")
        self.providers = providers
        self.last_attempts: List[RouteAttempt] = []

    def generate(self, request: InferenceRequest) -> InferenceResult:
        attempts: List[RouteAttempt] = []
        for provider in self.providers:
            if not provider.is_available():
                attempts.append(RouteAttempt(provider.name, False, "unavailable"))
                continue
            try:
                result = provider.generate(request)
                attempts.append(RouteAttempt(provider.name, True))
                self.last_attempts = attempts
                return result
            except ProviderError as exc:
                attempts.append(RouteAttempt(provider.name, False, str(exc)))
                logger.warning("provider %s failed: %s", provider.name, exc)
            except Exception as exc:  # pragma: no cover - defensive catch-all
                attempts.append(RouteAttempt(provider.name, False, repr(exc)))
                logger.exception("provider %s raised unexpected error", provider.name)
        self.last_attempts = attempts
        raise NoProviderAvailableError(
            f"all providers failed: {[(a.provider, a.error) for a in attempts]}",
            attempts,
        )
