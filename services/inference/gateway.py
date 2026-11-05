"""Environment-configured inference gateway for supported model providers."""
from __future__ import annotations

import os

from .providers import (
    AnthropicProvider,
    EchoProvider,
    InferenceProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    VLLMProvider,
)
from .router import ModelRouter


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        raise ValueError(f"{name} must be a non-negative integer") from None


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        raise ValueError(f"{name} must be a positive number") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _openai_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key=os.getenv("AEGIS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("AEGIS_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
    )


def dev_providers_allowed() -> bool:
    """Whether non-model dev/test fallbacks (echo, local-hash) may be used."""
    return os.getenv("AEGIS_ALLOW_DEV_PROVIDERS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _provider_factory(name: str):
    factories = {
        "openai": _openai_provider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "vllm": VLLMProvider,
        "echo": EchoProvider,
    }
    if name == "echo" and not dev_providers_allowed():
        raise ValueError(
            "the echo provider is not an AI model and is only allowed when "
            "AEGIS_ALLOW_DEV_PROVIDERS=true (tests)"
        )
    try:
        return factories[name]
    except KeyError as exc:
        raise ValueError(f"unsupported inference provider: {name}") from exc


def configured_providers() -> list[InferenceProvider]:
    configured = os.getenv("AEGIS_INFERENCE_PROVIDERS")
    if configured:
        names = [name.strip().lower() for name in configured.split(",") if name.strip()]
        if not names:
            raise ValueError("AEGIS_INFERENCE_PROVIDERS must name at least one provider")
        providers = [_provider_factory(name)() for name in names]
    else:
        provider = _openai_provider()
        providers = [provider] if provider.is_available() else []
        if not providers and dev_providers_allowed():
            return [EchoProvider()]

    if not any(
        provider.is_available() and provider.name != EchoProvider.name
        for provider in providers
    ) and not dev_providers_allowed():
        raise RuntimeError(
            "no real model provider is configured; set AEGIS_LLM_API_KEY (and "
            "install the 'ai' extra) or AEGIS_INFERENCE_PROVIDERS with an "
            "available openai, anthropic, ollama, or vllm provider"
        )
    return providers


def build_inference_router() -> ModelRouter:
    """Construct the shared provider router from deployment configuration.

    Set ``AEGIS_INFERENCE_PROVIDERS`` to an ordered, comma-separated list of
    ``openai``, ``anthropic``, ``ollama``, and ``vllm``. Providers are attempted
    in that order. Startup fails unless at least one real model provider is
    available; ``echo`` is accepted only with ``AEGIS_ALLOW_DEV_PROVIDERS=true``.
    """
    return ModelRouter(
        configured_providers(),
        max_retries=_env_int("AEGIS_INFERENCE_MAX_RETRIES", 1),
        failure_threshold=_env_int("AEGIS_INFERENCE_BREAKER_FAILURES", 5),
        cooldown_seconds=_env_float("AEGIS_INFERENCE_BREAKER_COOLDOWN_SECONDS", 30.0),
    )
