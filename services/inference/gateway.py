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


def _openai_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key=os.getenv("AEGIS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("AEGIS_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
    )


def _provider_factory(name: str):
    factories = {
        "openai": _openai_provider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "vllm": VLLMProvider,
        "echo": EchoProvider,
    }
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
        return [_provider_factory(name)() for name in names]

    provider = _openai_provider()
    return [provider] if provider.is_available() else [EchoProvider()]


def build_inference_router() -> ModelRouter:
    """Construct the shared provider router from deployment configuration.

    Set ``AEGIS_INFERENCE_PROVIDERS`` to an ordered, comma-separated list of
    ``openai``, ``anthropic``, ``ollama``, ``vllm``, and optionally ``echo``.
    Providers are attempted in that order; an explicit list does not silently
    add the echo provider.
    """
    return ModelRouter(configured_providers())
