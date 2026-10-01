"""Inference layer: model registry, provider abstraction, routing, batching.

This package provides a production-grade-but-dependency-light inference
stack for Aegis. All external SDKs (openai, anthropic, httpx, etc.) are
imported lazily so the module tree can be imported in minimal environments
(e.g. CI, unit tests) without those packages installed.
"""

from .registry import ModelMetadata, ModelRegistry
from .providers import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    EchoProvider,
    OpenAICompatibleProvider,
    AnthropicProvider,
    OllamaProvider,
    VLLMProvider,
)
from .router import ModelRouter, NoProviderAvailableError
from .tokens import count_tokens, estimate_cost
from .batch import BatchInferenceQueue

__all__ = [
    "ModelMetadata",
    "ModelRegistry",
    "InferenceProvider",
    "InferenceRequest",
    "InferenceResult",
    "EchoProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "VLLMProvider",
    "ModelRouter",
    "NoProviderAvailableError",
    "count_tokens",
    "estimate_cost",
    "BatchInferenceQueue",
]
