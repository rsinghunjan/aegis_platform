"""Provider abstraction for the inference layer.

Each provider implements a minimal synchronous ``generate`` method and an
optional ``stream`` generator. Real network/SDK calls are performed lazily
(imported inside the method) so importing this module never requires the
optional third-party SDKs to be installed.
"""
from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class InferenceRequest:
    prompt: str
    model: str = "default"
    max_tokens: int = 256
    temperature: float = 0.0
    stop: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str = "stop"
    raw: Optional[Dict[str, Any]] = None


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a request."""


class InferenceProvider(abc.ABC):
    """Abstract base class for a model-serving backend."""

    name: str = "provider"

    @abc.abstractmethod
    def generate(self, request: InferenceRequest) -> InferenceResult:
        raise NotImplementedError

    def stream(self, request: InferenceRequest) -> Iterator[str]:
        """Default streaming implementation: yield the full result once.

        Providers that support true token streaming should override this.
        """
        result = self.generate(request)
        yield result.text

    def is_available(self) -> bool:
        """Whether this provider is usable in the current environment."""
        return True


class EchoProvider(InferenceProvider):
    """Deterministic local provider with no external dependencies.

    Useful as a safe default/fallback and for tests. Echoes a transformed
    version of the prompt and performs naive whitespace tokenization.
    """

    name = "local-echo"

    def generate(self, request: InferenceRequest) -> InferenceResult:
        start = time.monotonic()
        text = request.prompt.strip()
        if request.max_tokens:
            tokens = text.split()
            text = " ".join(tokens[: request.max_tokens])
        latency_ms = (time.monotonic() - start) * 1000
        return InferenceResult(
            text=text,
            model=request.model,
            provider=self.name,
            input_tokens=len(request.prompt.split()),
            output_tokens=len(text.split()),
            latency_ms=latency_ms,
        )

    def stream(self, request: InferenceRequest) -> Iterator[str]:
        for word in request.prompt.strip().split():
            yield word + " "


class OpenAICompatibleProvider(InferenceProvider):
    """Provider for OpenAI and OpenAI-API-compatible endpoints.

    Requires the ``openai`` package and an API key (via constructor or the
    ``OPENAI_API_KEY`` environment variable) to actually call out to the
    network; otherwise :meth:`is_available` returns ``False`` so callers can
    fall back to another provider.
    """

    name = "openai"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def generate(self, request: InferenceRequest) -> InferenceResult:
        if not self.is_available():
            raise ProviderError("openai provider unavailable (missing SDK or API key)")
        import openai

        start = time.monotonic()
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
        )
        latency_ms = (time.monotonic() - start) * 1000
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return InferenceResult(
            text=choice.message.content or "",
            model=request.model,
            provider=self.name,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason or "stop",
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )


class AnthropicProvider(InferenceProvider):
    """Provider for the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def generate(self, request: InferenceRequest) -> InferenceResult:
        if not self.is_available():
            raise ProviderError("anthropic provider unavailable (missing SDK or API key)")
        import anthropic

        start = time.monotonic()
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=[{"role": "user", "content": request.prompt}],
        )
        latency_ms = (time.monotonic() - start) * 1000
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        usage = getattr(response, "usage", None)
        return InferenceResult(
            text=text,
            model=request.model,
            provider=self.name,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            latency_ms=latency_ms,
            finish_reason=getattr(response, "stop_reason", "stop") or "stop",
        )


class OllamaProvider(InferenceProvider):
    """Provider for a local Ollama server."""

    name = "ollama"

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    def is_available(self) -> bool:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    def generate(self, request: InferenceRequest) -> InferenceResult:
        if not self.is_available():
            raise ProviderError("ollama provider unavailable (missing httpx)")
        import httpx

        start = time.monotonic()
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": request.model, "prompt": request.prompt, "stream": False},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.monotonic() - start) * 1000
        text = data.get("response", "")
        return InferenceResult(
            text=text,
            model=request.model,
            provider=self.name,
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            latency_ms=latency_ms,
            raw=data,
        )


class VLLMProvider(InferenceProvider):
    """Provider for a self-hosted vLLM OpenAI-compatible server."""

    name = "vllm"

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000")

    def is_available(self) -> bool:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    def generate(self, request: InferenceRequest) -> InferenceResult:
        if not self.is_available():
            raise ProviderError("vllm provider unavailable (missing httpx)")
        import httpx

        start = time.monotonic()
        resp = httpx.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": request.model,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.monotonic() - start) * 1000
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return InferenceResult(
            text=choice.get("text", ""),
            model=request.model,
            provider=self.name,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason", "stop") or "stop",
            raw=data,
        )
