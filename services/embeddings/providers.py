"""Embedding provider abstraction."""
from __future__ import annotations

import abc
import hashlib
import math
import os
from typing import List, Optional


class EmbeddingProvider(abc.ABC):
    name: str = "provider"
    dimensions: int = 256

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def is_available(self) -> bool:
        return True


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding provider.

    Produces a fixed-size pseudo-embedding derived from token hashes. Not
    semantically meaningful in the way a trained model's embeddings are,
    but useful for offline tests, local development, and as a safe
    fallback when no embedding API is configured.
    """

    name = "local-hash"

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available():
            raise RuntimeError("openai embedding provider unavailable")
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class CohereEmbeddingProvider(EmbeddingProvider):
    name = "cohere"

    def __init__(self, api_key: Optional[str] = None, model: str = "embed-english-v3.0"):
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.model = model

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import cohere  # noqa: F401
        except ImportError:
            return False
        return True

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available():
            raise RuntimeError("cohere embedding provider unavailable")
        import cohere

        client = cohere.Client(self.api_key)
        response = client.embed(texts=texts, model=self.model, input_type="search_document")
        return list(response.embeddings)
