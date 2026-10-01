"""Model registry with versioning for the inference layer."""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable description of a registered model version.

    ``source`` identifies where the model artifact/weights come from, e.g.
    ``huggingface``, ``onnx``, ``custom``, or a provider name for hosted
    models (``openai``, ``anthropic``, ``ollama``, ``vllm``).
    """

    name: str
    version: str
    source: str
    provider: str
    context_window: int = 4096
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    tags: tuple = field(default_factory=tuple)
    registered_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def fingerprint(self) -> str:
        """Stable content hash for cache keys / audit trails."""
        payload = f"{self.name}|{self.version}|{self.source}|{self.provider}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]


class ModelRegistry:
    """Thread-safe in-memory registry of model versions.

    Supports registering multiple versions of the same model name and
    resolving either a specific version or the most-recently-registered
    ("latest") version.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: Dict[str, Dict[str, ModelMetadata]] = {}
        self._latest: Dict[str, str] = {}

    def register(self, metadata: ModelMetadata) -> ModelMetadata:
        with self._lock:
            versions = self._models.setdefault(metadata.name, {})
            versions[metadata.version] = metadata
            self._latest[metadata.name] = metadata.version
            return metadata

    def get(self, name: str, version: Optional[str] = None) -> ModelMetadata:
        with self._lock:
            versions = self._models.get(name)
            if not versions:
                raise KeyError(f"model '{name}' is not registered")
            resolved_version = version or self._latest.get(name)
            if resolved_version not in versions:
                raise KeyError(f"model '{name}' has no version '{version}'")
            return versions[resolved_version]

    def list_versions(self, name: str) -> List[ModelMetadata]:
        with self._lock:
            return sorted(
                self._models.get(name, {}).values(), key=lambda m: m.registered_at
            )

    def list_models(self) -> List[str]:
        with self._lock:
            return sorted(self._models.keys())

    def deregister(self, name: str, version: str) -> None:
        with self._lock:
            versions = self._models.get(name)
            if not versions or version not in versions:
                raise KeyError(f"model '{name}' version '{version}' not found")
            del versions[version]
            if not versions:
                del self._models[name]
                self._latest.pop(name, None)
            elif self._latest.get(name) == version:
                self._latest[name] = max(versions.values(), key=lambda m: m.registered_at).version
