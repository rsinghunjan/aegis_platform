"""Deterministic, tamper-evident metadata for registered runtime capabilities."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional


class CapabilityCatalog:
    """Catalog tool metadata without storing executable handlers or secrets."""

    def __init__(self, signer: Optional[Callable[[bytes], str]] = None):
        self._entries: dict[str, dict[str, Any]] = {}
        self._signer = signer

    def register(self, spec: Any) -> str:
        metadata = spec.dict()
        self._entries[spec.name] = metadata
        return self.version

    @property
    def version(self) -> str:
        return self.sha256[:16]

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self._entries, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def signature(self) -> Optional[str]:
        if self._signer is None:
            return None
        return self._signer(self.sha256.encode("ascii"))

    def list(self) -> list[dict[str, Any]]:
        return [self.inspect(name) for name in sorted(self._entries)]

    def inspect(self, name: str) -> dict[str, Any]:
        if name not in self._entries:
            raise KeyError(name)
        return dict(self._entries[name])

    def contains_version(self, name: str, version: str) -> bool:
        entry = self._entries.get(name)
        return bool(entry and entry.get("version") == version)

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sha256": self.sha256,
            "signature": self.signature,
            "signed": self._signer is not None,
            "capability_count": len(self._entries),
        }
