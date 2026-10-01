"""Metadata catalog abstraction, inspired by DataHub."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetMetadata:
    urn: str
    name: str
    owner: str
    schema: Dict[str, str] = field(default_factory=dict)
    tags: tuple = field(default_factory=tuple)
    description: str = ""
    registered_at: float = field(default_factory=time.time)
    lineage: tuple = field(default_factory=tuple)  # upstream dataset URNs


class MetadataCatalog:
    """Thread-safe in-memory metadata catalog.

    Provides a minimal subset of DataHub-style capabilities (dataset
    registration, search by tag, and lineage lookup) without requiring a
    running DataHub instance. A production deployment can implement the
    same interface against a real DataHub GMS client.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._datasets: Dict[str, DatasetMetadata] = {}

    def register(self, metadata: DatasetMetadata) -> DatasetMetadata:
        with self._lock:
            self._datasets[metadata.urn] = metadata
            return metadata

    def get(self, urn: str) -> DatasetMetadata:
        with self._lock:
            if urn not in self._datasets:
                raise KeyError(urn)
            return self._datasets[urn]

    def search_by_tag(self, tag: str) -> List[DatasetMetadata]:
        with self._lock:
            return [d for d in self._datasets.values() if tag in d.tags]

    def upstream_lineage(self, urn: str) -> List[DatasetMetadata]:
        dataset = self.get(urn)
        with self._lock:
            return [self._datasets[u] for u in dataset.lineage if u in self._datasets]

    def list_all(self) -> List[DatasetMetadata]:
        with self._lock:
            return list(self._datasets.values())
