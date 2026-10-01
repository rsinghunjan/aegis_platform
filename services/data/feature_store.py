"""Feature store abstraction (Feast/Tecton-style online + offline features)."""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeatureVector:
    entity_id: str
    features: Dict[str, Any]
    event_timestamp: float = field(default_factory=time.time)


class FeatureStore(abc.ABC):
    name: str = "feature-store"

    @abc.abstractmethod
    def write(self, feature_view: str, vectors: List[FeatureVector]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def read_online(self, feature_view: str, entity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return the latest feature values for each requested entity id."""
        raise NotImplementedError

    @abc.abstractmethod
    def read_historical(
        self, feature_view: str, entity_ids: Optional[List[str]] = None
    ) -> List[FeatureVector]:
        """Return the full point-in-time history for offline training."""
        raise NotImplementedError


class InMemoryFeatureStore(FeatureStore):
    """Simple in-process feature store with online + offline reads.

    This mirrors the dual online/offline access pattern of Feast/Tecton
    without requiring Redis/Parquet infrastructure, making it suitable for
    tests and local development. A production deployment can swap this for
    a Feast- or Tecton-backed adapter implementing the same interface.
    """

    name = "in-memory"

    def __init__(self) -> None:
        self._store: Dict[str, List[FeatureVector]] = {}

    def write(self, feature_view: str, vectors: List[FeatureVector]) -> None:
        self._store.setdefault(feature_view, []).extend(vectors)

    def read_online(self, feature_view: str, entity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        history = self._store.get(feature_view, [])
        latest: Dict[str, FeatureVector] = {}
        for vector in history:
            if vector.entity_id not in entity_ids:
                continue
            current = latest.get(vector.entity_id)
            if current is None or vector.event_timestamp >= current.event_timestamp:
                latest[vector.entity_id] = vector
        return {entity_id: v.features for entity_id, v in latest.items()}

    def read_historical(
        self, feature_view: str, entity_ids: Optional[List[str]] = None
    ) -> List[FeatureVector]:
        history = self._store.get(feature_view, [])
        if entity_ids is None:
            return list(history)
        return [v for v in history if v.entity_id in entity_ids]
