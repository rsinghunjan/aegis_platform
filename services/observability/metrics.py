"""Metrics export with a Prometheus-compatible fallback.

Uses ``prometheus_client`` when installed; otherwise falls back to an
in-memory registry that supports the same counter/gauge/histogram
operations used by callers, and can render a Prometheus text-format
compatible payload via :meth:`MetricsRegistry.render`.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

_LabelKey = Tuple[Tuple[str, str], ...]


def _normalize_labels(labels: Optional[Dict[str, str]]) -> _LabelKey:
    return tuple(sorted((labels or {}).items()))


class _Counter:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[_LabelKey, float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        key = _normalize_labels(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, labels: Optional[Dict[str, str]] = None) -> float:
        return self._values.get(_normalize_labels(labels), 0.0)


class _Gauge:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[_LabelKey, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = _normalize_labels(labels)
        with self._lock:
            self._values[key] = value

    def value(self, labels: Optional[Dict[str, str]] = None) -> float:
        return self._values.get(_normalize_labels(labels), 0.0)


class _Histogram:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._observations: Dict[_LabelKey, List[float]] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = _normalize_labels(labels)
        with self._lock:
            self._observations.setdefault(key, []).append(value)

    def observations(self, labels: Optional[Dict[str, str]] = None) -> List[float]:
        return list(self._observations.get(_normalize_labels(labels), []))


class MetricsRegistry:
    """In-memory metrics registry with Prometheus-like semantics."""

    def __init__(self) -> None:
        self._counters: Dict[str, _Counter] = {}
        self._gauges: Dict[str, _Gauge] = {}
        self._histograms: Dict[str, _Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> _Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = _Counter(name, description)
            return self._counters[name]

    def gauge(self, name: str, description: str = "") -> _Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = _Gauge(name, description)
            return self._gauges[name]

    def histogram(self, name: str, description: str = "") -> _Histogram:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = _Histogram(name, description)
            return self._histograms[name]

    def render(self) -> str:
        """Render all metrics in a simplified Prometheus text format."""
        lines: List[str] = []
        for counter in self._counters.values():
            lines.append(f"# HELP {counter.name} {counter.description}")
            lines.append(f"# TYPE {counter.name} counter")
            for key, value in counter._values.items():
                lines.append(f"{counter.name}{_format_labels(key)} {value}")
        for gauge in self._gauges.values():
            lines.append(f"# HELP {gauge.name} {gauge.description}")
            lines.append(f"# TYPE {gauge.name} gauge")
            for key, value in gauge._values.items():
                lines.append(f"{gauge.name}{_format_labels(key)} {value}")
        for hist in self._histograms.values():
            lines.append(f"# HELP {hist.name} {hist.description}")
            lines.append(f"# TYPE {hist.name} histogram")
            for key, values in hist._observations.items():
                count = len(values)
                total = sum(values)
                lines.append(f"{hist.name}_count{_format_labels(key)} {count}")
                lines.append(f"{hist.name}_sum{_format_labels(key)} {total}")
        return "\n".join(lines) + "\n"


def _format_labels(key: _LabelKey) -> str:
    if not key:
        return ""
    body = ",".join(f'{k}="{v}"' for k, v in key)
    return "{" + body + "}"


_registry_lock = threading.Lock()
_registry_instance: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    global _registry_instance
    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = MetricsRegistry()
        return _registry_instance
