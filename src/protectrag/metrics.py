"""Lightweight metrics counters (export to Prometheus, StatsD, or Phoenix-style dashboards)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol


class MetricsSink(Protocol):
    """Pluggable backend (Prometheus client, OpenTelemetry Meter, custom HTTP)."""

    def increment(self, name: str, value: int = 1, **labels: str) -> None: ...

    def observe(self, name: str, value: float, **labels: str) -> None: ...


@dataclass
class InMemoryMetrics:
    """
    Thread-safe in-process counters and histograms for tests and small services.

    Call :meth:`snapshot` to scrape or log periodically.
    """

    _counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = field(default_factory=dict)
    _hist: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _lock: Lock = field(default_factory=Lock)

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._hist[key].append(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": {f"{k[0]}{dict(k[1])}": v for k, v in self._counters.items()},
                "histograms": {
                    f"{k[0]}{dict(k[1])}": vals[:] for k, vals in self._hist.items()
                },
            }
