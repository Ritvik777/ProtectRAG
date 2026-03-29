"""Lightweight metrics counters (export to Prometheus, StatsD, or Phoenix-style dashboards)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol

_SHARD_COUNT = 32


class MetricsSink(Protocol):
    """Pluggable backend (Prometheus client, OpenTelemetry Meter, custom HTTP)."""

    def increment(self, name: str, value: int = 1, **labels: str) -> None: ...

    def observe(self, name: str, value: float, **labels: str) -> None: ...


@dataclass
class InMemoryMetrics:
    """
    Thread-safe in-process counters and histograms for tests and small services.

    Uses striped locks to reduce contention under many concurrent writers.

    Call :meth:`snapshot` to scrape or log periodically.
    """

    _counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = field(default_factory=dict)
    _hist: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _locks: tuple[Lock, ...] = field(
        default_factory=lambda: tuple(Lock() for _ in range(_SHARD_COUNT))
    )

    def _lock_for(self, key: tuple[str, tuple[tuple[str, str], ...]]) -> Lock:
        return self._locks[hash(key) % _SHARD_COUNT]

    def increment(self, name: str, value: int = 1, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        lock = self._lock_for(key)
        with lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        lock = self._lock_for(key)
        with lock:
            self._hist[key].append(value)

    def snapshot(self) -> dict[str, Any]:
        for lk in self._locks:
            lk.acquire()
        try:
            return {
                "counters": {f"{k[0]}{dict(k[1])}": v for k, v in self._counters.items()},
                "histograms": {
                    f"{k[0]}{dict(k[1])}": vals[:] for k, vals in self._hist.items()
                },
            }
        finally:
            for lk in reversed(self._locks):
                lk.release()
