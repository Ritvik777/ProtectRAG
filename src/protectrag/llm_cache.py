"""Pluggable LLM classification caches (cross-process / multi-replica deduplication)."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Protocol

from protectrag.scanner import DocumentScanResult, InjectionSeverity


def document_scan_result_to_dict(r: DocumentScanResult) -> dict[str, Any]:
    """JSON-serializable dict for cache backends."""
    return {
        "document_id": r.document_id,
        "severity": int(r.severity),
        "score": r.score,
        "matched_rules": r.matched_rules,
        "snippets": r.snippets,
        "detector": r.detector,
        "rationale": r.rationale,
    }


def document_scan_result_from_dict(d: dict[str, Any]) -> DocumentScanResult:
    """Restore :class:`~protectrag.scanner.DocumentScanResult` from :func:`document_scan_result_to_dict`."""
    return DocumentScanResult(
        document_id=str(d.get("document_id", "unknown")),
        severity=InjectionSeverity(int(d["severity"])),
        score=float(d.get("score", 0.0)),
        matched_rules=list(d.get("matched_rules") or []),
        snippets=list(d.get("snippets") or []),
        detector=str(d.get("detector", "llm")),
        rationale=d.get("rationale"),
    )


class LLMClassificationCache(Protocol):
    """Sync key-value cache for LLM scan results (Redis, Memcached, etc.)."""

    def get(self, key: str) -> DocumentScanResult | None:
        """Return a cached result or ``None``."""

    def set(self, key: str, value: DocumentScanResult) -> None:
        """Store a result (implementations may normalize ``document_id``)."""


class RedisLLMClassificationCache:
    """
    Redis-backed cache shared across workers (install ``protectrag[redis]``).

    Uses JSON values. Typical key: ``protectrag:llm:v1:<sha256>``.
    """

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = "protectrag:llm:v1:",
        ttl_seconds: int | None = 86_400,
    ) -> None:
        try:
            import redis  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Redis cache requires redis. Install with: pip install 'protectrag[redis]'"
            ) from e
        self._client = client
        self._key_prefix = key_prefix
        self._ttl = ttl_seconds

    def get(self, key: str) -> DocumentScanResult | None:
        raw = self._client.get(self._key_prefix + key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        d = json.loads(raw)
        return document_scan_result_from_dict(d)

    def set(self, key: str, value: DocumentScanResult) -> None:
        stored = replace(value, document_id="cached")
        payload = json.dumps(document_scan_result_to_dict(stored), default=str)
        k = self._key_prefix + key
        if self._ttl is not None and self._ttl > 0:
            self._client.setex(k, self._ttl, payload)
        else:
            self._client.set(k, payload)
