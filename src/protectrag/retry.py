"""Retry + fallback logic for LLM calls (exponential backoff, degrade to heuristic)."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from protectrag.scanner import DocumentScanResult, scan_document_for_injection

_logger = logging.getLogger("protectrag.retry")


@dataclass
class RetryConfig:
    """Backoff settings for LLM HTTP calls."""

    max_retries: int = 3
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)
    fallback_to_heuristic: bool = True


def _sleep_for(attempt: int, cfg: RetryConfig) -> float:
    delay = min(cfg.initial_backoff_s * (cfg.backoff_multiplier ** attempt), cfg.max_backoff_s)
    if cfg.jitter:
        delay *= 0.5 + random.random()
    return delay


def with_retry(
    fn: Callable[[], DocumentScanResult],
    *,
    text: str,
    document_id: str,
    config: RetryConfig | None = None,
) -> DocumentScanResult:
    """
    Call ``fn`` with retry on transient HTTP errors.

    On permanent failure (exhausted retries), falls back to heuristic scan
    if ``config.fallback_to_heuristic`` is True, else re-raises.
    """
    cfg = config or RetryConfig()
    last_exc: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            status = _extract_status(exc)
            if status is not None and status not in cfg.retryable_status_codes:
                break
            if attempt < cfg.max_retries:
                delay = _sleep_for(attempt, cfg)
                _logger.warning(
                    "Retry %d/%d after %.1fs (status=%s, doc=%s)",
                    attempt + 1, cfg.max_retries, delay, status, document_id,
                )
                time.sleep(delay)

    if cfg.fallback_to_heuristic:
        _logger.warning(
            "LLM scan failed after %d retries for doc=%s; falling back to heuristic. error=%s",
            cfg.max_retries, document_id, last_exc,
        )
        return scan_document_for_injection(text, document_id=document_id)

    raise last_exc  # type: ignore[misc]


def _extract_status(exc: Exception) -> int | None:
    """Best-effort extraction of HTTP status from httpx or generic exceptions."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    return None
