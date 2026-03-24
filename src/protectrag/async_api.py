"""Async scan + batch API for high-throughput pipelines."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from protectrag.scanner import DocumentScanResult, InjectionSeverity, scan_document_for_injection


@dataclass(frozen=True)
class BatchResult:
    """Results of scanning a batch of documents."""

    results: list[DocumentScanResult]
    total: int
    blocked: int
    warned: int
    allowed: int

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "blocked": self.blocked,
            "warned": self.warned,
            "allowed": self.allowed,
            "block_rate": round(self.blocked / self.total, 4) if self.total else 0.0,
        }


ScanFn = Callable[[str, str], DocumentScanResult]


async def async_scan(
    text: str,
    *,
    document_id: str = "unknown",
    scan_fn: ScanFn | None = None,
) -> DocumentScanResult:
    """
    Run a scan in the default executor so it doesn't block the event loop.

    For heuristic-only (CPU-bound regex), this uses a thread.
    For LLM scans, you can also pass a sync scan_fn and it'll be wrapped.
    """
    fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, text, document_id)


async def async_scan_batch(
    items: Sequence[tuple[str, str]],
    *,
    scan_fn: ScanFn | None = None,
    max_concurrency: int = 10,
    warn_on: InjectionSeverity = InjectionSeverity.MEDIUM,
    block_on: InjectionSeverity = InjectionSeverity.HIGH,
) -> BatchResult:
    """
    Scan many (text, document_id) pairs concurrently.

    ``max_concurrency`` caps parallel tasks (important for LLM rate limits).
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(text: str, doc_id: str) -> DocumentScanResult:
        async with sem:
            return await async_scan(text, document_id=doc_id, scan_fn=scan_fn)

    tasks = [_one(text, doc_id) for text, doc_id in items]
    results = await asyncio.gather(*tasks)

    blocked = sum(1 for r in results if r.severity >= block_on)
    warned = sum(1 for r in results if warn_on <= r.severity < block_on)
    allowed = len(results) - blocked - warned

    return BatchResult(
        results=list(results),
        total=len(results),
        blocked=blocked,
        warned=warned,
        allowed=allowed,
    )
