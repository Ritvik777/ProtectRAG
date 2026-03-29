"""Async scan + batch API for high-throughput pipelines."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
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
AsyncScanFn = Callable[[str, str], Awaitable[DocumentScanResult]]


async def async_scan(
    text: str,
    *,
    document_id: str = "unknown",
    scan_fn: ScanFn | None = None,
    async_scan_fn: AsyncScanFn | None = None,
) -> DocumentScanResult:
    """
    Run a scan in the default executor so it doesn't block the event loop.

    For heuristic-only (CPU-bound regex), this uses a thread.
    For LLM scans, you can also pass a sync ``scan_fn`` and it'll be wrapped.

    If ``async_scan_fn`` is set (e.g. ``hybrid.ascan``), it is awaited directly
    and ``scan_fn`` is ignored.
    """
    if async_scan_fn is not None:
        return await async_scan_fn(text, document_id)
    fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, text, document_id)


async def async_scan_batch(
    items: Sequence[tuple[str, str]],
    *,
    scan_fn: ScanFn | None = None,
    async_scan_fn: AsyncScanFn | None = None,
    max_concurrency: int = 10,
    warn_on: InjectionSeverity = InjectionSeverity.MEDIUM,
    block_on: InjectionSeverity = InjectionSeverity.HIGH,
    batch_chunk_size: int = 2000,
) -> BatchResult:
    """
    Scan many (text, document_id) pairs concurrently.

    ``max_concurrency`` caps parallel tasks (important for LLM rate limits).

    ``batch_chunk_size`` limits how many asyncio tasks are scheduled at once
    (default 2000) so very large batches do not allocate one Task per item in
    a single gather. Chunks are processed sequentially; each chunk still
    respects ``max_concurrency``.
    """
    items_list = list(items)
    if not items_list:
        return BatchResult(results=[], total=0, blocked=0, warned=0, allowed=0)

    chunk_sz = batch_chunk_size if batch_chunk_size > 0 else len(items_list)
    results: list[DocumentScanResult] = []

    for offset in range(0, len(items_list), chunk_sz):
        slice_ = items_list[offset : offset + chunk_sz]
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(text: str, doc_id: str) -> DocumentScanResult:
            async with sem:
                return await async_scan(
                    text,
                    document_id=doc_id,
                    scan_fn=scan_fn,
                    async_scan_fn=async_scan_fn,
                )

        chunk_results = await asyncio.gather(
            *(_one(text, doc_id) for text, doc_id in slice_)
        )
        results.extend(chunk_results)

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
