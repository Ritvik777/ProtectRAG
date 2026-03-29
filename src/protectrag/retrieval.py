"""Retrieval-time screening: scan chunks *after* they are retrieved, before they reach the LLM."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from protectrag.async_api import AsyncScanFn, async_scan
from protectrag.scanner import DocumentScanResult, InjectionSeverity, scan_document_for_injection

ScanFn = Callable[[str, str], DocumentScanResult]


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a vector DB query, with optional metadata."""

    text: str
    chunk_id: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # similarity / relevance score from the vector DB


@dataclass(frozen=True)
class ScreenedChunk:
    """Result of screening one retrieved chunk."""

    chunk: RetrievedChunk
    scan: DocumentScanResult
    passed: bool  # True = safe to feed to the LLM


@dataclass(frozen=True)
class RetrievalScreenResult:
    """Aggregate result of screening all retrieved chunks."""

    screened: list[ScreenedChunk]
    passed_chunks: list[RetrievedChunk]
    blocked_chunks: list[RetrievedChunk]
    total: int
    n_passed: int
    n_blocked: int

    def passed_texts(self) -> list[str]:
        return [c.text for c in self.passed_chunks]


def screen_retrieved_chunks(
    chunks: list[RetrievedChunk],
    *,
    scan_fn: ScanFn | None = None,
    block_on: InjectionSeverity = InjectionSeverity.MEDIUM,
    max_workers: int | None = None,
) -> RetrievalScreenResult:
    """
    Screen chunks **after retrieval** and **before** they are passed as context to the LLM.

    Chunks with severity >= ``block_on`` are filtered out. The default (MEDIUM)
    is stricter than ingest-time because at retrieval time you're about to send
    text directly to the model.

    Pass ``scan_fn`` to use hybrid or LLM classification (same signature as ingest).

    ``max_workers`` > 1 runs scans in a thread pool (order of ``screened`` matches
    ``chunks``). Default is sequential.
    """
    fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))

    screened: list[ScreenedChunk] = []
    passed: list[RetrievedChunk] = []
    blocked: list[RetrievedChunk] = []

    n = len(chunks)
    workers = max_workers if max_workers is not None else 1
    if workers <= 1 or n <= 1:
        results = [fn(c.text, c.chunk_id) for c in chunks]
    else:
        w = min(max(1, workers), n, min(32, (os.cpu_count() or 1) + 4))
        with ThreadPoolExecutor(max_workers=w) as pool:
            results = list(pool.map(lambda c: fn(c.text, c.chunk_id), chunks))

    for chunk, r in zip(chunks, results):
        ok = r.severity < block_on
        screened.append(ScreenedChunk(chunk=chunk, scan=r, passed=ok))
        if ok:
            passed.append(chunk)
        else:
            blocked.append(chunk)

    return RetrievalScreenResult(
        screened=screened,
        passed_chunks=passed,
        blocked_chunks=blocked,
        total=len(chunks),
        n_passed=len(passed),
        n_blocked=len(blocked),
    )


async def screen_retrieved_chunks_async(
    chunks: list[RetrievedChunk],
    *,
    scan_fn: ScanFn | None = None,
    async_scan_fn: AsyncScanFn | None = None,
    block_on: InjectionSeverity = InjectionSeverity.MEDIUM,
    max_concurrency: int = 10,
) -> RetrievalScreenResult:
    """
    Same as :func:`screen_retrieved_chunks`, but run scans concurrently (bounded
    by ``max_concurrency``). Each scan uses :func:`~protectrag.async_api.async_scan`
    so sync ``scan_fn`` runs in a thread pool and does not block the event loop.

    Pass ``async_scan_fn`` for native async classifiers (e.g. ``hybrid.ascan``).
    """
    if not chunks:
        return RetrievalScreenResult(
            screened=[],
            passed_chunks=[],
            blocked_chunks=[],
            total=0,
            n_passed=0,
            n_blocked=0,
        )

    sem = asyncio.Semaphore(max_concurrency)

    async def _one(c: RetrievedChunk) -> tuple[RetrievedChunk, DocumentScanResult]:
        async with sem:
            r = await async_scan(
                c.text,
                document_id=c.chunk_id,
                scan_fn=scan_fn,
                async_scan_fn=async_scan_fn,
            )
            return c, r

    pairs = await asyncio.gather(*(_one(c) for c in chunks))

    screened: list[ScreenedChunk] = []
    passed: list[RetrievedChunk] = []
    blocked: list[RetrievedChunk] = []

    for chunk, r in pairs:
        ok = r.severity < block_on
        screened.append(ScreenedChunk(chunk=chunk, scan=r, passed=ok))
        if ok:
            passed.append(chunk)
        else:
            blocked.append(chunk)

    return RetrievalScreenResult(
        screened=screened,
        passed_chunks=passed,
        blocked_chunks=blocked,
        total=len(chunks),
        n_passed=len(passed),
        n_blocked=len(blocked),
    )
