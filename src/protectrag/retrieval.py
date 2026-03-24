"""Retrieval-time screening: scan chunks *after* they are retrieved, before they reach the LLM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from protectrag.scanner import DocumentScanResult, InjectionSeverity, scan_document_for_injection


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
    scan_fn: Callable[[str, str], DocumentScanResult] | None = None,
    block_on: InjectionSeverity = InjectionSeverity.MEDIUM,
) -> RetrievalScreenResult:
    """
    Screen chunks **after retrieval** and **before** they are passed as context to the LLM.

    Chunks with severity >= ``block_on`` are filtered out. The default (MEDIUM)
    is stricter than ingest-time because at retrieval time you're about to send
    text directly to the model.

    Pass ``scan_fn`` to use hybrid or LLM classification (same signature as ingest).
    """
    fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))

    screened: list[ScreenedChunk] = []
    passed: list[RetrievedChunk] = []
    blocked: list[RetrievedChunk] = []

    for chunk in chunks:
        r = fn(chunk.text, chunk.chunk_id)
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
