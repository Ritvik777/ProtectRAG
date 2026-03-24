"""Ingest pipeline helpers: scan → decide → observable events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from protectrag.callbacks import CallbackRegistry
from protectrag.context import RunContext
from protectrag.metrics import MetricsSink
from protectrag.observability import emit_ingest_event, trace_ingest_screen
from protectrag.scanner import DocumentScanResult, InjectionSeverity, scan_document_for_injection


class IngestDecision(str, Enum):
    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    BLOCK = "block"


@dataclass(frozen=True)
class IngestResult:
    decision: IngestDecision
    scan: DocumentScanResult
    message: str
    latency_ms: float = 0.0


def _text_preview(text: str, max_len: int = 200) -> str:
    t = " ".join(text.split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def ingest_document(
    text: str,
    *,
    document_id: str = "unknown",
    block_on: InjectionSeverity = InjectionSeverity.HIGH,
    warn_on: InjectionSeverity = InjectionSeverity.MEDIUM,
    scan: Callable[[str, str], DocumentScanResult] | None = None,
    context: RunContext | None = None,
    metrics: MetricsSink | None = None,
    callbacks: CallbackRegistry | None = None,
) -> IngestResult:
    """
    Full ingest check with observability.

    - BLOCK: severity >= block_on (default HIGH)
    - ALLOW_WITH_WARNING: severity >= warn_on but < block_on
    - ALLOW: below warn_on

    Pass ``scan`` to use an LLM or hybrid scanner: ``scan=lambda t, d: hybrid.scan(t, document_id=d)``.
    Default is heuristic-only :func:`~protectrag.scanner.scan_document_for_injection`.

    ``context`` attaches run/project metadata to logs (experiment or deploy correlation).
    ``metrics`` receives counters when implementing :class:`~protectrag.metrics.MetricsSink`.
    ``callbacks`` fires user-defined hooks on block / warn / allow decisions.
    """
    def _run() -> DocumentScanResult:
        if scan is None:
            return scan_document_for_injection(text, document_id=document_id)
        return scan(text, document_id)

    scan_result, latency_ms = trace_ingest_screen(document_id, _run)
    preview = _text_preview(text)

    def _emit_metrics(decision: IngestDecision) -> None:
        if metrics is None:
            return
        metrics.increment(
            "protectrag_ingest_total",
            decision=decision.value,
            severity=scan_result.severity.name,
            detector=scan_result.detector,
        )
        metrics.observe(
            "protectrag_ingest_severity_numeric",
            float(scan_result.severity.value),
            decision=decision.value,
        )
        metrics.observe(
            "protectrag_ingest_latency_ms",
            latency_ms,
            decision=decision.value,
        )

    if scan_result.severity >= block_on:
        emit_ingest_event(
            scan_result,
            action="ingest_blocked",
            extra={"block_threshold": block_on.name},
            context=context,
            latency_ms=latency_ms,
            text_preview=preview,
        )
        _emit_metrics(IngestDecision.BLOCK)
        if callbacks:
            callbacks.fire_block(text, scan_result)
        return IngestResult(
            decision=IngestDecision.BLOCK,
            scan=scan_result,
            message=f"Blocked: injection severity {scan_result.severity.name} (threshold {block_on.name})",
            latency_ms=round(latency_ms, 2),
        )

    if scan_result.severity >= warn_on:
        emit_ingest_event(
            scan_result,
            action="ingest_allowed_with_warning",
            extra={"warn_threshold": warn_on.name},
            context=context,
            latency_ms=latency_ms,
            text_preview=preview,
        )
        _emit_metrics(IngestDecision.ALLOW_WITH_WARNING)
        if callbacks:
            callbacks.fire_warn(text, scan_result)
        return IngestResult(
            decision=IngestDecision.ALLOW_WITH_WARNING,
            scan=scan_result,
            message=f"Allowed with warning: severity {scan_result.severity.name}",
            latency_ms=round(latency_ms, 2),
        )

    emit_ingest_event(
        scan_result,
        action="ingest_allowed",
        context=context,
        latency_ms=latency_ms,
    )
    _emit_metrics(IngestDecision.ALLOW)
    if callbacks:
        callbacks.fire_allow(text, scan_result)
    return IngestResult(
        decision=IngestDecision.ALLOW,
        scan=scan_result,
        message="Clean",
        latency_ms=round(latency_ms, 2),
    )
