"""Ingest pipeline helpers: scan → decide → observable events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
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


def ingest_document(
    text: str,
    *,
    document_id: str = "unknown",
    block_on: InjectionSeverity = InjectionSeverity.HIGH,
    warn_on: InjectionSeverity = InjectionSeverity.MEDIUM,
    scan: Callable[[str, str], DocumentScanResult] | None = None,
    context: RunContext | None = None,
    metrics: MetricsSink | None = None,
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
    """
    def _run() -> DocumentScanResult:
        if scan is None:
            return scan_document_for_injection(text, document_id=document_id)
        return scan(text, document_id)

    scan_result = trace_ingest_screen(document_id, _run)

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

    if scan_result.severity >= block_on:
        emit_ingest_event(
            scan_result,
            action="ingest_blocked",
            extra={"block_threshold": block_on.name},
            context=context,
        )
        _emit_metrics(IngestDecision.BLOCK)
        return IngestResult(
            decision=IngestDecision.BLOCK,
            scan=scan_result,
            message=f"Blocked: injection severity {scan_result.severity.name} (threshold {block_on.name})",
        )

    if scan_result.severity >= warn_on:
        emit_ingest_event(
            scan_result,
            action="ingest_allowed_with_warning",
            extra={"warn_threshold": warn_on.name},
            context=context,
        )
        _emit_metrics(IngestDecision.ALLOW_WITH_WARNING)
        return IngestResult(
            decision=IngestDecision.ALLOW_WITH_WARNING,
            scan=scan_result,
            message=f"Allowed with warning: severity {scan_result.severity.name}",
        )

    emit_ingest_event(scan_result, action="ingest_allowed", context=context)
    _emit_metrics(IngestDecision.ALLOW)
    return IngestResult(
        decision=IngestDecision.ALLOW,
        scan=scan_result,
        message="Clean",
    )
