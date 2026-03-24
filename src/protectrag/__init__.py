"""ProtectRAG: evals and observability helpers for RAG document security screening."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("protectrag")
except PackageNotFoundError:  # pragma: no cover - local dev without install
    __version__ = "0.0.0"

from protectrag.context import RunContext
from protectrag.evals import EvalCase, EvalReport, GroundTruth, run_eval_dataset
from protectrag.ingest import IngestDecision, IngestResult, ingest_document
from protectrag.metrics import InMemoryMetrics, MetricsSink
from protectrag.observability import (
    configure_logging,
    emit_ingest_event,
    result_to_telemetry_dict,
    trace_ingest_screen,
)
from protectrag.scanner import (
    DocumentScanResult,
    InjectionSeverity,
    scan_document_for_injection,
)
from protectrag.semconv import span_attributes_for_ingest_scan
from protectrag.llm import (
    HybridPolicy,
    HybridScanner,
    LLMScanConfig,
    LLMScanner,
    scan_document_llm,
)

__all__ = [
    "__version__",
    "DocumentScanResult",
    "EvalCase",
    "EvalReport",
    "GroundTruth",
    "HybridPolicy",
    "HybridScanner",
    "InMemoryMetrics",
    "InjectionSeverity",
    "IngestDecision",
    "IngestResult",
    "LLMScanConfig",
    "LLMScanner",
    "MetricsSink",
    "RunContext",
    "configure_logging",
    "emit_ingest_event",
    "ingest_document",
    "result_to_telemetry_dict",
    "run_eval_dataset",
    "scan_document_for_injection",
    "scan_document_llm",
    "span_attributes_for_ingest_scan",
    "trace_ingest_screen",
]
