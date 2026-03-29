"""ProtectRAG: evals and observability helpers for RAG document security screening."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("protectrag")
except PackageNotFoundError:  # pragma: no cover - local dev without install
    __version__ = "0.0.0"

from protectrag.async_api import AsyncScanFn, BatchResult, async_scan, async_scan_batch
from protectrag.callbacks import CallbackRegistry
from protectrag.context import RunContext
from protectrag.datasets import load_golden_v1
from protectrag.evals import EvalCase, EvalReport, GroundTruth, run_eval_dataset
from protectrag.ingest import IngestDecision, IngestResult, ingest_document, ingest_document_async
from protectrag.llm_cache import (
    LLMClassificationCache,
    RedisLLMClassificationCache,
    document_scan_result_from_dict,
    document_scan_result_to_dict,
)
from protectrag.llm import (
    HybridPolicy,
    HybridScanner,
    LLMScanConfig,
    LLMScanner,
    scan_document_llm,
)
from protectrag.metrics import InMemoryMetrics, MetricsSink
from protectrag.observability import (
    configure_logging,
    emit_ingest_event,
    result_to_telemetry_dict,
    trace_ingest_screen,
    trace_ingest_screen_async,
)
from protectrag.retrieval import (
    RetrievalScreenResult,
    RetrievedChunk,
    ScreenedChunk,
    screen_retrieved_chunks,
    screen_retrieved_chunks_async,
)
from protectrag.retry import RetryConfig, with_retry, with_retry_async
from protectrag.scanner import (
    DocumentScanResult,
    InjectionSeverity,
    scan_document_for_injection,
)
from protectrag.semconv import span_attributes_for_ingest_scan

__all__ = [
    "__version__",
    # Core
    "DocumentScanResult",
    "InjectionSeverity",
    "IngestDecision",
    "IngestResult",
    "scan_document_for_injection",
    "ingest_document",
    "ingest_document_async",
    # LLM cache
    "LLMClassificationCache",
    "RedisLLMClassificationCache",
    "document_scan_result_to_dict",
    "document_scan_result_from_dict",
    # LLM
    "HybridPolicy",
    "HybridScanner",
    "LLMScanConfig",
    "LLMScanner",
    "scan_document_llm",
    # Async / batch
    "BatchResult",
    "AsyncScanFn",
    "async_scan",
    "async_scan_batch",
    # Retrieval-time
    "RetrievedChunk",
    "ScreenedChunk",
    "RetrievalScreenResult",
    "screen_retrieved_chunks",
    "screen_retrieved_chunks_async",
    # Retry
    "RetryConfig",
    "with_retry",
    "with_retry_async",
    # Callbacks
    "CallbackRegistry",
    # Evals
    "EvalCase",
    "EvalReport",
    "GroundTruth",
    "run_eval_dataset",
    "load_golden_v1",
    # Observability
    "RunContext",
    "InMemoryMetrics",
    "MetricsSink",
    "configure_logging",
    "emit_ingest_event",
    "result_to_telemetry_dict",
    "span_attributes_for_ingest_scan",
    "trace_ingest_screen",
    "trace_ingest_screen_async",
]
