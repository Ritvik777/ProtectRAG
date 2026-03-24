"""
Suggested attribute names aligned with OpenTelemetry GenAI / tracing conventions.

Use these when exporting to Phoenix, Jaeger, or vendor backends so spans and logs
line up with LLM observability tooling (see OTel GenAI semantic conventions).
"""

from __future__ import annotations

from typing import Any

from protectrag.scanner import DocumentScanResult

# Common keys (vendors may extend; see OpenTelemetry semantic conventions for gen_ai.*)
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_SYSTEM = "gen_ai.system"

PROTECTRAG_NAMESPACE = "protectrag"

# Suggested span names when composing a full RAG trace (retrieve → chunk → guardrail).
SPAN_RAG_RETRIEVE = "rag.retrieve"
SPAN_RAG_CHUNK = "rag.chunk"
SPAN_RAG_EMBEDDING = "rag.embedding"
SPAN_PROTECTRAG_INGEST = "protectrag.ingest_screen"


def span_attributes_for_ingest_scan(
    result: DocumentScanResult,
    *,
    operation_name: str = "protectrag.ingest_screen",
    latency_ms: float | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Flat dict suitable for OpenTelemetry span attributes or log enrichment.

    Phoenix-style: attach these to the span that wraps retrieval + guardrail steps.
    """
    attrs: dict[str, Any] = {
        GEN_AI_OPERATION_NAME: operation_name,
        f"{PROTECTRAG_NAMESPACE}.document_id": result.document_id,
        f"{PROTECTRAG_NAMESPACE}.severity": result.severity.name,
        f"{PROTECTRAG_NAMESPACE}.score": result.score,
        f"{PROTECTRAG_NAMESPACE}.should_alert": result.should_alert,
        f"{PROTECTRAG_NAMESPACE}.detector": result.detector,
        f"{PROTECTRAG_NAMESPACE}.matched_rules": ",".join(result.matched_rules),
    }
    if result.rationale:
        attrs[f"{PROTECTRAG_NAMESPACE}.rationale"] = result.rationale[:2000]
    if latency_ms is not None:
        attrs[f"{PROTECTRAG_NAMESPACE}.latency_ms"] = latency_ms
    if model:
        attrs[GEN_AI_REQUEST_MODEL] = model
        attrs[GEN_AI_SYSTEM] = "protectrag"
    return attrs
