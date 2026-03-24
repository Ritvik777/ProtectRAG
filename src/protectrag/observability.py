"""Structured logging and optional OpenTelemetry hooks for RAG ingest screening."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Any, Callable

from protectrag.context import RunContext
from protectrag.scanner import DocumentScanResult

_logger = logging.getLogger("protectrag")

# Optional OTel: use only if opentelemetry is installed
try:
    from opentelemetry import trace  # type: ignore

    _tracer = trace.get_tracer("protectrag", "0.1.0")
    _HAS_OTEL = True
except Exception:  # pragma: no cover - optional dependency
    _tracer = None
    _HAS_OTEL = False


def configure_logging(level: int = logging.INFO, json_format: bool = True) -> None:
    """Configure root protectrag logger (call once at app startup)."""
    if _logger.handlers:
        return
    handler = logging.StreamHandler()
    if json_format:

        class JsonFormatter(logging.Formatter):
            """If the log message is already JSON, emit one line; else wrap."""

            def format(self, record: logging.LogRecord) -> str:
                msg = record.getMessage()
                if msg.startswith("{") and msg.rstrip().endswith("}"):
                    return msg
                return json.dumps(
                    {
                        "level": record.levelname,
                        "logger": record.name,
                        "message": msg,
                    },
                    default=str,
                )

        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    _logger.addHandler(handler)
    _logger.setLevel(level)


def emit_ingest_event(
    result: DocumentScanResult,
    *,
    action: str,
    extra: dict[str, Any] | None = None,
    context: RunContext | None = None,
) -> None:
    """
    Emit one structured event for dashboards / SIEM.

    action: e.g. 'ingest_blocked', 'ingest_allowed_with_warning', 'ingest_allowed'

    Pass :class:`~protectrag.context.RunContext` to correlate with experiments
    or production deployments (run_id, project, environment).
    """
    payload: dict[str, Any] = {
        "event": "rag_document_screen",
        "action": action,
        "document_id": result.document_id,
        "severity": result.severity.name,
        "score": result.score,
        "matched_rules": result.matched_rules,
        "should_alert": result.should_alert,
        "detector": result.detector,
        "rationale": result.rationale,
    }
    if context is not None:
        payload.update(context.to_log_fields())
    if extra:
        payload.update(extra)
    line = json.dumps(payload, default=str)
    if action == "ingest_allowed":
        _logger.debug(line)
    elif result.should_alert:
        _logger.warning(line)
    else:
        _logger.info(line)


def trace_ingest_screen(
    document_id: str,
    fn: Callable[[], DocumentScanResult],
    *,
    span_name: str = "protectrag.ingest_screen",
) -> DocumentScanResult:
    """Run a scan inside a span when OpenTelemetry is available (Phoenix-style trace segment)."""
    if not _HAS_OTEL or _tracer is None:
        return fn()
    with _tracer.start_as_current_span(span_name) as span:
        span.set_attribute("protectrag.document_id", document_id)
        t0 = time.perf_counter()
        result = fn()
        span.set_attribute("protectrag.severity", result.severity.name)
        span.set_attribute("protectrag.score", result.score)
        span.set_attribute("protectrag.alert", result.should_alert)
        span.set_attribute("protectrag.detector", result.detector)
        span.set_attribute("protectrag.latency_ms", (time.perf_counter() - t0) * 1000)
        return result


def result_to_telemetry_dict(result: DocumentScanResult) -> dict[str, Any]:
    """Stable dict for exporting to your metrics backend (Prometheus labels, etc.)."""
    d = asdict(result)
    d["severity"] = result.severity.name
    return d
