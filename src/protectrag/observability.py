"""Structured logging and optional OpenTelemetry hooks for RAG ingest screening."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

from protectrag.context import RunContext
from protectrag.scanner import DocumentScanResult

_logger = logging.getLogger("protectrag")

# Optional OTel: use only if opentelemetry is installed
try:
    from opentelemetry import trace  # type: ignore

    _tracer = trace.get_tracer("protectrag", "0.5.0")
    _HAS_OTEL = True
except Exception:  # pragma: no cover - optional dependency
    _tracer = None
    _HAS_OTEL = False

_RULE_DESCRIPTIONS: dict[str, str] = {
    "instruction_override": "Text attempts to override or replace the model's instructions",
    "instruction_override_multilang": "Instruction override in a non-English language",
    "role_delimiter": "Fake system/assistant/user role markers injected into content",
    "prompt_leak_request": "Attempts to extract or reveal the system prompt",
    "data_exfiltration": "Attempts to send retrieved data to an external URL",
    "fake_tool_or_api": "Fake function call or tool invocation injected into content",
    "encoding_trick": "Obfuscation via base64, rot13, hex, or similar encoding",
    "unicode_manipulation": "Invisible or bidirectional Unicode characters used to hide content",
    "html_script_injection": "HTML script tags, iframes, or event handlers in content",
    "markdown_injection": "Markdown-based injection (data URIs, event handlers, comments)",
    "indirect_injection": "Deferred attack triggered when text is retrieved for specific queries",
    "payload_splitting": "Instructions split across multiple chunks to evade detection",
    "llm_classifier": "Flagged by the LLM-based classifier",
    "llm_parse_error": "LLM returned an unparseable response (treated as suspicious)",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _build_rule_explanations(matched_rules: list[str]) -> list[dict[str, str]]:
    return [
        {"rule": r, "description": _RULE_DESCRIPTIONS.get(r, r)}
        for r in matched_rules
    ]


def emit_ingest_event(
    result: DocumentScanResult,
    *,
    action: str,
    extra: dict[str, Any] | None = None,
    context: RunContext | None = None,
    latency_ms: float | None = None,
    text_preview: str | None = None,
) -> None:
    """
    Emit one structured event for dashboards / SIEM.

    action: e.g. 'ingest_blocked', 'ingest_allowed_with_warning', 'ingest_allowed'

    Pass :class:`~protectrag.context.RunContext` to correlate with experiments
    or production deployments (run_id, project, environment).
    """
    explanations = _build_rule_explanations(result.matched_rules)
    rationale = result.rationale
    if not rationale and explanations:
        rationale = "; ".join(e["description"] for e in explanations)

    payload: dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "event": "rag_document_screen",
        "action": action,
        "document_id": result.document_id,
        "severity": result.severity.name,
        "severity_numeric": int(result.severity),
        "score": result.score,
        "detector": result.detector,
        "matched_rules": result.matched_rules,
        "rule_explanations": explanations,
        "snippets": result.snippets,
        "rationale": rationale,
        "should_alert": result.should_alert,
    }
    if latency_ms is not None:
        payload["latency_ms"] = round(latency_ms, 2)
    if text_preview is not None:
        payload["text_preview"] = text_preview[:200]
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
) -> tuple[DocumentScanResult, float]:
    """
    Run a scan, measure latency, optionally inside an OTel span.

    Returns (result, latency_ms).
    """
    if not _HAS_OTEL or _tracer is None:
        t0 = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        return result, elapsed
    with _tracer.start_as_current_span(span_name) as span:
        span.set_attribute("protectrag.document_id", document_id)
        t0 = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        span.set_attribute("protectrag.severity", result.severity.name)
        span.set_attribute("protectrag.score", result.score)
        span.set_attribute("protectrag.alert", result.should_alert)
        span.set_attribute("protectrag.detector", result.detector)
        span.set_attribute("protectrag.latency_ms", elapsed)
        return result, elapsed


def result_to_telemetry_dict(result: DocumentScanResult) -> dict[str, Any]:
    """Stable dict for exporting to your metrics backend (Prometheus labels, etc.)."""
    d = asdict(result)
    d["severity"] = result.severity.name
    d["severity_numeric"] = int(result.severity)
    return d
