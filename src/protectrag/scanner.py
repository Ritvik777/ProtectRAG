"""Heuristic prompt-injection detection for text destined for RAG stores."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum


class InjectionSeverity(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


# Patterns often used to hijack model behavior when embedded in retrieved chunks.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    (
        "instruction_override",
        re.compile(
            r"(?i)\b("
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|text|context)|"
            r"disregard\s+(all\s+)?(previous|prior|above)|"
            r"forget\s+(everything|all)\s+(above|before)|"
            r"new\s+instructions?\s*:|"
            r"override\s+(the\s+)?(system|developer)|"
            r"you\s+are\s+now\s+(a|an|the)\b"
            r")\b",
        ),
        3,
    ),
    (
        "role_delimiter",
        re.compile(
            r"(?i)(<\|?(?:system|assistant|user|developer)\|?>|"
            r"\[?\s*(?:SYSTEM|ASSISTANT|USER)\s*\]?|"
            r"^###\s*(?:system|assistant|user)\s*$)",
            re.MULTILINE,
        ),
        2,
    ),
    (
        "prompt_leak_request",
        re.compile(
            r"(?i)\b("
            r"repeat\s+(your|the)\s+(system\s+)?prompt|"
            r"print\s+(your\s+)?(system\s+)?prompt|"
            r"output\s+(your\s+)?(entire\s+)?(system\s+)?prompt|"
            r"what\s+(are|were)\s+your\s+(original\s+)?instructions"
            r")\b",
        ),
        3,
    ),
    (
        "fake_tool_or_api",
        re.compile(
            r"(?i)\b("
            r"function\s*call\s*:|"
            r"```\s*json\s*\{[^}]*\"role\"\s*:\s*\"system\""
            r")\b",
        ),
        2,
    ),
    (
        "encoding_trick",
        re.compile(r"(?i)(base64\s*decode|rot13|from\s+hex|unicode\s+escape)"),
        1,
    ),
]


@dataclass(frozen=True)
class DocumentScanResult:
    """Outcome of scanning a single document or chunk."""

    document_id: str
    severity: InjectionSeverity
    score: float  # 0.0–1.0 aggregate heuristic confidence
    matched_rules: list[str] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)  # short excerpts for observability
    detector: str = "heuristic"  # "heuristic" | "llm" | "hybrid"
    rationale: str | None = None  # short LLM or hybrid explanation when available

    @property
    def should_alert(self) -> bool:
        return self.severity >= InjectionSeverity.MEDIUM


def _clip(text: str, max_len: int = 120) -> str:
    t = " ".join(text.split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def scan_document_for_injection(
    text: str,
    *,
    document_id: str = "unknown",
) -> DocumentScanResult:
    """
    Scan raw text for likely prompt-injection content.

    This is a **heuristic** screen (fast, no LLM). Use it at ingest time and
    optionally combine with a classifier in production.
    """
    if not text or not text.strip():
        return DocumentScanResult(
            document_id=document_id,
            severity=InjectionSeverity.NONE,
            score=0.0,
            detector="heuristic",
        )

    matched: list[str] = []
    snippets: list[str] = []
    max_weight = 0
    total_weight = 0.0

    for name, pattern, weight in _INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            matched.append(name)
            total_weight += weight
            max_weight = max(max_weight, weight)
            span = m.group(0)
            snippets.append(_clip(span))

    if not matched:
        return DocumentScanResult(
            document_id=document_id,
            severity=InjectionSeverity.NONE,
            score=0.0,
            detector="heuristic",
        )

    # Normalize score: more matches and higher max weight → higher score
    n = len(_INJECTION_PATTERNS)
    score = min(1.0, (total_weight / (3 * n)) * 3 + (max_weight / 3) * 0.5)

    if max_weight >= 3 or len(matched) >= 3:
        severity = InjectionSeverity.HIGH
    elif max_weight >= 2 or len(matched) >= 2:
        severity = InjectionSeverity.MEDIUM
    else:
        severity = InjectionSeverity.LOW

    return DocumentScanResult(
        document_id=document_id,
        severity=severity,
        score=round(score, 4),
        matched_rules=matched,
        snippets=snippets[:5],
        detector="heuristic",
    )
