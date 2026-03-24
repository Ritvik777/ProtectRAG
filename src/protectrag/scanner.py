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


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------
# Each entry: (rule_name, compiled_regex, weight 1–3).
# Higher weight = stronger signal.  Rules are grouped by attack family.
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    # ── Instruction override ──────────────────────────────────────────────
    (
        "instruction_override",
        re.compile(
            r"(?i)\b("
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|text|context|prompts?)|"
            r"disregard\s+(all\s+)?(previous|prior|above)|"
            r"forget\s+(everything|all)\s+(above|before|you\s+know)|"
            r"do\s+not\s+follow\s+(the\s+)?(previous|above|prior)|"
            r"stop\s+being\s+(a|an)\s+\w+\s+assistant|"
            r"new\s+instructions?\s*:|"
            r"override\s+(the\s+)?(system|developer)|"
            r"you\s+are\s+now\s+(a|an|the)\b|"
            r"act\s+as\s+(if\s+)?(you\s+are\s+)?(a|an|the)\b|"
            r"from\s+now\s+on\s+(you|your)\b|"
            r"switch\s+to\s+\w+\s+mode"
            r")\b",
        ),
        3,
    ),
    # ── Multi-language instruction override ───────────────────────────────
    (
        "instruction_override_multilang",
        re.compile(
            r"(?i)("
            # Spanish
            r"ignora\s+(todas?\s+)?(las\s+)?instrucciones\s+(anteriores|previas)|"
            # French
            r"ignore[rz]?\s+(toutes?\s+)?(les\s+)?instructions?\s+(pr[eé]c[eé]dentes?|ant[eé]rieures?)|"
            # German
            r"ignoriere?\s+(alle\s+)?(vorherigen?\s+)?Anweisungen|"
            # Chinese
            r"忽略(所有)?(之前|以上|上面)(的)?(指令|指示|说明|提示)|"
            # Japanese
            r"(以前|上記|これまで)の(指示|命令|プロンプト)を(無視|忘れ)|"
            # Arabic
            r"تجاهل\s+(جميع\s+)?التعليمات\s+السابقة|"
            # Korean
            r"(이전|위의)\s*(지시|명령|지침)(을|를)\s*(무시|잊)"
            r")",
        ),
        3,
    ),
    # ── Role delimiter injection ──────────────────────────────────────────
    (
        "role_delimiter",
        re.compile(
            r"(?i)("
            r"<\|?(?:system|assistant|user|developer|im_start|im_end)\|?>|"
            r"\[?\s*(?:SYSTEM|ASSISTANT|USER|DEVELOPER)\s*\]?\s*:|"
            r"^###\s*(?:system|assistant|user|developer)\s*$|"
            r"\bEND_OF_(?:SYSTEM|USER)_(?:PROMPT|MESSAGE)\b|"
            r"---\s*(?:SYSTEM|ASSISTANT|USER)\s*---"
            r")",
            re.MULTILINE,
        ),
        2,
    ),
    # ── Prompt leak / exfiltration request ────────────────────────────────
    (
        "prompt_leak_request",
        re.compile(
            r"(?i)\b("
            r"repeat\s+(your|the)\s+(system\s+)?prompt|"
            r"print\s+(your\s+)?(system\s+)?prompt|"
            r"output\s+(your\s+)?(entire\s+)?(system\s+)?prompt|"
            r"show\s+(me\s+)?(your\s+)?(initial|original|hidden)\s+(instructions?|prompt)|"
            r"what\s+(are|were)\s+your\s+(original\s+)?instructions|"
            r"reveal\s+(your\s+)?(system|secret|hidden)\s+(prompt|instructions?)|"
            r"display\s+(the\s+)?(developer|system)\s+message"
            r")\b",
        ),
        3,
    ),
    # ── Data exfiltration patterns ────────────────────────────────────────
    (
        "data_exfiltration",
        re.compile(
            r"(?i)("
            r"append\s+(all|the|every)\s+.{0,30}(context|retriev|chunk|document).{0,30}(to|into|at)\s+(https?://|the\s+url)|"
            r"send\s+(all|the|every)\s+.{0,30}(to|via)\s+https?://|"
            r"include\s+(all|every)\s+retriev.{0,20}in\s+(your|the)\s+(response|output|reply)|"
            r"(fetch|load|request|visit|navigate)\s+(https?://\S+)|"
            r"!\[.*?\]\(https?://[^)]*\?\w+=|"  # markdown image with query params (exfil via rendering)
            r"<img\s[^>]*src\s*=\s*[\"']https?://[^\"']*\?"  # html img exfil
            r")",
        ),
        3,
    ),
    # ── Fake tool / API / function call injection ─────────────────────────
    (
        "fake_tool_or_api",
        re.compile(
            r"(?i)("
            r"function\s*call\s*:|"
            r"```\s*json\s*\{[^}]*\"role\"\s*:\s*\"system\"|"
            r"\btool_call\s*\(|"
            r"</?tool_call>|"
            r"</?function_call>|"
            r"\{\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:"
            r")",
        ),
        2,
    ),
    # ── Encoding / obfuscation tricks ─────────────────────────────────────
    (
        "encoding_trick",
        re.compile(
            r"(?i)("
            r"base64[\s_-]*(decode|encode|decrypt)|"
            r"rot[\s_-]?13|"
            r"from[\s_-]hex|"
            r"unicode[\s_-]escape|"
            r"url[\s_-]?decode|"
            r"char\s*code|"
            r"ascii[\s_-](art|encode)|"
            r"decode\s+this\s*:|"
            r"the\s+following\s+is\s+(encoded|base64|hex)"
            r")",
        ),
        1,
    ),
    # ── Invisible / zero-width / Unicode tricks ───────────────────────────
    (
        "unicode_manipulation",
        re.compile(
            r"("
            r"[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff]{2,}|"  # clusters of ZW chars
            r"[\u202a\u202b\u202c\u202d\u202e]|"  # bidi overrides
            r"[\u2066\u2067\u2068\u2069]|"  # bidi isolates
            r"[\U000e0001-\U000e007f]"  # tags block (invisible language tags)
            r")",
        ),
        2,
    ),
    # ── HTML / script injection in documents ──────────────────────────────
    (
        "html_script_injection",
        re.compile(
            r"(?i)("
            r"<script[\s>]|"
            r"<iframe[\s>]|"
            r"javascript\s*:|"
            r"on(load|error|click|mouseover)\s*=|"
            r"<object[\s>].*data\s*=|"
            r"<!--\s*(ignore|system|instructions?|override)"
            r")",
        ),
        2,
    ),
    # ── Markdown injection ────────────────────────────────────────────────
    (
        "markdown_injection",
        re.compile(
            r"(?i)("
            r"!\[.*?\]\([^)]*\b(onerror|onload|javascript)\b|"  # image with event handler
            r"\[.*?\]\(data:|"  # data URI link
            r"<!--\s*(system|ignore|override|prompt)"
            r")",
        ),
        2,
    ),
    # ── Indirect / deferred injection ─────────────────────────────────────
    (
        "indirect_injection",
        re.compile(
            r"(?i)\b("
            r"when\s+(the\s+)?(user|human|person)\s+(asks?|quer|request|mention|say|type)\s+.{0,40}(instead|reply|respond|answer|say|tell)\b|"
            r"if\s+(anyone|someone|the\s+user)\s+(asks?|quer).{0,40}(respond|reply|say|tell|answer)\b|"
            r"whenever\s+(this|the)\s+(text|chunk|document|passage)\s+is\s+(retriev|return|shown|used)|"
            r"upon\s+retrieval\s+(of\s+this|,)\s*(respond|reply|say|output)"
            r")\b",
        ),
        3,
    ),
    # ── Payload splitting hints ───────────────────────────────────────────
    (
        "payload_splitting",
        re.compile(
            r"(?i)\b("
            r"(this|the)\s+is\s+part\s+\d+\s+of\s+\d+|"
            r"combine\s+(this|these)\s+(with|and)\s+(the\s+)?(next|other|previous)\s+(chunk|part|segment)|"
            r"continued\s+(from|in)\s+(the\s+)?(previous|next)\s+(chunk|part|section)|"
            r"assemble\s+(the\s+)?(full\s+)?(instruction|payload|command)"
            r")\b",
        ),
        2,
    ),
]


@dataclass(frozen=True)
class DocumentScanResult:
    """Outcome of scanning a single document or chunk."""

    document_id: str
    severity: InjectionSeverity
    score: float  # 0.0–1.0 aggregate heuristic confidence
    matched_rules: list[str] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)
    detector: str = "heuristic"
    rationale: str | None = None

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
