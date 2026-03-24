"""Callback hooks for ingest and retrieval decisions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from protectrag.scanner import DocumentScanResult

_logger = logging.getLogger("protectrag.callbacks")

OnBlockFn = Callable[[str, DocumentScanResult], Any]
OnWarnFn = Callable[[str, DocumentScanResult], Any]
OnAllowFn = Callable[[str, DocumentScanResult], Any]


@dataclass
class CallbackRegistry:
    """
    Register functions to be called on ingest / retrieval decisions.

    Typical use: send alerts (Slack, PagerDuty), write to a quarantine queue,
    or trigger a human review workflow.
    """

    on_block: list[OnBlockFn] = field(default_factory=list)
    on_warn: list[OnWarnFn] = field(default_factory=list)
    on_allow: list[OnAllowFn] = field(default_factory=list)

    def fire_block(self, text: str, result: DocumentScanResult) -> None:
        for fn in self.on_block:
            try:
                fn(text, result)
            except Exception:
                _logger.exception("on_block callback failed for doc=%s", result.document_id)

    def fire_warn(self, text: str, result: DocumentScanResult) -> None:
        for fn in self.on_warn:
            try:
                fn(text, result)
            except Exception:
                _logger.exception("on_warn callback failed for doc=%s", result.document_id)

    def fire_allow(self, text: str, result: DocumentScanResult) -> None:
        for fn in self.on_allow:
            try:
                fn(text, result)
            except Exception:
                _logger.exception("on_allow callback failed for doc=%s", result.document_id)
