"""Shared LLM result cache (fakeredis)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from protectrag.llm import LLMScanConfig, LLMScanner
from protectrag.llm_cache import (
    RedisLLMClassificationCache,
    document_scan_result_from_dict,
    document_scan_result_to_dict,
)
from protectrag.scanner import DocumentScanResult, InjectionSeverity


def test_serde_roundtrip() -> None:
    r = DocumentScanResult(
        document_id="d",
        severity=InjectionSeverity.HIGH,
        score=0.9,
        matched_rules=["x"],
        snippets=["y"],
        detector="llm",
        rationale="z",
    )
    r2 = document_scan_result_from_dict(document_scan_result_to_dict(r))
    assert r2.severity == r.severity
    assert r2.score == r.score
    assert r2.matched_rules == r.matched_rules


def test_redis_cache_shared_between_scanners() -> None:
    fakeredis = pytest.importorskip("fakeredis")

    r = fakeredis.FakeRedis()
    cache = RedisLLMClassificationCache(r, ttl_seconds=3600, key_prefix="t:")
    cfg = LLMScanConfig(api_key="k", cache_max_entries=0)
    raw = '{"severity":"none","confidence":0.99,"brief":"ok"}'

    s1 = LLMScanner(cfg, shared_cache=cache)
    s2 = LLMScanner(cfg, shared_cache=cache)
    with patch.object(LLMScanner, "_post_chat_completions", return_value=raw):
        s1.scan("identical body", document_id="a")
    with patch.object(LLMScanner, "_post_chat_completions") as post:
        s2.scan("identical body", document_id="b")
    post.assert_not_called()
    s1.close()
    s2.close()
