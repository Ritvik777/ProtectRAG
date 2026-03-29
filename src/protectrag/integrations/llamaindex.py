"""LlamaIndex integration: NodePostprocessor that screens retrieved nodes."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from protectrag.scanner import InjectionSeverity, scan_document_for_injection

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
except ImportError:  # pragma: no cover
    BaseNodePostprocessor = object  # type: ignore[assignment,misc]
    NodeWithScore = None  # type: ignore[assignment,misc]
    QueryBundle = None  # type: ignore[assignment,misc]

ScanFn = Callable[[str, str], DocumentScanResult]


class ProtectRAGPostprocessor(BaseNodePostprocessor):  # type: ignore[misc]
    """
    LlamaIndex NodePostprocessor: filters or tags nodes that contain prompt injection.

    Usage::

        from llama_index.core import VectorStoreIndex
        from protectrag.integrations.llamaindex import ProtectRAGPostprocessor

        index = VectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine(
            node_postprocessors=[ProtectRAGPostprocessor()],
        )
    """

    block_on: InjectionSeverity = InjectionSeverity.MEDIUM
    remove_blocked: bool = True

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        *,
        scan_fn: ScanFn | None = None,
        block_on: InjectionSeverity = InjectionSeverity.MEDIUM,
        remove_blocked: bool = True,
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._scan_fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))
        self.block_on = block_on
        self.remove_blocked = remove_blocked
        self._max_workers = max_workers

    def _process_one(self, nws: Any) -> Any | None:
        node = nws.node
        text = node.get_content()
        node_id = node.node_id or "unknown"
        r = self._scan_fn(text, str(node_id))
        node.metadata["protectrag_severity"] = r.severity.name
        node.metadata["protectrag_score"] = r.score
        node.metadata["protectrag_detector"] = r.detector
        if r.rationale:
            node.metadata["protectrag_rationale"] = r.rationale
        if r.severity >= self.block_on and self.remove_blocked:
            return None
        return nws

    def _postprocess_nodes(
        self,
        nodes: list[Any],
        query_bundle: Any | None = None,
    ) -> list[Any]:
        n = len(nodes)
        if n == 0:
            return []

        workers = self._max_workers
        if workers is None:
            workers = min(8, n, min(32, (os.cpu_count() or 1) + 4))
        if workers <= 1:
            out: list[Any] = []
            for nws in nodes:
                kept = self._process_one(nws)
                if kept is not None:
                    out.append(kept)
            return out

        w = max(1, min(workers, n))
        with ThreadPoolExecutor(max_workers=w) as pool:
            results = list(pool.map(self._process_one, nodes))
        return [x for x in results if x is not None]
