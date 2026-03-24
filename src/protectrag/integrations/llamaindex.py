"""LlamaIndex integration: NodePostprocessor that screens retrieved nodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from protectrag.scanner import DocumentScanResult, InjectionSeverity, scan_document_for_injection

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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._scan_fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))
        self.block_on = block_on
        self.remove_blocked = remove_blocked

    def _postprocess_nodes(
        self,
        nodes: list[Any],
        query_bundle: Any | None = None,
    ) -> list[Any]:
        out: list[Any] = []
        for nws in nodes:
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
                continue
            out.append(nws)
        return out
