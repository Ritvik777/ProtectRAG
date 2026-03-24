"""LangChain integration: DocumentTransformer that filters injected chunks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from protectrag.scanner import DocumentScanResult, InjectionSeverity, scan_document_for_injection

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    Document = None  # type: ignore[assignment,misc]

ScanFn = Callable[[str, str], DocumentScanResult]


class ProtectRAGFilter:
    """
    LangChain-compatible document transformer.

    Removes or tags documents whose text triggers prompt-injection detection.

    Usage with a LangChain retriever::

        from langchain.retrievers import ... as MyRetriever
        from protectrag.integrations.langchain import ProtectRAGFilter

        retriever = MyRetriever(...)
        guard = ProtectRAGFilter()
        docs = retriever.get_relevant_documents(query)
        safe_docs = guard.transform_documents(docs)
    """

    def __init__(
        self,
        *,
        scan_fn: ScanFn | None = None,
        block_on: InjectionSeverity = InjectionSeverity.MEDIUM,
        tag_key: str = "protectrag_severity",
        remove_blocked: bool = True,
    ) -> None:
        self.scan_fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))
        self.block_on = block_on
        self.tag_key = tag_key
        self.remove_blocked = remove_blocked

    def transform_documents(
        self,
        documents: Sequence[Any],
        **kwargs: Any,
    ) -> list[Any]:
        if Document is None:
            raise ImportError("langchain-core is required: pip install langchain-core")
        out: list[Any] = []
        for doc in documents:
            doc_id = doc.metadata.get("id", doc.metadata.get("source", "unknown"))
            r = self.scan_fn(doc.page_content, str(doc_id))
            doc.metadata[self.tag_key] = r.severity.name
            doc.metadata["protectrag_score"] = r.score
            doc.metadata["protectrag_detector"] = r.detector
            if r.rationale:
                doc.metadata["protectrag_rationale"] = r.rationale
            if r.severity >= self.block_on and self.remove_blocked:
                continue
            out.append(doc)
        return out
