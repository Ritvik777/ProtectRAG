"""LangChain integration: DocumentTransformer that filters injected chunks."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from protectrag.scanner import InjectionSeverity, scan_document_for_injection

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
        max_workers: int | None = None,
    ) -> None:
        self.scan_fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))
        self.block_on = block_on
        self.tag_key = tag_key
        self.remove_blocked = remove_blocked
        self._max_workers = max_workers

    def _process_one(self, doc: Any) -> Any | None:
        if Document is None:
            raise ImportError("langchain-core is required: pip install langchain-core")
        doc_id = doc.metadata.get("id", doc.metadata.get("source", "unknown"))
        r = self.scan_fn(doc.page_content, str(doc_id))
        doc.metadata[self.tag_key] = r.severity.name
        doc.metadata["protectrag_score"] = r.score
        doc.metadata["protectrag_detector"] = r.detector
        if r.rationale:
            doc.metadata["protectrag_rationale"] = r.rationale
        if r.severity >= self.block_on and self.remove_blocked:
            return None
        return doc

    def transform_documents(
        self,
        documents: Sequence[Any],
        **kwargs: Any,
    ) -> list[Any]:
        if Document is None:
            raise ImportError("langchain-core is required: pip install langchain-core")
        docs_list = list(documents)
        n = len(docs_list)
        if n == 0:
            return []

        workers = self._max_workers
        if workers is None:
            workers = min(8, n, min(32, (os.cpu_count() or 1) + 4))
        if workers <= 1:
            out: list[Any] = []
            for doc in docs_list:
                kept = self._process_one(doc)
                if kept is not None:
                    out.append(kept)
            return out

        w = max(1, min(workers, n))
        with ThreadPoolExecutor(max_workers=w) as pool:
            results = list(pool.map(self._process_one, docs_list))
        return [x for x in results if x is not None]
