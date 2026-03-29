"""FastAPI integration: dependency + middleware for screening uploads and request bodies."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from protectrag.async_api import AsyncScanFn
from protectrag.ingest import IngestResult, ingest_document, ingest_document_async
from protectrag.scanner import DocumentScanResult, InjectionSeverity, scan_document_for_injection

try:
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover
    HTTPException = Exception  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]

ScanFn = Callable[[str, str], DocumentScanResult]


def screen_text_dependency(
    *,
    scan_fn: ScanFn | None = None,
    block_on: InjectionSeverity = InjectionSeverity.HIGH,
    warn_on: InjectionSeverity = InjectionSeverity.MEDIUM,
) -> Callable[..., Any]:
    """
    Returns a FastAPI dependency function that screens text.

    Usage::

        from fastapi import FastAPI, Depends
        from protectrag.integrations.fastapi import screen_text_dependency

        app = FastAPI()
        screen = screen_text_dependency()

        @app.post("/ingest")
        def ingest(text: str, doc_id: str, result: IngestResult = Depends(screen)):
            if result.decision == "block":
                raise HTTPException(400, detail=result.message)
            ...
    """
    fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))

    def _dep(text: str, doc_id: str = "unknown") -> IngestResult:
        return ingest_document(
            text,
            document_id=doc_id,
            block_on=block_on,
            warn_on=warn_on,
            scan=fn,
        )

    return _dep


def screen_text_dependency_async(
    *,
    scan_fn: ScanFn | None = None,
    async_scan_fn: AsyncScanFn | None = None,
    block_on: InjectionSeverity = InjectionSeverity.HIGH,
    warn_on: InjectionSeverity = InjectionSeverity.MEDIUM,
) -> Callable[..., Any]:
    """
    Async FastAPI dependency: use with ``async_scan_fn`` for :meth:`~protectrag.llm.HybridScanner.ascan`
    or ``scan_fn`` for sync scans (executed in a thread pool inside :func:`~protectrag.ingest.ingest_document_async`).
    """
    if scan_fn is not None and async_scan_fn is not None:
        raise ValueError("Pass at most one of scan_fn and async_scan_fn")
    fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))

    async def _dep(text: str, doc_id: str = "unknown") -> IngestResult:
        return await ingest_document_async(
            text,
            document_id=doc_id,
            block_on=block_on,
            warn_on=warn_on,
            scan=fn if async_scan_fn is None else None,
            async_scan=async_scan_fn,
        )

    return _dep


def create_screening_middleware(
    *,
    scan_fn: ScanFn | None = None,
    block_on: InjectionSeverity = InjectionSeverity.HIGH,
    paths: list[str] | None = None,
    text_field: str = "text",
    doc_id_field: str = "document_id",
) -> Any:
    """
    Returns an ASGI middleware class that screens JSON request bodies on specified paths.

    Blocks the request with 422 if injection severity >= ``block_on``.

    Usage::

        from fastapi import FastAPI
        from protectrag.integrations.fastapi import create_screening_middleware

        app = FastAPI()
        app.add_middleware(create_screening_middleware(paths=["/api/ingest"]))
    """
    fn = scan_fn or (lambda t, d: scan_document_for_injection(t, document_id=d))
    target_paths = set(paths or [])

    from starlette.middleware.base import BaseHTTPMiddleware

    class ProtectRAGMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Any, call_next: Any) -> Any:
            if target_paths and request.url.path not in target_paths:
                return await call_next(request)
            if request.method not in ("POST", "PUT", "PATCH"):
                return await call_next(request)
            try:
                body = await request.json()
            except Exception:
                return await call_next(request)
            text = body.get(text_field)
            if not isinstance(text, str) or not text.strip():
                return await call_next(request)
            doc_id = str(body.get(doc_id_field, "unknown"))
            r = await asyncio.to_thread(fn, text, doc_id)
            if r.severity >= block_on:
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": "Content blocked by ProtectRAG",
                        "severity": r.severity.name,
                        "score": r.score,
                        "matched_rules": r.matched_rules,
                    },
                )
            return await call_next(request)

    return ProtectRAGMiddleware
