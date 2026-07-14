"""
errors.py
=========
Maps this project's `DocIntelError` hierarchy (src/utils/exceptions.py)
onto HTTP responses, so every router in every future phase can just
`raise` the same exceptions the Streamlit UI already catches — no
per-route try/except, no per-route knowledge of status codes.

Design
------
`_STATUS_MAP` is an ordered list, most-specific exception first. A
single `isinstance()` walk picks the first (and therefore most
specific) match, which sidesteps the ambiguity of Starlette resolving
multiple registered `@app.exception_handler(...)` classes against a
type hierarchy — there is exactly one handler for the whole
`DocIntelError` tree, and this module decides the specificity order
explicitly instead of relying on registration order or MRO lookup.

Every response body has the same shape (`ErrorResponse` in
`schemas/common.py`): `{"error": {"type": ..., "message": ...}}`. The
frontend never needs to special-case which endpoint failed to parse
an error.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.utils.exceptions import (
    CorruptedFileError,
    DatabaseError,
    DocIntelError,
    DocumentProcessingError,
    DuplicateDocumentError,
    EmbeddingError,
    FileTooLargeError,
    LLMAuthenticationError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    RecordNotFoundError,
    RetrievalError,
    SummaryGenerationError,
    UnsupportedFileTypeError,
    ValidationError,
    VectorStoreError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Ordered most-specific -> least-specific. `DocIntelError` is the
# catch-all fallback at the end.
_STATUS_MAP: list[tuple[type[DocIntelError], int]] = [
    (RecordNotFoundError, 404),
    (DuplicateDocumentError, 409),
    (UnsupportedFileTypeError, 400),
    (FileTooLargeError, 400),
    (ValidationError, 400),
    (LLMTimeoutError, 504),
    (LLMAuthenticationError, 502),
    (LLMResponseError, 502),
    (LLMError, 502),
    (CorruptedFileError, 422),
    (DocumentProcessingError, 422),
    (EmbeddingError, 500),
    (VectorStoreError, 500),
    (RetrievalError, 500),
    (SummaryGenerationError, 500),
    (DatabaseError, 500),
    (DocIntelError, 500),
]


def status_code_for(exc: DocIntelError) -> int:
    """Resolve the HTTP status code for a given `DocIntelError` instance."""
    for exc_type, code in _STATUS_MAP:
        if isinstance(exc, exc_type):
            return code
    return 500  # pragma: no cover - unreachable, DocIntelError always matches


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the two exception handlers that cover every route in the app."""

    @app.exception_handler(DocIntelError)
    async def handle_docintel_error(request: Request, exc: DocIntelError) -> JSONResponse:
        status_code = status_code_for(exc)
        if status_code >= 500:
            logger.error("%s %s -> %s: %s", request.method, request.url.path, status_code, exc, exc_info=True)
        else:
            logger.info("%s %s -> %s: %s", request.method, request.url.path, status_code, exc)
        return JSONResponse(
            status_code=status_code,
            content={"error": {"type": type(exc).__name__, "message": str(exc)}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # Anything that isn't a DocIntelError is a bug, not an expected
        # failure mode — log the full traceback but never leak it to
        # the client, matching the Streamlit UI's "clean st.error,
        # never a raw traceback" rule from Phase 13.
        logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "InternalServerError", "message": "An unexpected error occurred."}},
        )
