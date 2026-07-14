"""
routers/export.py
==================
Downloadable exports of a chat conversation — mounted under
`/chat/sessions` (a distinct `APIRouter` instance from
`routers/chat.py`, no route overlap: `/{id}/export` is a new path).

`ExportService` itself already raises `RecordNotFoundError` for a bad
session ID and `ValidationError` for a session with no messages yet —
both handled by the same global error mapping from Phase A, so no
extra validation is needed here.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from api.dependencies import get_export_service
from src.services.export_service import ExportService

router = APIRouter(prefix="/chat/sessions", tags=["export"])

_MEDIA_TYPES: dict[str, str] = {
    "markdown": "text/markdown",
    "text": "text/plain",
    "pdf": "application/pdf",
}
_EXTENSIONS: dict[str, str] = {"markdown": "md", "text": "txt", "pdf": "pdf"}


@router.get("/{session_id}/export")
def export_conversation(
    session_id: str,
    format: Literal["markdown", "text", "pdf"] = Query(default="markdown"),
    export_service: ExportService = Depends(get_export_service),
) -> Response:
    if format == "markdown":
        content: str | bytes = export_service.export_to_markdown(session_id)
    elif format == "text":
        content = export_service.export_to_text(session_id)
    else:
        content = export_service.export_to_pdf(session_id)

    body = content.encode("utf-8") if isinstance(content, str) else content
    filename = f"conversation_{session_id}.{_EXTENSIONS[format]}"

    return Response(
        content=body,
        media_type=_MEDIA_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
