"""
routers/summary.py
===================
AI document summarization — a document sub-resource, mounted under the
same `/documents` path as `routers/documents.py` (a distinct
`APIRouter` instance, no route overlap: `/{id}/summary` vs. that
router's `/{id}` and `/{id}/preview`).

`POST`, not `GET`: nothing is cached or persisted here — each call
re-runs the LLM and returns a fresh summary, exactly like clicking
"Generate Summary" again in the Streamlit library does. That's a
non-idempotent, cost-incurring action, which `POST` communicates
honestly; a `GET` would imply it's free/cacheable, which it isn't.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_summary_service
from api.schemas.summary import DocumentSummaryResponse
from src.services.summary_service import SummaryService

router = APIRouter(prefix="/documents", tags=["summary"])


@router.post("/{document_id}/summary", response_model=DocumentSummaryResponse)
def generate_summary(
    document_id: str, summary_service: SummaryService = Depends(get_summary_service)
) -> DocumentSummaryResponse:
    summary = summary_service.generate_summary(document_id)
    return DocumentSummaryResponse.from_summary(summary)
