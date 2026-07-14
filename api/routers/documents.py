"""
routers/documents.py
====================
REST endpoints for document upload, listing, retrieval, preview, and
deletion — a thin wrapper around `DocumentService`, which already does
100% of the real work (validation, extraction, chunking, embedding,
persistence). Nothing here talks to SQLite, ChromaDB, or the
filesystem directly.

Upload accepts one-or-more files in a single request and always
responds 200, with a per-file success/failure result in the body
(`UploadBatchResponse`) — this mirrors `src/ui/upload.py`'s
`_process_uploaded_files`/`UploadOutcome` pattern exactly, so a
partial failure (e.g. one duplicate among five files) doesn't roll
back the four that succeeded. The alternative — one HTTP request per
file — would be more RESTful in isolation but would force the
frontend to reimplement the "some succeeded, some didn't" batching
logic that already exists and is already tested; matching the
Streamlit behavior means one fewer thing to re-verify during
migration.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from api.dependencies import get_document_service
from api.schemas.documents import (
    DocumentPreviewResponse,
    DocumentResponse,
    UploadBatchResponse,
    UploadItemResult,
)
from src.database.models import DocumentStatus
from src.services.document_service import DocumentService
from src.utils.exceptions import DocIntelError
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _to_response(document_service: DocumentService, document) -> DocumentResponse:
    chunk_count = (
        document_service.get_chunk_count(document.id)
        if document.status == DocumentStatus.READY
        else None
    )
    return DocumentResponse.from_document(document, chunk_count=chunk_count)


@router.post("/upload", response_model=UploadBatchResponse)
async def upload_documents(
    files: list[UploadFile] = File(..., description="One or more PDF/DOCX/TXT/MD files."),
    chunk_size: int | None = Query(default=None, ge=100, le=8000),
    chunk_overlap: int | None = Query(default=None, ge=0, le=2000),
    document_service: DocumentService = Depends(get_document_service),
) -> UploadBatchResponse:
    results: list[UploadItemResult] = []

    for upload in files:
        filename = upload.filename or "unnamed"
        content = await upload.read()
        try:
            document = document_service.process_upload(
                filename, content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            results.append(
                UploadItemResult(
                    filename=filename,
                    success=True,
                    message="Processed successfully",
                    document=_to_response(document_service, document),
                )
            )
        except DocIntelError as exc:
            # Matches _attempt_process in src/ui/upload.py: a single
            # file's failure is reported inline, not raised — the rest
            # of the batch still gets processed.
            logger.info("Upload failed for '%s': %s", filename, exc)
            results.append(UploadItemResult(filename=filename, success=False, message=str(exc)))

    return UploadBatchResponse(results=results)


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    status_filter: DocumentStatus | None = Query(default=None, alias="status"),
    collection_id: str | None = Query(default=None),
    document_service: DocumentService = Depends(get_document_service),
) -> list[DocumentResponse]:
    documents = document_service.list_documents(collection_id=collection_id, status=status_filter)
    return [_to_response(document_service, doc) for doc in documents]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str, document_service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    document = document_service.get_document(document_id)
    return _to_response(document_service, document)


@router.get("/{document_id}/preview", response_model=DocumentPreviewResponse)
def preview_document(
    document_id: str,
    max_chars: int = Query(default=2000, ge=100, le=20000),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentPreviewResponse:
    # get_document first so a bad ID surfaces as 404 (RecordNotFoundError)
    # rather than get_preview_text's "(No preview available...)" filler text.
    document_service.get_document(document_id)
    preview_text = document_service.get_preview_text(document_id, max_chars=max_chars)
    return DocumentPreviewResponse(document_id=document_id, preview_text=preview_text)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str, document_service: DocumentService = Depends(get_document_service)
) -> None:
    document_service.delete_document(document_id)
