"""
schemas/documents.py
====================
Request/response models for the Documents API. These mirror
`src.database.models.Document` field-for-field rather than inventing a
parallel shape — the frontend should see exactly the same data the
Streamlit library table shows today.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.database.models import Document, DocumentStatus, FileType


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: FileType
    file_size_bytes: int
    status: DocumentStatus
    collection_id: str | None = None
    page_count: int | None = None
    uploaded_at: datetime
    processed_at: datetime | None = None
    chunk_count: int | None = Field(
        default=None,
        description="Only populated for READY documents — matches what the "
        "Streamlit library table computes per row.",
    )

    @classmethod
    def from_document(cls, document: Document, chunk_count: int | None = None) -> "DocumentResponse":
        return cls(
            id=document.id,
            filename=document.filename,
            file_type=document.file_type,
            file_size_bytes=document.file_size_bytes,
            status=document.status,
            collection_id=document.collection_id,
            page_count=document.page_count,
            uploaded_at=document.uploaded_at,
            processed_at=document.processed_at,
            chunk_count=chunk_count,
        )


class UploadItemResult(BaseModel):
    """
    Outcome of processing a single file within a (possibly multi-file)
    upload request — deliberately mirrors `src.ui.upload.UploadOutcome`
    so both surfaces report upload results the same way: a batch upload
    always returns 200, with per-file success/failure inside the body,
    rather than failing the whole batch because one file was a
    duplicate or unsupported.
    """

    filename: str
    success: bool
    message: str
    document: DocumentResponse | None = None


class UploadBatchResponse(BaseModel):
    results: list[UploadItemResult]


class DocumentPreviewResponse(BaseModel):
    document_id: str
    preview_text: str
