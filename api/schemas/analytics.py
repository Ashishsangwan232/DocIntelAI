"""schemas/analytics.py — request/response models for the Analytics dashboard."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.services.analytics_service import AnalyticsSummary


class MostQueriedDocumentResponse(BaseModel):
    document_id: str
    filename: str
    query_count: int


class AnalyticsSummaryResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_embeddings: int
    total_chat_sessions: int
    total_queries: int
    average_response_time_ms: float | None = None
    total_storage_bytes: int
    # Pre-formatted, human-readable strings (e.g. "12.4 MB", "2.3s") —
    # computed once here rather than re-implemented in the frontend,
    # matching what the Streamlit dashboard already displays.
    total_storage_display: str
    average_response_time_display: str
    most_queried_documents: list[MostQueriedDocumentResponse] = Field(default_factory=list)
    uploads_per_day: list[tuple[str, int]] = Field(default_factory=list)

    @classmethod
    def from_summary(cls, summary: AnalyticsSummary) -> "AnalyticsSummaryResponse":
        return cls(
            total_documents=summary.total_documents,
            total_chunks=summary.total_chunks,
            total_embeddings=summary.total_embeddings,
            total_chat_sessions=summary.total_chat_sessions,
            total_queries=summary.total_queries,
            average_response_time_ms=summary.average_response_time_ms,
            total_storage_bytes=summary.total_storage_bytes,
            total_storage_display=summary.total_storage_display,
            average_response_time_display=summary.average_response_time_display,
            most_queried_documents=[
                MostQueriedDocumentResponse(
                    document_id=d.document_id, filename=d.filename, query_count=d.query_count
                )
                for d in summary.most_queried_documents
            ],
            uploads_per_day=summary.uploads_per_day,
        )
