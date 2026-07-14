"""schemas/summary.py — request/response models for document AI summaries."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.services.summary_service import DocumentSummary


class DocumentSummaryResponse(BaseModel):
    document_id: str
    filename: str
    executive_summary: str
    key_insights: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    source_chunk_count: int
    used_map_reduce: bool
    generated_at: datetime

    @classmethod
    def from_summary(cls, summary: DocumentSummary) -> "DocumentSummaryResponse":
        return cls(
            document_id=summary.document_id,
            filename=summary.filename,
            executive_summary=summary.executive_summary,
            key_insights=summary.key_insights,
            topics=summary.topics,
            source_chunk_count=summary.source_chunk_count,
            used_map_reduce=summary.used_map_reduce,
            generated_at=summary.generated_at,
        )
