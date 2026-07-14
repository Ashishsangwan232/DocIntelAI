"""schemas/search.py — request/response models for standalone semantic search."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.services.search_service import SearchHit


class SearchHitResponse(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    page_number: int | None = None
    similarity_score: float
    snippet: str
    full_content: str

    @classmethod
    def from_hit(cls, hit: SearchHit) -> "SearchHitResponse":
        return cls(
            document_id=hit.document_id,
            filename=hit.filename,
            chunk_id=hit.chunk_id,
            chunk_index=hit.chunk_index,
            page_number=hit.page_number,
            similarity_score=hit.similarity_score,
            snippet=hit.snippet,
            full_content=hit.full_content,
        )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHitResponse]
