"""
routers/search.py
==================
Standalone semantic search — retrieval only, no LLM call, no
persistence beyond the same query-analytics logging Chat also does.
Thin wrapper around `SearchService`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_search_service
from api.schemas.search import SearchHitResponse, SearchRequest, SearchResponse
from src.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    body: SearchRequest, search_service: SearchService = Depends(get_search_service)
) -> SearchResponse:
    hits = search_service.search(body.query, top_k=body.top_k, document_ids=body.document_ids)
    return SearchResponse(hits=[SearchHitResponse.from_hit(h) for h in hits])
