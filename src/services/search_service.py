"""
search_service.py
==================
Standalone semantic search — retrieval only, no LLM call. This is
what powers the "Semantic Search" feature (distinct from Chat): users
see ranked chunks with similarity scores and highlighted matching
terms, without waiting on or paying for an LLM generation.

Reuses `Retriever` directly (the same one `RAGPipeline` uses for chat)
rather than duplicating retrieval logic — the only thing this service
adds on top is highlighting and query analytics logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.database.models import QueryLog
from src.database.sqlite_manager import SQLiteManager
from src.rag.retriever import Retriever
from src.utils.exceptions import ValidationError
from src.utils.helpers import truncate_text
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Characters of context shown around the first matched term in a
# snippet — enough to read the surrounding sentence, short enough to
# stay scannable in a results list.
_SNIPPET_CONTEXT_CHARS = 220

# Query tokens shorter than this are treated as noise (stopword-like)
# and excluded from highlighting, so highlighting "a document about
# fees" doesn't bold every stray "a" in the results.
_MIN_HIGHLIGHT_TOKEN_LENGTH = 2


@dataclass
class SearchHit:
    """A single ranked semantic search result, ready for display."""

    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    page_number: int | None
    similarity_score: float
    snippet: str  # markdown, with matching terms wrapped in **bold**
    full_content: str


class SearchService:
    """Standalone semantic search over the document library — no LLM involved."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        db: SQLiteManager | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.db = db or SQLiteManager()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        """
        Run a semantic search and return ranked, highlighted results.

        Args:
            query: The user's search text.
            top_k: Number of results to return.
            document_ids: Optional filter to scope search to specific documents.

        Returns:
            Ranked `SearchHit` list (may be empty — no matches above
            the similarity threshold is a normal, non-error outcome).

        Raises:
            ValidationError: If `query` is empty.
            RetrievalError: If query embedding or vector search fails outright.
        """
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty.")

        retrieval = self.retriever.retrieve(query, top_k=top_k, document_ids=document_ids)
        self._log_query_analytics(query, retrieval.chunks)

        hits = [
            SearchHit(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                similarity_score=chunk.similarity_score,
                snippet=_build_snippet(chunk.content, query),
                full_content=chunk.content,
            )
            for chunk in retrieval.chunks
        ]

        logger.info("Semantic search for '%s' returned %d hits", query[:80], len(hits))
        return hits

    def _log_query_analytics(self, query: str, chunks: list) -> None:
        top_chunk = chunks[0] if chunks else None
        self.db.log_query(
            QueryLog(
                query_text=query,
                document_id=top_chunk.document_id if top_chunk else None,
                top_score=top_chunk.similarity_score if top_chunk else None,
            )
        )


# ------------------------------------------------------------------
# Highlighting
# ------------------------------------------------------------------
def _tokenize_query(query: str) -> list[str]:
    """Extract meaningful word tokens from a query for highlighting."""
    words = re.findall(r"\w+", query.lower())
    return sorted({w for w in words if len(w) >= _MIN_HIGHLIGHT_TOKEN_LENGTH}, key=len, reverse=True)


def _build_snippet(content: str, query: str, context_chars: int = _SNIPPET_CONTEXT_CHARS) -> str:
    """
    Build a markdown snippet centered on the first matching query term,
    with matching terms wrapped in `**bold**`. Falls back to a plain
    truncated preview if no query terms are found verbatim in the
    content — expected for semantic (non-lexical) matches, where the
    chunk can be relevant without sharing exact words with the query.
    """
    tokens = _tokenize_query(query)
    if not tokens:
        return truncate_text(content, max_length=context_chars)

    pattern = re.compile("(" + "|".join(re.escape(t) for t in tokens) + ")", re.IGNORECASE)
    match = pattern.search(content)

    if match is None:
        return truncate_text(content, max_length=context_chars)

    half_window = context_chars // 2
    start = max(0, match.start() - half_window)
    end = min(len(content), match.end() + half_window)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    window = content[start:end]

    highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", window)
    return f"{prefix}{highlighted}{suffix}"
