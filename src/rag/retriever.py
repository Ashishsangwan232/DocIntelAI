"""
retriever.py
============
Semantic retrieval: embeds a user query and fetches the most similar
document chunks from ChromaDB.

Deliberately does nothing besides retrieval — no prompt construction,
no LLM calls, no database logging. `rag/pipeline.py` composes this
with `PromptBuilder` and a `BaseLLM`; `search_service.py` (Phase 8)
will use this same class directly for the standalone semantic search
feature, without needing an LLM at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import settings
from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.chroma_manager import ChromaManager, VectorSearchResult
from src.utils.exceptions import EmbeddingError, RetrievalError, VectorStoreError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Output of a retrieval operation: ranked chunks plus query metadata."""

    query: str
    chunks: list[VectorSearchResult]

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def top_score(self) -> float | None:
        return self.chunks[0].similarity_score if self.chunks else None


class Retriever:
    """Embeds queries and retrieves the most relevant chunks from ChromaDB."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        chroma_manager: ChromaManager | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.chroma_manager = chroma_manager or ChromaManager()
        self.score_threshold = (
            settings.rag.similarity_score_threshold
            if score_threshold is None
            else score_threshold
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> RetrievalResult:
        """
        Retrieve the most relevant chunks for a natural-language query.

        Args:
            query: The user's natural-language question or search text.
            top_k: Number of chunks to retrieve (defaults to `settings.rag.top_k`).
            document_ids: Optional filter to scope retrieval to specific
                documents (e.g. chatting with a single document).

        Returns:
            A `RetrievalResult` with chunks above `score_threshold`,
            ranked by similarity (highest first). May be empty if
            nothing in the vector store clears the threshold — this is
            an expected, non-error outcome the caller must handle
            gracefully (e.g. "no relevant information found").

        Raises:
            RetrievalError: If query embedding or vector search fails outright.
        """
        if not query or not query.strip():
            raise RetrievalError("Cannot retrieve results for an empty query.")

        try:
            query_embedding = self.embedding_service.embed_query(query)
        except EmbeddingError as exc:
            logger.error("Query embedding failed: %s", exc)
            raise RetrievalError(f"Failed to embed query: {exc}") from exc

        try:
            raw_results = self.chroma_manager.query(
                query_embedding=query_embedding,
                top_k=top_k,
                document_ids=document_ids,
            )
        except VectorStoreError as exc:
            logger.error("Vector search failed: %s", exc)
            raise RetrievalError(f"Semantic search failed: {exc}") from exc

        filtered = [r for r in raw_results if r.similarity_score >= self.score_threshold]

        logger.info(
            "Retrieved %d/%d chunks above threshold %.2f for query: '%s'",
            len(filtered), len(raw_results), self.score_threshold, query[:80],
        )
        return RetrievalResult(query=query, chunks=filtered)
