"""
chroma_manager.py
=================
Wraps ChromaDB behind a typed interface. This is the ONLY module in
the codebase allowed to import `chromadb` directly — `retriever.py`,
`document_service.py`, and every other caller work exclusively through
`ChromaManager`'s methods.

Design notes
------------
- A single global collection is used, with `document_id` stored as
  metadata on every vector (rather than one ChromaDB collection per
  document or per document-collection). This was the "open decision"
  flagged at the end of Phase 1: the tradeoff is that cross-document
  search is trivial (one query, optionally filtered by metadata) at
  the cost of not getting free per-collection isolation from Chroma
  itself. Isolation is instead enforced by always filtering on
  `document_id` / `collection_id` in the `where` clause when needed.
- Embeddings are always supplied explicitly by the caller (via
  `EmbeddingService`) rather than letting ChromaDB compute its own
  embeddings internally. This keeps embedding generation fully under
  our control (model choice, batching, error handling) and avoids
  ChromaDB silently downloading its own default embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import settings
from src.database.models import Chunk, Document
from src.utils.exceptions import VectorStoreError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VectorSearchResult:
    """A single result returned from a ChromaDB similarity query."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    content: str
    similarity_score: float  # normalized to [0, 1], higher = more similar


class ChromaManager:
    """Typed wrapper over ChromaDB collection CRUD and similarity search."""

    def __init__(
        self,
        persist_directory: Path | None = None,
        collection_name: str | None = None,
        distance_metric: str | None = None,
    ) -> None:
        self.persist_directory = persist_directory or settings.vectorstore.persist_directory
        self.collection_name = collection_name or settings.vectorstore.collection_name
        self.distance_metric = distance_metric or settings.vectorstore.distance_metric
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Client / collection lifecycle
    # ------------------------------------------------------------------
    def _get_collection(self):
        """Lazily create the ChromaDB client and collection on first use."""
        if self._collection is None:
            try:
                import chromadb

                self._client = chromadb.PersistentClient(path=str(self.persist_directory))
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric},
                )
            except Exception as exc:
                logger.error("Failed to initialize ChromaDB collection: %s", exc)
                raise VectorStoreError(f"Could not initialize ChromaDB: {exc}") from exc
        return self._collection

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def add_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add a document's chunks and their embeddings to the vector store.

        Args:
            document: The parent document (used for filename metadata).
            chunks: The chunks to store, must be same length/order as embeddings.
            embeddings: Pre-computed embedding vectors for each chunk.

        Raises:
            VectorStoreError: On any mismatch or ChromaDB failure.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk/embedding count mismatch: {len(chunks)} chunks vs "
                f"{len(embeddings)} embeddings."
            )

        try:
            collection = self._get_collection()
            collection.add(
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                documents=[c.content for c in chunks],
                metadatas=[
                    {
                        "document_id": c.document_id,
                        "filename": document.filename,
                        "chunk_index": c.chunk_index,
                        "page_number": c.page_number if c.page_number is not None else -1,
                    }
                    for c in chunks
                ],
            )
            logger.info(
                "Added %d vectors to ChromaDB for document %s", len(chunks), document.id
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.error("Failed to add vectors to ChromaDB: %s", exc, exc_info=True)
            raise VectorStoreError(f"Failed to store embeddings: {exc}") from exc

    def delete_by_document(self, document_id: str) -> None:
        """Remove all vectors belonging to a document (e.g. on document delete)."""
        try:
            collection = self._get_collection()
            collection.delete(where={"document_id": document_id})
            logger.info("Deleted vectors for document %s from ChromaDB", document_id)
        except Exception as exc:
            logger.error("Failed to delete vectors for document %s: %s", document_id, exc)
            raise VectorStoreError(f"Failed to delete vectors: {exc}") from exc

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def query(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Run a similarity search against the vector store.

        Args:
            query_embedding: The embedded query vector.
            top_k: Number of results to return (defaults to `settings.rag.top_k`).
            document_ids: Optional filter restricting search to specific documents
                (e.g. when chatting with a single document rather than the
                whole library).

        Returns:
            Results ordered by similarity, most similar first.

        Raises:
            VectorStoreError: If the underlying query fails.
        """
        k = top_k or settings.rag.top_k
        where_filter = {"document_id": {"$in": document_ids}} if document_ids else None

        try:
            collection = self._get_collection()
            if collection.count() == 0:
                return []

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count()),
                where=where_filter,
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc, exc_info=True)
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        return self._parse_query_results(results)

    def count(self) -> int:
        """Return the total number of vectors currently stored."""
        try:
            return self._get_collection().count()
        except Exception as exc:
            raise VectorStoreError(f"Failed to count vectors: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_query_results(self, raw_results: dict) -> list[VectorSearchResult]:
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        parsed: list[VectorSearchResult] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            page_number = metadata.get("page_number")
            parsed.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    filename=metadata.get("filename", "unknown"),
                    chunk_index=metadata.get("chunk_index", -1),
                    page_number=None if page_number == -1 else page_number,
                    content=content,
                    similarity_score=self._distance_to_similarity(distance),
                )
            )
        return parsed

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """
        Convert a ChromaDB cosine distance into an intuitive [0, 1]
        similarity score (1 = identical, 0 = unrelated), clamped to
        that range to guard against floating-point edge cases.
        """
        similarity = 1.0 - (distance / 2.0)
        return max(0.0, min(1.0, similarity))
