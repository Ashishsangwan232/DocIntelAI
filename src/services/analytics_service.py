"""
analytics_service.py
======================
Aggregates usage statistics for the Analytics dashboard: document/chunk/
embedding counts, average response time, storage usage, most-queried
documents, and upload activity over time.

Kept read-only and side-effect-free — this service never writes
anything, it only reads and shapes data already captured by other
services (`document_service`, `chat_service`, `search_service` all
write the underlying rows via `SQLiteManager`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.database.sqlite_manager import SQLiteManager
from src.utils.exceptions import RecordNotFoundError
from src.utils.helpers import format_bytes
from src.utils.logger import get_logger
from src.vectorstore.chroma_manager import ChromaManager

logger = get_logger(__name__)


@dataclass
class MostQueriedDocument:
    """A single entry in the most-queried-documents ranking."""

    document_id: str
    filename: str
    query_count: int


@dataclass
class AnalyticsSummary:
    """Aggregate usage statistics shown on the Analytics dashboard."""

    total_documents: int
    total_chunks: int
    total_embeddings: int
    total_chat_sessions: int
    total_queries: int
    average_response_time_ms: float | None
    total_storage_bytes: int
    most_queried_documents: list[MostQueriedDocument] = field(default_factory=list)
    uploads_per_day: list[tuple[str, int]] = field(default_factory=list)

    @property
    def total_storage_display(self) -> str:
        return format_bytes(self.total_storage_bytes)

    @property
    def average_response_time_display(self) -> str:
        if self.average_response_time_ms is None:
            return "N/A"
        return f"{self.average_response_time_ms / 1000:.1f}s"


class AnalyticsService:
    """Read-only aggregation of usage statistics for the Analytics dashboard."""

    def __init__(
        self,
        db: SQLiteManager | None = None,
        chroma_manager: ChromaManager | None = None,
    ) -> None:
        self.db = db or SQLiteManager()
        # Optional: if omitted, total_embeddings falls back to chunk
        # count (in this application's design, every READY document's
        # chunks are always embedded 1:1 — see document_service.py —
        # so this is accurate; the real ChromaDB count is used when
        # available for an extra layer of ground truth).
        self.chroma_manager = chroma_manager

    def get_summary(self, most_queried_limit: int = 5, activity_days: int = 14) -> AnalyticsSummary:
        """Compute the full analytics summary in one call."""
        total_chunks = self.db.count_chunks()
        total_embeddings = self._get_total_embeddings(fallback=total_chunks)

        return AnalyticsSummary(
            total_documents=self.db.count_documents(),
            total_chunks=total_chunks,
            total_embeddings=total_embeddings,
            total_chat_sessions=self.db.count_chat_sessions(),
            total_queries=self.db.count_total_queries(),
            average_response_time_ms=self.db.get_average_response_time_ms(),
            total_storage_bytes=self.db.get_total_storage_bytes(),
            most_queried_documents=self._get_most_queried(most_queried_limit),
            uploads_per_day=self.db.get_documents_uploaded_per_day(activity_days),
        )

    def _get_total_embeddings(self, fallback: int) -> int:
        if self.chroma_manager is None:
            return fallback
        try:
            return self.chroma_manager.count()
        except Exception:
            logger.warning("Could not read ChromaDB vector count; falling back to chunk count")
            return fallback

    def _get_most_queried(self, limit: int) -> list[MostQueriedDocument]:
        raw = self.db.get_most_queried_documents(limit=limit)
        results: list[MostQueriedDocument] = []
        for document_id, query_count in raw:
            try:
                document = self.db.get_document(document_id)
                filename = document.filename
            except RecordNotFoundError:
                filename = "(deleted document)"
            results.append(
                MostQueriedDocument(document_id=document_id, filename=filename, query_count=query_count)
            )
        return results
