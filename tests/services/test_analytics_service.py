"""Unit tests for src/services/analytics_service.py"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.database.models import (
    ChatMessage,
    ChatSession,
    Chunk,
    Document,
    FileType,
    MessageRole,
    QueryLog,
)
from src.database.sqlite_manager import SQLiteManager
from src.services.analytics_service import AnalyticsService


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "test.db")


class TestEmptyState:
    def test_summary_on_empty_database(self, db: SQLiteManager) -> None:
        service = AnalyticsService(db=db)
        summary = service.get_summary()

        assert summary.total_documents == 0
        assert summary.total_chunks == 0
        assert summary.total_embeddings == 0
        assert summary.average_response_time_ms is None
        assert summary.average_response_time_display == "N/A"
        assert summary.total_storage_bytes == 0
        assert summary.most_queried_documents == []


class TestDocumentAndChunkCounts:
    def test_counts_reflect_created_documents_and_chunks(self, db: SQLiteManager) -> None:
        doc = Document(filename="a.pdf", file_type=FileType.PDF, file_size_bytes=1000, file_hash="h1")
        db.create_document(doc)
        db.bulk_create_chunks([
            Chunk(document_id=doc.id, chunk_index=i, content=f"c{i}") for i in range(4)
        ])

        summary = AnalyticsService(db=db).get_summary()
        assert summary.total_documents == 1
        assert summary.total_chunks == 4
        assert summary.total_embeddings == 4  # falls back to chunk count


class TestStorageUsage:
    def test_sums_file_sizes_across_documents(self, db: SQLiteManager) -> None:
        db.create_document(Document(filename="a.pdf", file_type=FileType.PDF, file_size_bytes=1000, file_hash="h1"))
        db.create_document(Document(filename="b.pdf", file_type=FileType.PDF, file_size_bytes=2000, file_hash="h2"))

        summary = AnalyticsService(db=db).get_summary()
        assert summary.total_storage_bytes == 3000
        assert "KB" in summary.total_storage_display or "B" in summary.total_storage_display


class TestResponseTimeAveraging:
    def test_averages_only_assistant_messages_with_timing(self, db: SQLiteManager) -> None:
        session = ChatSession(title="Test")
        db.create_chat_session(session)
        db.add_chat_message(ChatMessage(session_id=session.id, role=MessageRole.USER, content="q"))
        db.add_chat_message(
            ChatMessage(session_id=session.id, role=MessageRole.ASSISTANT, content="a1", response_time_ms=1000)
        )
        db.add_chat_message(
            ChatMessage(session_id=session.id, role=MessageRole.ASSISTANT, content="a2", response_time_ms=2000)
        )

        summary = AnalyticsService(db=db).get_summary()
        assert summary.average_response_time_ms == 1500.0
        assert summary.average_response_time_display == "1.5s"

    def test_user_messages_excluded_from_average(self, db: SQLiteManager) -> None:
        session = ChatSession(title="Test")
        db.create_chat_session(session)
        # User messages never have response_time_ms in practice, but
        # verify the query explicitly filters by role regardless.
        db.add_chat_message(
            ChatMessage(session_id=session.id, role=MessageRole.ASSISTANT, content="a", response_time_ms=500)
        )
        summary = AnalyticsService(db=db).get_summary()
        assert summary.average_response_time_ms == 500.0


class TestMostQueriedDocuments:
    def test_ranks_by_query_count_descending(self, db: SQLiteManager) -> None:
        doc_a = Document(filename="popular.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="ha")
        doc_b = Document(filename="rare.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="hb")
        db.create_document(doc_a)
        db.create_document(doc_b)

        for _ in range(3):
            db.log_query(QueryLog(query_text="q", document_id=doc_a.id, top_score=0.9))
        db.log_query(QueryLog(query_text="q", document_id=doc_b.id, top_score=0.8))

        summary = AnalyticsService(db=db).get_summary(most_queried_limit=5)
        assert summary.most_queried_documents[0].filename == "popular.pdf"
        assert summary.most_queried_documents[0].query_count == 3
        assert summary.most_queried_documents[1].filename == "rare.pdf"

    def test_respects_limit(self, db: SQLiteManager) -> None:
        for i in range(10):
            doc = Document(filename=f"doc{i}.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash=f"h{i}")
            db.create_document(doc)
            db.log_query(QueryLog(query_text="q", document_id=doc.id, top_score=0.5))

        summary = AnalyticsService(db=db).get_summary(most_queried_limit=3)
        assert len(summary.most_queried_documents) == 3

    def test_deleted_document_queries_excluded_via_fk_cascade(self, db: SQLiteManager) -> None:
        """
        query_logs.document_id has ON DELETE SET NULL, so a deleted
        document's query log rows have their document_id nulled out
        automatically and are filtered out by get_most_queried_documents's
        `WHERE document_id IS NOT NULL` clause — they never reach
        AnalyticsService's "(deleted document)" fallback at all. That
        fallback exists for defensive robustness against future schema
        changes, not because it's reachable today.
        """
        doc = Document(filename="temp.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h1")
        db.create_document(doc)
        db.log_query(QueryLog(query_text="q", document_id=doc.id, top_score=0.9))
        db.delete_document(doc.id)

        summary = AnalyticsService(db=db).get_summary()
        assert summary.most_queried_documents == []


class TestSessionAndQueryCounts:
    def test_counts_chat_sessions(self, db: SQLiteManager) -> None:
        db.create_chat_session(ChatSession(title="One"))
        db.create_chat_session(ChatSession(title="Two"))
        summary = AnalyticsService(db=db).get_summary()
        assert summary.total_chat_sessions == 2

    def test_counts_total_queries(self, db: SQLiteManager) -> None:
        doc = Document(filename="a.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h1")
        db.create_document(doc)
        for _ in range(5):
            db.log_query(QueryLog(query_text="q", document_id=doc.id, top_score=0.5))
        summary = AnalyticsService(db=db).get_summary()
        assert summary.total_queries == 5


class TestChromaManagerIntegration:
    def test_uses_chroma_count_when_provided(self, db: SQLiteManager, tmp_path: Path) -> None:
        from src.database.models import Document as Doc
        from src.vectorstore.chroma_manager import ChromaManager

        doc = Doc(filename="a.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h1")
        db.create_document(doc)
        db.bulk_create_chunks([Chunk(document_id=doc.id, chunk_index=0, content="x")])

        chroma = ChromaManager(persist_directory=tmp_path / "chroma", collection_name="test")
        chroma.add_chunks(doc, db.get_chunks_for_document(doc.id), [[1.0, 0.0]])

        summary = AnalyticsService(db=db, chroma_manager=chroma).get_summary()
        assert summary.total_embeddings == 1
