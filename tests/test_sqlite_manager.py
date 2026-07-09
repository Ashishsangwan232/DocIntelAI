"""
Unit tests for src/database/sqlite_manager.py

Each test gets a fresh, isolated SQLite database file in a pytest
tmp_path fixture, so tests never touch the real application database
and can run in any order without side effects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.database.models import (
    ChatMessage,
    ChatSession,
    Chunk,
    Citation,
    Collection,
    Document,
    DocumentStatus,
    FileType,
    MessageRole,
    QueryLog,
)
from src.database.sqlite_manager import SQLiteManager
from src.utils.exceptions import RecordNotFoundError


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "test.db")


@pytest.fixture()
def sample_document(db: SQLiteManager) -> Document:
    doc = Document(
        filename="sample.pdf",
        file_type=FileType.PDF,
        file_size_bytes=1024,
        file_hash="abc123",
    )
    db.create_document(doc)
    return doc


class TestCollections:
    def test_create_and_get(self, db: SQLiteManager) -> None:
        col = Collection(name="Contracts", description="Legal")
        db.create_collection(col)
        fetched = db.get_collection(col.id)
        assert fetched.name == "Contracts"
        assert fetched.description == "Legal"

    def test_get_missing_raises(self, db: SQLiteManager) -> None:
        with pytest.raises(RecordNotFoundError):
            db.get_collection("does-not-exist")

    def test_list_ordered_newest_first(self, db: SQLiteManager) -> None:
        db.create_collection(Collection(name="First"))
        db.create_collection(Collection(name="Second"))
        names = [c.name for c in db.list_collections()]
        assert names == ["Second", "First"]


class TestDocuments:
    def test_create_and_get(self, sample_document: Document, db: SQLiteManager) -> None:
        fetched = db.get_document(sample_document.id)
        assert fetched.filename == "sample.pdf"
        assert fetched.status == DocumentStatus.PROCESSING

    def test_duplicate_hash_lookup(self, sample_document: Document, db: SQLiteManager) -> None:
        found = db.get_document_by_hash("abc123")
        assert found is not None
        assert found.id == sample_document.id

    def test_hash_lookup_miss_returns_none(self, db: SQLiteManager) -> None:
        assert db.get_document_by_hash("nonexistent") is None

    def test_update_status_to_ready(self, sample_document: Document, db: SQLiteManager) -> None:
        db.update_document_status(
            sample_document.id, DocumentStatus.READY, page_count=10
        )
        fetched = db.get_document(sample_document.id)
        assert fetched.status == DocumentStatus.READY
        assert fetched.page_count == 10

    def test_filter_by_status(self, db: SQLiteManager) -> None:
        d1 = Document(filename="a.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h1")
        d2 = Document(filename="b.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h2")
        db.create_document(d1)
        db.create_document(d2)
        db.update_document_status(d1.id, DocumentStatus.READY)
        ready_docs = db.list_documents(status=DocumentStatus.READY)
        assert len(ready_docs) == 1
        assert ready_docs[0].id == d1.id

    def test_delete_document(self, sample_document: Document, db: SQLiteManager) -> None:
        db.delete_document(sample_document.id)
        with pytest.raises(RecordNotFoundError):
            db.get_document(sample_document.id)

    def test_count_documents(self, sample_document: Document, db: SQLiteManager) -> None:
        assert db.count_documents() == 1


class TestChunks:
    def test_bulk_insert_and_fetch_ordered(
        self, sample_document: Document, db: SQLiteManager
    ) -> None:
        chunks = [
            Chunk(document_id=sample_document.id, chunk_index=i, content=f"text {i}")
            for i in reversed(range(3))
        ]
        db.bulk_create_chunks(chunks)
        fetched = db.get_chunks_for_document(sample_document.id)
        assert [c.chunk_index for c in fetched] == [0, 1, 2]

    def test_cascade_delete_removes_chunks(
        self, sample_document: Document, db: SQLiteManager
    ) -> None:
        db.bulk_create_chunks(
            [Chunk(document_id=sample_document.id, chunk_index=0, content="x")]
        )
        db.delete_document(sample_document.id)
        assert db.get_chunks_for_document(sample_document.id) == []

    def test_empty_bulk_insert_is_noop(self, db: SQLiteManager) -> None:
        db.bulk_create_chunks([])  # should not raise


class TestChatSessionsAndMessages:
    def test_message_with_citations_round_trip(
        self, sample_document: Document, db: SQLiteManager
    ) -> None:
        session = ChatSession(title="Test session")
        db.create_chat_session(session)

        citation = Citation(
            document_id=sample_document.id,
            filename=sample_document.filename,
            chunk_id="chunk-1",
            chunk_index=0,
            similarity_score=0.92,
            excerpt="relevant excerpt",
            page_number=3,
        )
        message = ChatMessage(
            session_id=session.id,
            role=MessageRole.ASSISTANT,
            content="The answer is 42.",
            citations=[citation],
            response_time_ms=1200,
        )
        db.add_chat_message(message)

        fetched = db.get_messages_for_session(session.id)
        assert len(fetched) == 1
        assert fetched[0].content == "The answer is 42."
        assert fetched[0].citations[0].similarity_score == 0.92
        assert fetched[0].citations[0].page_number == 3

    def test_messages_ordered_chronologically(self, db: SQLiteManager) -> None:
        session = ChatSession(title="Ordering test")
        db.create_chat_session(session)
        db.add_chat_message(
            ChatMessage(session_id=session.id, role=MessageRole.USER, content="first")
        )
        db.add_chat_message(
            ChatMessage(session_id=session.id, role=MessageRole.ASSISTANT, content="second")
        )
        contents = [m.content for m in db.get_messages_for_session(session.id)]
        assert contents == ["first", "second"]


class TestQueryLogsAndAnalytics:
    def test_most_queried_documents(
        self, sample_document: Document, db: SQLiteManager
    ) -> None:
        for _ in range(3):
            db.log_query(QueryLog(query_text="q", document_id=sample_document.id, top_score=0.8))
        top = db.get_most_queried_documents()
        assert top[0] == (sample_document.id, 3)


class TestSettings:
    def test_set_and_get(self, db: SQLiteManager) -> None:
        db.set_setting("top_k", "5")
        assert db.get_setting("top_k") == "5"

    def test_upsert_overwrites(self, db: SQLiteManager) -> None:
        db.set_setting("top_k", "5")
        db.set_setting("top_k", "10")
        assert db.get_setting("top_k") == "10"

    def test_missing_key_returns_default(self, db: SQLiteManager) -> None:
        assert db.get_setting("missing", default="fallback") == "fallback"
