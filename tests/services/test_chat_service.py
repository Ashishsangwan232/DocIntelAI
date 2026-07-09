"""
Unit tests for src/services/chat_service.py

Uses a `FakeRAGPipeline`-free approach: instead, `RAGPipeline` is
constructed with a `FakeRetriever` + `FakeLLM` (same pattern as
tests/rag/test_pipeline.py), so `ChatService` is tested against the
real `RAGPipeline` orchestration logic, with only the outermost
retrieval/LLM dependencies faked out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from src.database.models import Document, FileType, MessageRole
from src.database.sqlite_manager import SQLiteManager
from src.llm.base import BaseLLM, LLMResponse
from src.rag.pipeline import RAGPipeline
from src.rag.prompt_builder import NO_CONTEXT_MESSAGE, PromptBuilder
from src.rag.retriever import RetrievalResult
from src.services.chat_service import ChatService
from src.utils.exceptions import ValidationError
from src.vectorstore.chroma_manager import VectorSearchResult


class FakeLLM(BaseLLM):
    def __init__(self, stream_chunks: list[str] | None = None):
        self.call_count = 0
        self.stream_chunks = stream_chunks or ["Streamed", " answer", " here."]

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None, model=None) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=f"Answer #{self.call_count}.", model="fake", latency_ms=5)

    def stream(self, prompt, system_prompt=None, temperature=None, max_tokens=None, model=None) -> Iterator[str]:
        self.call_count += 1
        yield from self.stream_chunks


class FakeRetriever:
    def __init__(self, chunks: list[VectorSearchResult]):
        self.chunks = chunks

    def retrieve(self, query, top_k=None, document_ids=None) -> RetrievalResult:
        return RetrievalResult(query=query, chunks=self.chunks)


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "test.db")


@pytest.fixture()
def backing_document(db: SQLiteManager) -> Document:
    """A real Document row so citation.document_id satisfies the FK constraint."""
    doc = Document(filename="contract.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h1")
    db.create_document(doc)
    return doc


@pytest.fixture()
def sample_chunk(backing_document: Document) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id="c1", document_id=backing_document.id, filename="contract.pdf",
        chunk_index=0, page_number=3, content="30 days notice required.", similarity_score=0.9,
    )


@pytest.fixture()
def chat_service(db: SQLiteManager, sample_chunk: VectorSearchResult) -> ChatService:
    fake_llm = FakeLLM()
    pipeline = RAGPipeline(
        retriever=FakeRetriever([sample_chunk]), prompt_builder=PromptBuilder(), llm=fake_llm
    )
    return ChatService(db=db, rag_pipeline=pipeline)


class TestSessionManagement:
    def test_create_session_defaults(self, chat_service: ChatService) -> None:
        session = chat_service.create_session()
        assert session.title == "New Conversation"

    def test_create_session_with_custom_title(self, chat_service: ChatService) -> None:
        session = chat_service.create_session(title="Contract Review")
        assert session.title == "Contract Review"

    def test_list_sessions_returns_created_sessions(self, chat_service: ChatService) -> None:
        chat_service.create_session(title="First")
        chat_service.create_session(title="Second")
        titles = [s.title for s in chat_service.list_sessions()]
        assert "First" in titles and "Second" in titles

    def test_delete_session_removes_it(self, chat_service: ChatService) -> None:
        session = chat_service.create_session()
        chat_service.delete_session(session.id)
        assert chat_service.list_sessions() == []


class TestSendMessage:
    def test_persists_user_and_assistant_messages(self, chat_service: ChatService) -> None:
        session = chat_service.create_session()
        reply = chat_service.send_message(session.id, "How do I terminate this contract?")

        assert reply.role == MessageRole.ASSISTANT
        assert "Answer #1" in reply.content
        assert len(reply.citations) == 1

        messages = chat_service.get_messages(session.id)
        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "How do I terminate this contract?"
        assert messages[1].role == MessageRole.ASSISTANT

    def test_empty_message_rejected(self, chat_service: ChatService) -> None:
        session = chat_service.create_session()
        with pytest.raises(ValidationError):
            chat_service.send_message(session.id, "   ")

    def test_empty_message_not_persisted(self, chat_service: ChatService) -> None:
        session = chat_service.create_session()
        with pytest.raises(ValidationError):
            chat_service.send_message(session.id, "")
        assert chat_service.get_messages(session.id) == []

    def test_query_logged_for_analytics(
        self, chat_service: ChatService, db: SQLiteManager, backing_document: Document
    ) -> None:
        session = chat_service.create_session()
        chat_service.send_message(session.id, "termination question")
        top_queried = db.get_most_queried_documents()
        assert top_queried[0][0] == backing_document.id

    def test_no_context_response_has_no_citations(self, db: SQLiteManager) -> None:
        pipeline = RAGPipeline(
            retriever=FakeRetriever([]), prompt_builder=PromptBuilder(), llm=FakeLLM()
        )
        service = ChatService(db=db, rag_pipeline=pipeline)
        session = service.create_session()
        reply = service.send_message(session.id, "unrelated question")
        assert reply.content == NO_CONTEXT_MESSAGE
        assert reply.citations == []


class TestRegenerateResponse:
    def test_replaces_last_assistant_message(self, chat_service: ChatService) -> None:
        session = chat_service.create_session()
        chat_service.send_message(session.id, "first question")

        regenerated = chat_service.regenerate_last_response(session.id)
        assert "Answer #2" in regenerated.content

        messages = chat_service.get_messages(session.id)
        assert len(messages) == 2  # still just user + regenerated assistant
        assert messages[1].content == regenerated.content

    def test_regenerate_uses_original_question(self, chat_service: ChatService) -> None:
        session = chat_service.create_session()
        chat_service.send_message(session.id, "original question text")
        chat_service.regenerate_last_response(session.id)
        messages = chat_service.get_messages(session.id)
        assert messages[0].content == "original question text"  # user message unchanged

    def test_regenerate_with_no_messages_raises(self, chat_service: ChatService) -> None:
        session = chat_service.create_session()
        with pytest.raises(ValidationError):
            chat_service.regenerate_last_response(session.id)


class TestStreamingMessages:
    def test_citations_available_before_stream_consumed(self, chat_service: ChatService) -> None:
        session = chat_service.create_session()
        stream, citations = chat_service.send_message_streaming(session.id, "question")
        assert len(citations) == 1  # available immediately, before iterating stream

    def test_assistant_message_persisted_only_after_stream_consumed(
        self, chat_service: ChatService
    ) -> None:
        session = chat_service.create_session()
        stream, _ = chat_service.send_message_streaming(session.id, "question")

        # Before consuming: only the user message exists.
        assert len(chat_service.get_messages(session.id)) == 1

        full_answer = "".join(stream)

        # After consuming: assistant message now persisted.
        messages = chat_service.get_messages(session.id)
        assert len(messages) == 2
        assert messages[1].content == full_answer

    def test_streamed_content_matches_concatenated_chunks(self, db: SQLiteManager, sample_chunk) -> None:
        fake_llm = FakeLLM(stream_chunks=["Hello", " ", "world"])
        pipeline = RAGPipeline(
            retriever=FakeRetriever([sample_chunk]), prompt_builder=PromptBuilder(), llm=fake_llm
        )
        service = ChatService(db=db, rag_pipeline=pipeline)
        session = service.create_session()

        stream, _ = service.send_message_streaming(session.id, "q")
        assert "".join(stream) == "Hello world"


class TestClearConversation:
    def test_clear_removes_messages_but_keeps_session(self, chat_service: ChatService) -> None:
        session = chat_service.create_session()
        chat_service.send_message(session.id, "a question")
        chat_service.clear_conversation(session.id)

        assert chat_service.get_messages(session.id) == []
        assert any(s.id == session.id for s in chat_service.list_sessions())


class TestDeriveSessionTitle:
    def test_short_message_used_as_is(self) -> None:
        assert ChatService.derive_session_title("Hello there") == "Hello there"

    def test_long_message_truncated_with_ellipsis(self) -> None:
        long_msg = "What are the termination clauses in this contract regarding penalties?"
        title = ChatService.derive_session_title(long_msg)
        assert len(title) <= 53
        assert title.endswith("...")

    def test_collapses_internal_whitespace(self) -> None:
        assert ChatService.derive_session_title("  hello   world  ") == "hello world"
