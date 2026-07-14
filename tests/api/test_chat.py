"""
Tests for Phase C — the Chat API.

Uses a real `ChatService` backed by a real `RAGPipeline`, with only
the outermost `Retriever`/`BaseLLM` dependencies faked — same pattern
as `tests/services/test_chat_service.py` — wired in via
`app.dependency_overrides`. No real embedding model, vector store, or
LLM call happens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_chat_service, get_settings_service
from api.main import app
from src.database.models import Document, FileType, MessageRole
from src.database.sqlite_manager import SQLiteManager
from src.llm.base import BaseLLM, LLMResponse
from src.rag.pipeline import RAGPipeline
from src.rag.prompt_builder import PromptBuilder
from src.rag.retriever import RetrievalResult
from src.services.chat_service import ChatService
from src.services.settings_service import RuntimeSettingsService
from src.utils.exceptions import LLMTimeoutError
from src.vectorstore.chroma_manager import VectorSearchResult


class FakeLLM(BaseLLM):
    def __init__(self, stream_chunks: list[str] | None = None, raise_on_stream: Exception | None = None):
        self.call_count = 0
        self.stream_chunks = stream_chunks or ["Streamed", " answer", " here."]
        self.raise_on_stream = raise_on_stream

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None, model=None) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=f"Answer #{self.call_count}.", model="fake", latency_ms=5)

    def stream(self, prompt, system_prompt=None, temperature=None, max_tokens=None, model=None) -> Iterator[str]:
        self.call_count += 1
        if self.raise_on_stream is not None:
            yield self.stream_chunks[0]
            raise self.raise_on_stream
        yield from self.stream_chunks


class FakeRetriever:
    def __init__(self, chunks: list[VectorSearchResult]):
        self.chunks = chunks

    def retrieve(self, query, top_k=None, document_ids=None) -> RetrievalResult:
        return RetrievalResult(query=query, chunks=self.chunks)


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "chat_api_test.db")


@pytest.fixture()
def backing_document(db: SQLiteManager) -> Document:
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
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def chat_service(db: SQLiteManager, sample_chunk: VectorSearchResult, fake_llm: FakeLLM) -> ChatService:
    pipeline = RAGPipeline(retriever=FakeRetriever([sample_chunk]), prompt_builder=PromptBuilder(), llm=fake_llm)
    return ChatService(db=db, rag_pipeline=pipeline)


@pytest.fixture()
def settings_service(db: SQLiteManager) -> RuntimeSettingsService:
    return RuntimeSettingsService(db=db)


@pytest.fixture()
def client(chat_service: ChatService, settings_service: RuntimeSettingsService) -> TestClient:
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_settings_service] = lambda: settings_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _create_session(client: TestClient, title: str = "Test Session") -> str:
    response = client.post("/api/v1/chat/sessions", json={"title": title})
    assert response.status_code == 201
    return response.json()["id"]


class TestSessions:
    def test_create_session_defaults_title(self, client: TestClient) -> None:
        response = client.post("/api/v1/chat/sessions", json={})
        assert response.status_code == 201
        assert response.json()["title"] == "New Conversation"

    def test_create_session_with_custom_title(self, client: TestClient) -> None:
        response = client.post("/api/v1/chat/sessions", json={"title": "Contract Review"})
        assert response.json()["title"] == "Contract Review"

    def test_list_sessions(self, client: TestClient) -> None:
        _create_session(client, "First")
        _create_session(client, "Second")
        response = client.get("/api/v1/chat/sessions")
        titles = [s["title"] for s in response.json()]
        assert "First" in titles and "Second" in titles

    def test_delete_session(self, client: TestClient) -> None:
        session_id = _create_session(client)
        response = client.delete(f"/api/v1/chat/sessions/{session_id}")
        assert response.status_code == 204
        assert client.get(f"/api/v1/chat/sessions/{session_id}/messages").status_code == 404

    def test_delete_nonexistent_session_returns_404(self, client: TestClient) -> None:
        response = client.delete("/api/v1/chat/sessions/does-not-exist")
        assert response.status_code == 404


class TestSendMessage:
    def test_send_message_returns_assistant_reply_with_citations(self, client: TestClient) -> None:
        session_id = _create_session(client)
        response = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages", json={"query": "What is the notice period?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["role"] == "assistant"
        assert body["session_id"] == session_id
        assert len(body["citations"]) == 1
        assert body["citations"][0]["filename"] == "contract.pdf"

    def test_send_message_to_nonexistent_session_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/chat/sessions/does-not-exist/messages", json={"query": "hello"}
        )
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "RecordNotFoundError"

    def test_empty_query_returns_400(self, client: TestClient) -> None:
        session_id = _create_session(client)
        response = client.post(f"/api/v1/chat/sessions/{session_id}/messages", json={"query": "   "})
        assert response.status_code == 400

    def test_missing_query_field_returns_422(self, client: TestClient) -> None:
        session_id = _create_session(client)
        response = client.post(f"/api/v1/chat/sessions/{session_id}/messages", json={})
        assert response.status_code == 422

    def test_get_messages_returns_full_history(self, client: TestClient) -> None:
        session_id = _create_session(client)
        client.post(f"/api/v1/chat/sessions/{session_id}/messages", json={"query": "Question one"})
        response = client.get(f"/api/v1/chat/sessions/{session_id}/messages")
        roles = [m["role"] for m in response.json()]
        assert roles == ["user", "assistant"]

    def test_get_messages_for_nonexistent_session_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/chat/sessions/does-not-exist/messages")
        assert response.status_code == 404


class TestClearConversation:
    def test_clear_removes_messages_but_keeps_session(self, client: TestClient) -> None:
        session_id = _create_session(client)
        client.post(f"/api/v1/chat/sessions/{session_id}/messages", json={"query": "hi"})

        response = client.delete(f"/api/v1/chat/sessions/{session_id}/messages")
        assert response.status_code == 204

        messages = client.get(f"/api/v1/chat/sessions/{session_id}/messages").json()
        assert messages == []
        sessions = [s["id"] for s in client.get("/api/v1/chat/sessions").json()]
        assert session_id in sessions


class TestRegenerate:
    def test_regenerate_replaces_last_assistant_message(
        self, client: TestClient, fake_llm: FakeLLM
    ) -> None:
        session_id = _create_session(client)
        client.post(f"/api/v1/chat/sessions/{session_id}/messages", json={"query": "hi"})
        assert fake_llm.call_count == 1

        response = client.post(f"/api/v1/chat/sessions/{session_id}/regenerate", json={})
        assert response.status_code == 200
        assert fake_llm.call_count == 2

        messages = client.get(f"/api/v1/chat/sessions/{session_id}/messages").json()
        assert len(messages) == 2  # still just [user, assistant] — old assistant reply replaced

    def test_regenerate_on_empty_session_returns_400(self, client: TestClient) -> None:
        session_id = _create_session(client)
        response = client.post(f"/api/v1/chat/sessions/{session_id}/regenerate", json={})
        assert response.status_code == 400

    def test_regenerate_on_nonexistent_session_returns_404(self, client: TestClient) -> None:
        response = client.post("/api/v1/chat/sessions/does-not-exist/regenerate", json={})
        assert response.status_code == 404


class TestStreaming:
    def _parse_sse(self, text: str) -> list[tuple[str, str]]:
        events = []
        for block in text.strip().split("\n\n"):
            if not block.strip():
                continue
            lines = block.strip().split("\n")
            event_line = next((l for l in lines if l.startswith("event:")), "event: message")
            data_line = next((l for l in lines if l.startswith("data:")), "data: ")
            events.append((event_line.split(":", 1)[1].strip(), data_line.split(":", 1)[1].strip()))
        return events

    def test_stream_emits_citations_then_tokens_then_done(self, client: TestClient) -> None:
        session_id = _create_session(client)
        response = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages/stream", json={"query": "hi"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = self._parse_sse(response.text)
        event_names = [name for name, _ in events]
        assert event_names[0] == "citations"
        assert event_names[-1] == "done"
        assert "token" in event_names

    def test_stream_persists_assistant_message_once_consumed(self, client: TestClient) -> None:
        session_id = _create_session(client)
        client.post(f"/api/v1/chat/sessions/{session_id}/messages/stream", json={"query": "hi"})

        messages = client.get(f"/api/v1/chat/sessions/{session_id}/messages").json()
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"] == "Streamed answer here."

    def test_stream_to_nonexistent_session_returns_404_not_sse(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/chat/sessions/does-not-exist/messages/stream", json={"query": "hi"}
        )
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")

    def test_stream_reports_llm_error_as_sse_error_event(
        self, db: SQLiteManager, sample_chunk: VectorSearchResult
    ) -> None:
        failing_llm = FakeLLM(raise_on_stream=LLMTimeoutError("Ollama Cloud did not respond in time."))
        pipeline = RAGPipeline(
            retriever=FakeRetriever([sample_chunk]), prompt_builder=PromptBuilder(), llm=failing_llm
        )
        failing_chat_service = ChatService(db=db, rag_pipeline=pipeline)

        app.dependency_overrides[get_chat_service] = lambda: failing_chat_service
        app.dependency_overrides[get_settings_service] = lambda: RuntimeSettingsService(db=db)
        try:
            with TestClient(app) as test_client:
                session_id = _create_session(test_client)
                response = test_client.post(
                    f"/api/v1/chat/sessions/{session_id}/messages/stream", json={"query": "hi"}
                )
                assert response.status_code == 200  # already committed by the time the error fires
                events = self._parse_sse(response.text)
                event_names = [name for name, _ in events]
                assert "error" in event_names
                assert event_names[-1] == "done"
        finally:
            app.dependency_overrides.clear()
