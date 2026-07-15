"""Tests for Phase D — the Export API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_export_service
from api.main import app
from src.database.models import ChatMessage, ChatSession, Citation, MessageRole
from src.database.sqlite_manager import SQLiteManager
from src.services.export_service import ExportService


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "test.db")


@pytest.fixture()
def export_service(db: SQLiteManager) -> ExportService:
    return ExportService(db=db)


@pytest.fixture()
def populated_session(db: SQLiteManager) -> ChatSession:
    session = ChatSession(title="Termination Clause Discussion")
    db.create_chat_session(session)
    db.add_chat_message(
        ChatMessage(session_id=session.id, role=MessageRole.USER, content="How do I terminate this contract?")
    )
    citation = Citation(
        document_id="doc-1", filename="contract.pdf", chunk_id="c1",
        chunk_index=0, similarity_score=0.92, excerpt="30 days notice", page_number=3,
    )
    db.add_chat_message(
        ChatMessage(
            session_id=session.id, role=MessageRole.ASSISTANT,
            content="You need to give 30 days written notice.", citations=[citation],
        )
    )
    return session


@pytest.fixture()
def client(export_service: ExportService) -> TestClient:
    app.dependency_overrides[get_export_service] = lambda: export_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestExportFormats:
    def test_markdown_export(self, client: TestClient, populated_session: ChatSession) -> None:
        response = client.get(
            f"/api/v1/chat/sessions/{populated_session.id}/export", params={"format": "markdown"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert "attachment" in response.headers["content-disposition"]
        assert ".md" in response.headers["content-disposition"]
        assert "30 days written notice" in response.text

    def test_text_export(self, client: TestClient, populated_session: ChatSession) -> None:
        response = client.get(
            f"/api/v1/chat/sessions/{populated_session.id}/export", params={"format": "text"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert ".txt" in response.headers["content-disposition"]

    def test_pdf_export(self, client: TestClient, populated_session: ChatSession) -> None:
        response = client.get(
            f"/api/v1/chat/sessions/{populated_session.id}/export", params={"format": "pdf"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert ".pdf" in response.headers["content-disposition"]
        assert response.content[:4] == b"%PDF"

    def test_default_format_is_markdown(self, client: TestClient, populated_session: ChatSession) -> None:
        response = client.get(f"/api/v1/chat/sessions/{populated_session.id}/export")
        assert response.headers["content-type"].startswith("text/markdown")

    def test_invalid_format_returns_422(self, client: TestClient, populated_session: ChatSession) -> None:
        response = client.get(
            f"/api/v1/chat/sessions/{populated_session.id}/export", params={"format": "docx"}
        )
        assert response.status_code == 422


class TestExportErrors:
    def test_nonexistent_session_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/chat/sessions/does-not-exist/export")
        assert response.status_code == 404

    def test_session_with_no_messages_returns_400(self, client: TestClient, db: SQLiteManager) -> None:
        empty_session = ChatSession(title="Empty")
        db.create_chat_session(empty_session)
        response = client.get(f"/api/v1/chat/sessions/{empty_session.id}/export")
        assert response.status_code == 400
