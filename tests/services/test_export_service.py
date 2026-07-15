"""Unit tests for src/services/export_service.py"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.database.models import ChatMessage, ChatSession, Citation, MessageRole
from src.database.sqlite_manager import SQLiteManager
from src.services.export_service import ExportService
from src.utils.exceptions import ValidationError


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "test.db")


@pytest.fixture()
def service(db: SQLiteManager) -> ExportService:
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


class TestExportToMarkdown:
    def test_includes_title_and_messages(
        self, service: ExportService, populated_session: ChatSession
    ) -> None:
        markdown = service.export_to_markdown(populated_session.id)
        assert "# Termination Clause Discussion" in markdown
        assert "How do I terminate this contract?" in markdown
        assert "You need to give 30 days written notice." in markdown

    def test_includes_citations(self, service: ExportService, populated_session: ChatSession) -> None:
        markdown = service.export_to_markdown(populated_session.id)
        assert "Sources" in markdown
        assert "contract.pdf, page 3" in markdown
        assert "92% match" in markdown

    def test_distinguishes_user_and_assistant(
        self, service: ExportService, populated_session: ChatSession
    ) -> None:
        markdown = service.export_to_markdown(populated_session.id)
        assert "**You:**" in markdown
        assert "**DocIntel AI:**" in markdown

    def test_empty_conversation_raises(self, service: ExportService, db: SQLiteManager) -> None:
        empty_session = ChatSession(title="Empty")
        db.create_chat_session(empty_session)
        with pytest.raises(ValidationError):
            service.export_to_markdown(empty_session.id)


class TestExportToText:
    def test_no_markdown_syntax_present(
        self, service: ExportService, populated_session: ChatSession
    ) -> None:
        text = service.export_to_text(populated_session.id)
        assert "**" not in text
        assert "#" not in text

    def test_includes_speaker_labels_and_content(
        self, service: ExportService, populated_session: ChatSession
    ) -> None:
        text = service.export_to_text(populated_session.id)
        assert "You:" in text
        assert "DocIntel AI:" in text
        assert "How do I terminate this contract?" in text

    def test_includes_citation_sources(
        self, service: ExportService, populated_session: ChatSession
    ) -> None:
        text = service.export_to_text(populated_session.id)
        assert "contract.pdf" in text

    def test_empty_conversation_raises(self, service: ExportService, db: SQLiteManager) -> None:
        empty_session = ChatSession(title="Empty")
        db.create_chat_session(empty_session)
        with pytest.raises(ValidationError):
            service.export_to_text(empty_session.id)


class TestExportToPdf:
    def test_produces_valid_pdf_bytes(
        self, service: ExportService, populated_session: ChatSession
    ) -> None:
        pdf_bytes = service.export_to_pdf(populated_session.id)
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 500

    def test_empty_conversation_raises(self, service: ExportService, db: SQLiteManager) -> None:
        empty_session = ChatSession(title="Empty")
        db.create_chat_session(empty_session)
        with pytest.raises(ValidationError):
            service.export_to_pdf(empty_session.id)

    def test_handles_special_characters_without_raising(
        self, service: ExportService, db: SQLiteManager
    ) -> None:
        session = ChatSession(title="Special <Characters> & Test")
        db.create_chat_session(session)
        db.add_chat_message(
            ChatMessage(
                session_id=session.id, role=MessageRole.USER,
                content='What about <script>alert(1)</script> & "quotes"?',
            )
        )
        db.add_chat_message(
            ChatMessage(
                session_id=session.id, role=MessageRole.ASSISTANT,
                content="Line one.\nLine two with <b>tags</b> & ampersands.",
            )
        )
        pdf_bytes = service.export_to_pdf(session.id)
        assert pdf_bytes[:4] == b"%PDF"

    def test_handles_conversation_without_citations(
        self, service: ExportService, db: SQLiteManager
    ) -> None:
        session = ChatSession(title="No Citations")
        db.create_chat_session(session)
        db.add_chat_message(ChatMessage(session_id=session.id, role=MessageRole.USER, content="Hello"))
        db.add_chat_message(
            ChatMessage(session_id=session.id, role=MessageRole.ASSISTANT, content="Hi there, no sources needed.")
        )
        pdf_bytes = service.export_to_pdf(session.id)
        assert pdf_bytes[:4] == b"%PDF"

    def test_handles_many_messages(self, service: ExportService, db: SQLiteManager) -> None:
        session = ChatSession(title="Long Conversation")
        db.create_chat_session(session)
        for i in range(20):
            db.add_chat_message(ChatMessage(session_id=session.id, role=MessageRole.USER, content=f"Question {i}"))
            db.add_chat_message(
                ChatMessage(session_id=session.id, role=MessageRole.ASSISTANT, content=f"Answer {i} " * 20)
            )
        pdf_bytes = service.export_to_pdf(session.id)
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 2000


class TestNonexistentSession:
    def test_markdown_raises_for_unknown_session(self, service: ExportService) -> None:
        with pytest.raises(Exception):  # RecordNotFoundError from SQLiteManager
            service.export_to_markdown("nonexistent-id")
