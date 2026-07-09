"""
analytics_app_harness.py
=========================
A minimal, self-contained Streamlit script used only by
`tests/ui/test_analytics_ui.py` via Streamlit's `AppTest` framework.

Seeds documents, chunks, chat messages, and query logs directly into
the database (no LLM/embedding calls needed — analytics/settings are
pure read/write over already-captured data) and renders
`ui/analytics.py` against real `AnalyticsService`/`RuntimeSettingsService`
instances.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.database.models import (
    ChatMessage,
    ChatSession,
    Chunk,
    Document,
    DocumentStatus,
    FileType,
    MessageRole,
    QueryLog,
)
from src.database.sqlite_manager import SQLiteManager
from src.services.analytics_service import AnalyticsService
from src.services.settings_service import RuntimeSettingsService
from src.ui import analytics as analytics_page

# Unique per-test directory (see search_app_harness.py for why this
# matters for ChromaDB paths — not strictly needed here since this
# harness never touches ChromaDB, but kept consistent for the SQLite
# db path too so parallel test runs never collide).
_BASE_DIR = Path(os.environ.get("ANALYTICS_UI_HARNESS_DIR", "/tmp/analytics_ui_harness_default"))
_DB_PATH = _BASE_DIR / "harness.db"


@st.cache_resource(show_spinner=False)
def _harness_db() -> SQLiteManager:
    return SQLiteManager(db_path=_DB_PATH)


@st.cache_resource(show_spinner=False)
def _harness_bootstrap() -> bool:
    db = _harness_db()
    if db.count_documents() > 0:
        return True

    doc_a = Document(
        filename="contract.pdf", file_type=FileType.PDF,
        file_size_bytes=204800, file_hash="hash-a",
    )
    db.create_document(doc_a)
    db.bulk_create_chunks([
        Chunk(document_id=doc_a.id, chunk_index=i, content=f"chunk {i}") for i in range(5)
    ])
    db.update_document_status(doc_a.id, DocumentStatus.READY, page_count=3)

    doc_b = Document(
        filename="policy.docx", file_type=FileType.DOCX,
        file_size_bytes=51200, file_hash="hash-b",
    )
    db.create_document(doc_b)
    db.bulk_create_chunks([Chunk(document_id=doc_b.id, chunk_index=0, content="policy text")])
    db.update_document_status(doc_b.id, DocumentStatus.READY)

    session = ChatSession(title="Test session")
    db.create_chat_session(session)
    db.add_chat_message(ChatMessage(session_id=session.id, role=MessageRole.USER, content="q1"))
    db.add_chat_message(
        ChatMessage(session_id=session.id, role=MessageRole.ASSISTANT, content="a1", response_time_ms=1500)
    )

    for _ in range(3):
        db.log_query(QueryLog(query_text="termination clause", document_id=doc_a.id, top_score=0.9))
    db.log_query(QueryLog(query_text="policy details", document_id=doc_b.id, top_score=0.8))

    return True


@st.cache_resource(show_spinner=False)
def _harness_analytics_service() -> AnalyticsService:
    return AnalyticsService(db=_harness_db())


@st.cache_resource(show_spinner=False)
def _harness_settings_service() -> RuntimeSettingsService:
    return RuntimeSettingsService(db=_harness_db())


_harness_bootstrap()
analytics_page.render(
    analytics_service=_harness_analytics_service(),
    settings_service=_harness_settings_service(),
)
