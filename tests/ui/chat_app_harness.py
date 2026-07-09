"""
chat_app_harness.py
====================
A minimal, self-contained Streamlit script used only by
`tests/ui/test_chat_ui.py` via Streamlit's `AppTest` framework.

This is NOT part of the application (`app.py` is) — it exists purely
to let `ui/chat.py` be exercised as a real, running Streamlit app in
tests, with a fake LLM/retriever wired in so no network calls are made.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.database.models import Document, DocumentStatus, FileType
from src.database.sqlite_manager import SQLiteManager
from src.llm.base import BaseLLM, LLMResponse
from src.rag.pipeline import RAGPipeline
from src.rag.prompt_builder import PromptBuilder
from src.rag.retriever import RetrievalResult
from src.services.chat_service import ChatService
from src.services.document_service import DocumentService
from src.services.export_service import ExportService
from src.ui import chat as chat_page
from src.vectorstore.chroma_manager import VectorSearchResult


class FakeLLM(BaseLLM):
    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None, model=None):
        self.call_count += 1
        return LLMResponse(
            content=f"Answer #{self.call_count}: 30 days notice required.",
            model="fake", latency_ms=5,
        )

    def stream(self, prompt, system_prompt=None, temperature=None, max_tokens=None, model=None):
        self.call_count += 1
        for token in ["30 days ", "notice ", "required."]:
            yield token


class FakeRetriever:
    def __init__(self, chunks: list[VectorSearchResult]) -> None:
        self.chunks = chunks

    def retrieve(self, query, top_k=None, document_ids=None) -> RetrievalResult:
        return RetrievalResult(query=query, chunks=self.chunks)


@st.cache_resource(show_spinner=False)
def _harness_db() -> SQLiteManager:
    return SQLiteManager(db_path=Path("/tmp/chat_ui_harness.db"))


@st.cache_resource(show_spinner=False)
def _harness_seed_document() -> Document:
    db = _harness_db()
    existing = db.list_documents()
    if existing:
        return existing[0]
    doc = Document(
        filename="contract.pdf", file_type=FileType.PDF,
        file_size_bytes=1, file_hash="harness-hash",
    )
    db.create_document(doc)
    db.update_document_status(doc.id, DocumentStatus.READY)
    return doc


@st.cache_resource(show_spinner=False)
def _harness_chat_service() -> ChatService:
    doc = _harness_seed_document()
    chunk = VectorSearchResult(
        chunk_id="c1", document_id=doc.id, filename=doc.filename,
        chunk_index=0, page_number=3, content="30 days notice required.",
        similarity_score=0.9,
    )
    pipeline = RAGPipeline(
        retriever=FakeRetriever([chunk]), prompt_builder=PromptBuilder(), llm=FakeLLM()
    )
    return ChatService(db=_harness_db(), rag_pipeline=pipeline)


document_service = DocumentService(db=_harness_db())  # embedding/chroma unused by list_documents
chat_service = _harness_chat_service()
export_service = ExportService(db=_harness_db())

chat_page.render(
    chat_service=chat_service,
    document_service=document_service,
    export_service=export_service,
)
