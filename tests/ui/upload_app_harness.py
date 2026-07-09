"""
upload_app_harness.py
======================
A minimal, self-contained Streamlit script used only by
`tests/ui/test_upload_ui.py` via Streamlit's `AppTest` framework.

Seeds documents directly into the database (bypassing real file
processing/embedding, which needs no testing here — that's already
covered by `tests/ui/test_upload_processing.py` and
`tests/services/test_document_service.py`) so the library UI (search,
sort, preview, delete) can be exercised against real, persisted data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import streamlit as st

from src.database.models import Chunk, Document, DocumentStatus, FileType
from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.llm.base import BaseLLM, LLMResponse
from src.services.document_service import DocumentService
from src.services.summary_service import SummaryService
from src.ui import upload as upload_page
from src.vectorstore.chroma_manager import ChromaManager


class FakeModel:
    def encode(self, sentences, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
        import numpy as np
        return np.array([[0.1] * 8 for _ in sentences])

    def get_sentence_embedding_dimension(self) -> int:
        return 8


class FakeSummaryLLM(BaseLLM):
    """Deterministic fake for the summary dialog's AppTest coverage."""

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None) -> LLMResponse:
        return LLMResponse(
            content=json.dumps({
                "executive_summary": "This document outlines contract terms including termination and payment clauses.",
                "key_insights": ["30 day termination notice required", "Payment due within 15 days"],
                "topics": ["termination", "payment terms"],
            }),
            model="fake", latency_ms=5,
        )

    def stream(self, prompt, system_prompt=None, temperature=None, max_tokens=None) -> Iterator[str]:
        yield "unused"


@st.cache_resource(show_spinner=False)
def _harness_db() -> SQLiteManager:
    return SQLiteManager(db_path=Path("/tmp/upload_ui_harness.db"))


@st.cache_resource(show_spinner=False)
def _harness_document_service() -> DocumentService:
    db = _harness_db()
    return DocumentService(
        db=db,
        upload_dir=Path("/tmp/upload_ui_harness_files"),
        embedding_service=EmbeddingService(model=FakeModel()),
        chroma_manager=ChromaManager(
            persist_directory=Path("/tmp/upload_ui_harness_chroma"), collection_name="test"
        ),
    )


@st.cache_resource(show_spinner=False)
def _seed_documents() -> None:
    db = _harness_db()
    if db.count_documents() > 0:
        return

    ready_doc = Document(
        filename="contract_alpha.pdf", file_type=FileType.PDF,
        file_size_bytes=204800, file_hash="hash-alpha",
    )
    db.create_document(ready_doc)
    db.bulk_create_chunks(
        [Chunk(document_id=ready_doc.id, chunk_index=i, content=f"chunk {i} content") for i in range(3)]
    )
    db.update_document_status(ready_doc.id, DocumentStatus.READY, page_count=5)

    second_doc = Document(
        filename="policy_beta.docx", file_type=FileType.DOCX,
        file_size_bytes=51200, file_hash="hash-beta",
    )
    db.create_document(second_doc)
    db.bulk_create_chunks([Chunk(document_id=second_doc.id, chunk_index=0, content="policy content")])
    db.update_document_status(second_doc.id, DocumentStatus.READY)

    failed_doc = Document(
        filename="broken_gamma.pdf", file_type=FileType.PDF,
        file_size_bytes=1024, file_hash="hash-gamma",
    )
    db.create_document(failed_doc)
    db.update_document_status(failed_doc.id, DocumentStatus.FAILED)


@st.cache_resource(show_spinner=False)
def _harness_summary_service() -> SummaryService:
    return SummaryService(llm=FakeSummaryLLM(), db=_harness_db())


Path("/tmp/upload_ui_harness_files").mkdir(exist_ok=True)
_seed_documents()
document_service = _harness_document_service()
summary_service = _harness_summary_service()
upload_page.render(document_service=document_service, summary_service=summary_service)
