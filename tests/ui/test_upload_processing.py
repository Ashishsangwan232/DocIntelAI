"""
Unit tests for src/ui/upload.py's `_attempt_process` — the pure,
Streamlit-free unit of upload logic. Tested directly (no `AppTest`,
no `st.file_uploader` simulation needed) since it has zero Streamlit
dependency by design.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.services.document_service import DocumentService
from src.ui.upload import _attempt_process
from src.vectorstore.chroma_manager import ChromaManager
from tests.conftest import FakeEmbeddingModel


@pytest.fixture()
def document_service(tmp_path: Path) -> DocumentService:
    db = SQLiteManager(db_path=tmp_path / "test.db")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return DocumentService(
        db=db,
        upload_dir=upload_dir,
        embedding_service=EmbeddingService(model=FakeEmbeddingModel(dimension=16)),
        chroma_manager=ChromaManager(persist_directory=tmp_path / "chroma", collection_name="test"),
    )


def _make_pdf_bytes(text: str = "Some contract content here.") -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    return pdf.tobytes()


class TestAttemptProcess:
    def test_successful_upload_returns_success_outcome(self, document_service: DocumentService) -> None:
        outcome = _attempt_process("notes.txt", b"some plain text content", document_service)
        assert outcome.success is True
        assert outcome.filename == "notes.txt"
        assert outcome.document is not None
        assert outcome.document.filename == "notes.txt"

    def test_unsupported_extension_returns_failure_outcome(self, document_service: DocumentService) -> None:
        outcome = _attempt_process("virus.exe", b"binary data", document_service)
        assert outcome.success is False
        assert outcome.document is None
        assert "exe" in outcome.message.lower() or "not supported" in outcome.message.lower()

    def test_duplicate_upload_returns_failure_outcome(self, document_service: DocumentService) -> None:
        _attempt_process("first.txt", b"identical content", document_service)
        outcome = _attempt_process("second.txt", b"identical content", document_service)
        assert outcome.success is False
        assert "already-uploaded" in outcome.message or "duplicate" in outcome.message.lower() or "matches" in outcome.message.lower()

    def test_corrupted_pdf_returns_failure_outcome_not_exception(self, document_service: DocumentService) -> None:
        # Must not raise — the UI loop depends on this always returning
        # an UploadOutcome so one bad file doesn't crash the whole batch.
        outcome = _attempt_process("bad.pdf", b"not a real pdf", document_service)
        assert outcome.success is False
        assert "bad.pdf" in outcome.message

    def test_empty_file_returns_failure_outcome(self, document_service: DocumentService) -> None:
        outcome = _attempt_process("empty.txt", b"", document_service)
        assert outcome.success is False

    def test_valid_pdf_processed_successfully(self, document_service: DocumentService) -> None:
        outcome = _attempt_process("contract.pdf", _make_pdf_bytes(), document_service)
        assert outcome.success is True
        assert outcome.document.page_count == 1

    def test_batch_of_mixed_files_each_get_independent_outcome(
        self, document_service: DocumentService
    ) -> None:
        files = [
            ("good1.txt", b"unique content one"),
            ("bad.exe", b"binary junk"),
            ("good2.txt", b"unique content two"),
        ]
        outcomes = [_attempt_process(name, content, document_service) for name, content in files]
        assert [o.success for o in outcomes] == [True, False, True]
