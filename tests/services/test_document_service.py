"""
Unit tests for src/services/document_service.py

Uses a temp SQLite DB and temp upload directory per test (via
pytest fixtures) so every test is fully isolated and leaves no
artifacts behind.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.database.models import DocumentStatus
from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.services.document_service import DocumentService
from src.vectorstore.chroma_manager import ChromaManager
from src.utils.exceptions import (
    CorruptedFileError,
    DocumentProcessingError,
    DuplicateDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    ValidationError,
)
from tests.conftest import FakeEmbeddingModel


@pytest.fixture()
def service(tmp_path: Path) -> DocumentService:
    db = SQLiteManager(db_path=tmp_path / "test.db")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    embedding_service = EmbeddingService(model=FakeEmbeddingModel(dimension=16))
    chroma_manager = ChromaManager(
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
    )
    return DocumentService(
        db=db,
        upload_dir=upload_dir,
        embedding_service=embedding_service,
        chroma_manager=chroma_manager,
    )


def _make_pdf_bytes(text: str = "Some contract text about warranties.") -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    return pdf.tobytes()


class TestProcessUploadHappyPath:
    def test_processes_valid_pdf(self, service: DocumentService) -> None:
        document = service.process_upload("contract.pdf", _make_pdf_bytes())
        assert document.filename == "contract.pdf"
        assert document.page_count == 1

        chunks = service.db.get_chunks_for_document(document.id)
        assert len(chunks) >= 1
        assert "warranties" in chunks[0].content

    def test_processes_valid_txt(self, service: DocumentService) -> None:
        document = service.process_upload("notes.txt", b"Some plain text notes.")
        chunks = service.db.get_chunks_for_document(document.id)
        assert len(chunks) == 1

    def test_sanitizes_filename(self, service: DocumentService) -> None:
        document = service.process_upload("My File (v2).txt", b"content here")
        assert document.filename == "My_File_v2.txt"

    def test_file_saved_to_disk(self, service: DocumentService, tmp_path: Path) -> None:
        document = service.process_upload("notes.txt", b"some content")
        stored_files = list((tmp_path / "uploads").iterdir())
        assert any(document.id in f.name for f in stored_files)


class TestValidationErrors:
    def test_unsupported_extension_rejected(self, service: DocumentService) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            service.process_upload("virus.exe", b"binary content")

    def test_empty_file_rejected(self, service: DocumentService) -> None:
        with pytest.raises(ValidationError):
            service.process_upload("empty.txt", b"")

    def test_oversized_file_rejected(self, service: DocumentService) -> None:
        oversized = b"x" * (26 * 1024 * 1024)
        with pytest.raises(FileTooLargeError):
            service.process_upload("big.txt", oversized)

    def test_document_not_created_on_validation_failure(
        self, service: DocumentService
    ) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            service.process_upload("virus.exe", b"data")
        assert service.db.count_documents() == 0


class TestDuplicateDetection:
    def test_duplicate_content_rejected(self, service: DocumentService) -> None:
        service.process_upload("first.txt", b"identical content")
        with pytest.raises(DuplicateDocumentError):
            service.process_upload("second.txt", b"identical content")

    def test_different_content_not_flagged_as_duplicate(
        self, service: DocumentService
    ) -> None:
        service.process_upload("a.txt", b"content A")
        service.process_upload("b.txt", b"content B")
        assert service.db.count_documents() == 2


class TestCorruptedFiles:
    def test_corrupted_pdf_marks_document_failed(self, service: DocumentService) -> None:
        with pytest.raises(CorruptedFileError) as exc_info:
            service.process_upload("bad.pdf", b"not a real pdf")
        assert "bad.pdf" in str(exc_info.value)

        documents = service.db.list_documents(status=DocumentStatus.FAILED)
        assert len(documents) == 1
        assert documents[0].filename == "bad.pdf"

    def test_error_message_uses_original_filename_not_internal_path(
        self, service: DocumentService
    ) -> None:
        with pytest.raises(CorruptedFileError) as exc_info:
            service.process_upload("quarterly_report.pdf", b"garbage")
        message = str(exc_info.value)
        assert "quarterly_report.pdf" in message
        # internal storage uses a UUID prefix; that must never leak to the user
        assert message.count("_quarterly_report.pdf") == 0


class TestEmbeddingIntegration:
    def test_chunks_are_embedded_and_stored_in_chroma(
        self, service: DocumentService
    ) -> None:
        document = service.process_upload("contract.pdf", _make_pdf_bytes())
        assert document.status == DocumentStatus.READY
        assert service.chroma_manager.count() >= 1

    def test_embedding_failure_marks_document_failed_and_removes_chunks(
        self, tmp_path: Path
    ) -> None:
        class FailingEmbeddingModel:
            def encode(self, *args, **kwargs):
                raise RuntimeError("simulated embedding backend outage")

            def get_sentence_embedding_dimension(self) -> int:
                return 16

        db = SQLiteManager(db_path=tmp_path / "test.db")
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        failing_service = DocumentService(
            db=db,
            upload_dir=upload_dir,
            embedding_service=EmbeddingService(model=FailingEmbeddingModel()),
            chroma_manager=ChromaManager(
                persist_directory=tmp_path / "chroma", collection_name="test"
            ),
        )

        with pytest.raises(DocumentProcessingError) as exc_info:
            failing_service.process_upload("notes.txt", b"some plain content")
        assert "embedding failed" in str(exc_info.value).lower()

        failed_docs = db.list_documents(status=DocumentStatus.FAILED)
        assert len(failed_docs) == 1
        assert db.get_chunks_for_document(failed_docs[0].id) == []

    def test_delete_document_removes_vectors_from_chroma(
        self, service: DocumentService
    ) -> None:
        document = service.process_upload("contract.pdf", _make_pdf_bytes())
        count_before = service.chroma_manager.count()
        assert count_before > 0

        service.delete_document(document.id)
        # No vectors should remain for this document after deletion.
        remaining = service.chroma_manager.query(
            query_embedding=[0.5] * 16, top_k=10, document_ids=[document.id]
        )
        assert remaining == []


class TestDocumentManagement:
    def test_delete_document_removes_file_and_chunks(
        self, service: DocumentService, tmp_path: Path
    ) -> None:
        document = service.process_upload("notes.txt", b"some content here")
        service.delete_document(document.id)

        assert service.db.get_chunks_for_document(document.id) == []
        stored_files = list((tmp_path / "uploads").iterdir())
        assert not any(document.id in f.name for f in stored_files)

    def test_list_documents_filters_by_status(self, service: DocumentService) -> None:
        service.process_upload("good.txt", b"valid content")
        try:
            service.process_upload("bad.pdf", b"not a pdf")
        except CorruptedFileError:
            pass

        failed = service.list_documents(status=DocumentStatus.FAILED)
        assert len(failed) == 1
        assert failed[0].filename == "bad.pdf"
