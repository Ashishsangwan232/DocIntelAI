"""
Tests for Phase B — the Documents API.

Builds a real `DocumentService` wired to fakes (a fake embedding model,
temp SQLite, temp ChromaDB directory) and overrides
`api.dependencies.get_document_service` with it via FastAPI's
`dependency_overrides` — the API-layer equivalent of how the Streamlit
upload harness (`tests/ui/upload_app_harness.py`) injects the same
kind of fakes through constructor arguments. No real embedding model
is downloaded and no real LLM call is made.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_document_service
from api.main import app
from src.database.models import Chunk, Document, DocumentStatus, FileType
from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.services.document_service import DocumentService
from src.vectorstore.chroma_manager import ChromaManager


class FakeEncoderModel:
    def encode(self, sentences, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
        return np.array([[0.1] * 8 for _ in sentences])

    def get_sentence_embedding_dimension(self) -> int:
        return 8


@pytest.fixture()
def tmp_paths(tmp_path: Path):
    db_path = tmp_path / "documents_api_test.db"
    upload_dir = tmp_path / "uploads"
    chroma_dir = tmp_path / "chroma"
    upload_dir.mkdir()
    yield db_path, upload_dir, chroma_dir
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture()
def document_service(tmp_paths) -> DocumentService:
    db_path, upload_dir, chroma_dir = tmp_paths
    return DocumentService(
        db=SQLiteManager(db_path=db_path),
        upload_dir=upload_dir,
        embedding_service=EmbeddingService(model=FakeEncoderModel()),
        chroma_manager=ChromaManager(persist_directory=chroma_dir, collection_name="test"),
    )


@pytest.fixture()
def client(document_service: DocumentService) -> TestClient:
    app.dependency_overrides[get_document_service] = lambda: document_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _seed_ready_document(document_service: DocumentService, filename: str = "seed.txt") -> Document:
    document = Document(
        filename=filename, file_type=FileType.TXT, file_size_bytes=100, file_hash=f"hash-{filename}"
    )
    document_service.db.create_document(document)
    document_service.db.bulk_create_chunks(
        [Chunk(document_id=document.id, chunk_index=0, content="seeded chunk content")]
    )
    document_service.db.update_document_status(document.id, DocumentStatus.READY, page_count=1)
    return document


class TestUpload:
    def test_upload_single_txt_file_succeeds(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/documents/upload",
            files={"files": ("hello.txt", b"Hello world, this is a test document.", "text/plain")},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 1
        result = body["results"][0]
        assert result["success"] is True
        assert result["filename"] == "hello.txt"
        assert result["document"]["status"] == "ready"
        assert result["document"]["chunk_count"] >= 1

    def test_upload_multiple_files_reports_per_file_outcomes(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/documents/upload",
            files=[
                ("files", ("a.txt", b"Content of file A.", "text/plain")),
                ("files", ("b.txt", b"Content of file B, different text.", "text/plain")),
            ],
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 2
        assert all(r["success"] for r in results)

    def test_duplicate_upload_reports_failure_without_500(self, client: TestClient) -> None:
        content = b"Exact duplicate content for hash matching."
        first = client.post(
            "/api/v1/documents/upload", files={"files": ("dup1.txt", content, "text/plain")}
        )
        assert first.json()["results"][0]["success"] is True

        second = client.post(
            "/api/v1/documents/upload", files={"files": ("dup2.txt", content, "text/plain")}
        )
        assert second.status_code == 200  # batch endpoint always 200
        result = second.json()["results"][0]
        assert result["success"] is False
        assert "already-uploaded" in result["message"] or "duplicate" in result["message"].lower()

    def test_unsupported_file_type_reports_failure(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/documents/upload",
            files={"files": ("malware.exe", b"binary junk", "application/octet-stream")},
        )
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["success"] is False
        assert result["document"] is None

    def test_mixed_batch_partial_success(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/documents/upload",
            files=[
                ("files", ("good.txt", b"Perfectly valid content here.", "text/plain")),
                ("files", ("bad.exe", b"nope", "application/octet-stream")),
            ],
        )
        results = response.json()["results"]
        assert results[0]["success"] is True
        assert results[1]["success"] is False


class TestListAndGet:
    def test_list_documents_returns_seeded_document(
        self, client: TestClient, document_service: DocumentService
    ) -> None:
        _seed_ready_document(document_service, "listed.pdf".replace(".pdf", ".txt"))
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        filenames = [d["filename"] for d in response.json()]
        assert "listed.txt" in filenames

    def test_list_filters_by_status(self, client: TestClient, document_service: DocumentService) -> None:
        ready_doc = _seed_ready_document(document_service, "ready_one.txt")
        failed_doc = Document(
            filename="failed_one.txt", file_type=FileType.TXT, file_size_bytes=10, file_hash="hash-failed"
        )
        document_service.db.create_document(failed_doc)
        document_service.db.update_document_status(failed_doc.id, DocumentStatus.FAILED)

        response = client.get("/api/v1/documents", params={"status": "ready"})
        ids = [d["id"] for d in response.json()]
        assert ready_doc.id in ids
        assert failed_doc.id not in ids

    def test_get_document_by_id(self, client: TestClient, document_service: DocumentService) -> None:
        doc = _seed_ready_document(document_service, "single.txt")
        response = client.get(f"/api/v1/documents/{doc.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == doc.id
        assert body["chunk_count"] == 1

    def test_get_nonexistent_document_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/documents/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "RecordNotFoundError"

    def test_ready_document_includes_chunk_count(
        self, client: TestClient, document_service: DocumentService
    ) -> None:
        doc = _seed_ready_document(document_service, "counted.txt")
        response = client.get(f"/api/v1/documents/{doc.id}")
        assert response.json()["chunk_count"] == 1

    def test_failed_document_has_null_chunk_count(
        self, client: TestClient, document_service: DocumentService
    ) -> None:
        failed_doc = Document(
            filename="broken.txt", file_type=FileType.TXT, file_size_bytes=10, file_hash="hash-broken"
        )
        document_service.db.create_document(failed_doc)
        document_service.db.update_document_status(failed_doc.id, DocumentStatus.FAILED)

        response = client.get(f"/api/v1/documents/{failed_doc.id}")
        assert response.json()["chunk_count"] is None


class TestPreview:
    def test_preview_returns_chunk_text(self, client: TestClient, document_service: DocumentService) -> None:
        doc = _seed_ready_document(document_service, "preview_me.txt")
        response = client.get(f"/api/v1/documents/{doc.id}/preview")
        assert response.status_code == 200
        body = response.json()
        assert body["document_id"] == doc.id
        assert "seeded chunk content" in body["preview_text"]

    def test_preview_respects_max_chars(self, client: TestClient, document_service: DocumentService) -> None:
        doc = _seed_ready_document(document_service, "long_preview.txt")
        response = client.get(f"/api/v1/documents/{doc.id}/preview", params={"max_chars": 100})
        assert response.status_code == 200
        assert len(response.json()["preview_text"]) <= 120  # truncate_text may add an ellipsis marker

    def test_preview_of_nonexistent_document_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/documents/nope/preview")
        assert response.status_code == 404


class TestDelete:
    def test_delete_removes_document(self, client: TestClient, document_service: DocumentService) -> None:
        doc = _seed_ready_document(document_service, "to_delete.txt")
        response = client.delete(f"/api/v1/documents/{doc.id}")
        assert response.status_code == 204

        follow_up = client.get(f"/api/v1/documents/{doc.id}")
        assert follow_up.status_code == 404

    def test_delete_nonexistent_document_returns_404(self, client: TestClient) -> None:
        response = client.delete("/api/v1/documents/does-not-exist")
        assert response.status_code == 404
