"""Tests for Phase D — the Summary API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_summary_service
from api.main import app
from src.database.models import Chunk, Document, DocumentStatus, FileType
from src.database.sqlite_manager import SQLiteManager
from src.llm.base import BaseLLM, LLMResponse
from src.services.summary_service import SummaryService

_VALID_SUMMARY_JSON = json.dumps({
    "executive_summary": "This document covers vendor obligations and termination terms.",
    "key_insights": ["30 day notice required", "Warranty capped at 12 months"],
    "topics": ["termination", "warranty"],
})


class FakeLLM(BaseLLM):
    def __init__(self, response: str = _VALID_SUMMARY_JSON):
        self.response = response

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None, model=None) -> LLMResponse:
        return LLMResponse(content=self.response, model="fake", latency_ms=5)

    def stream(self, prompt, system_prompt=None, temperature=None, max_tokens=None, model=None) -> Iterator[str]:
        yield "unused"


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "test.db")


@pytest.fixture()
def ready_document(db: SQLiteManager) -> Document:
    document = Document(filename="contract.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h1")
    db.create_document(document)
    db.bulk_create_chunks([Chunk(document_id=document.id, chunk_index=0, content="Termination clause text.")])
    db.update_document_status(document.id, DocumentStatus.READY)
    return document


@pytest.fixture()
def summary_service(db: SQLiteManager) -> SummaryService:
    return SummaryService(llm=FakeLLM(), db=db)


@pytest.fixture()
def client(summary_service: SummaryService) -> TestClient:
    app.dependency_overrides[get_summary_service] = lambda: summary_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestGenerateSummary:
    def test_generates_summary_for_document(self, client: TestClient, ready_document: Document) -> None:
        response = client.post(f"/api/v1/documents/{ready_document.id}/summary")
        assert response.status_code == 200
        body = response.json()
        assert body["document_id"] == ready_document.id
        assert body["filename"] == "contract.pdf"
        assert "termination" in body["topics"]
        assert len(body["key_insights"]) == 2

    def test_nonexistent_document_returns_404(self, client: TestClient) -> None:
        response = client.post("/api/v1/documents/does-not-exist/summary")
        assert response.status_code == 404

    def test_document_with_no_chunks_returns_400(self, client: TestClient, db: SQLiteManager) -> None:
        empty_doc = Document(filename="empty.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h2")
        db.create_document(empty_doc)
        response = client.post(f"/api/v1/documents/{empty_doc.id}/summary")
        assert response.status_code == 400

    def test_malformed_llm_output_returns_502(self, db: SQLiteManager, ready_document: Document) -> None:
        broken_service = SummaryService(llm=FakeLLM(response="not valid json at all"), db=db)
        app.dependency_overrides[get_summary_service] = lambda: broken_service
        try:
            with TestClient(app) as test_client:
                response = test_client.post(f"/api/v1/documents/{ready_document.id}/summary")
                assert response.status_code == 500  # SummaryGenerationError maps to 500
        finally:
            app.dependency_overrides.clear()
