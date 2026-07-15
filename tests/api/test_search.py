"""Tests for Phase D — the Search API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_search_service
from api.main import app
from src.database.models import Chunk, Document, FileType
from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.rag.retriever import Retriever
from src.services.search_service import SearchService
from src.vectorstore.chroma_manager import ChromaManager


class DirectionalFakeModel:
    def encode(self, sentences, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
        vectors = []
        for text in sentences:
            if "notice" in text.lower():
                vectors.append(np.array([1.0, 0.0, 0.0], dtype="float32"))
            else:
                vectors.append(np.array([0.0, 1.0, 0.0], dtype="float32"))
        return np.array(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        return 3


@pytest.fixture()
def search_service(tmp_path: Path) -> SearchService:
    db = SQLiteManager(db_path=tmp_path / "test.db")
    embedding_service = EmbeddingService(model=DirectionalFakeModel())
    chroma = ChromaManager(persist_directory=tmp_path / "chroma", collection_name="test")

    document = Document(filename="contract.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h1")
    db.create_document(document)
    chunk = Chunk(
        document_id=document.id, chunk_index=0, page_number=4,
        content="Either party may terminate this agreement with 30 days written notice.",
    )
    embeddings = embedding_service.embed_texts([chunk.content])
    chroma.add_chunks(document, [chunk], embeddings)

    retriever = Retriever(embedding_service=embedding_service, chroma_manager=chroma, score_threshold=0.1)
    return SearchService(retriever=retriever, db=db)


@pytest.fixture()
def client(search_service: SearchService) -> TestClient:
    app.dependency_overrides[get_search_service] = lambda: search_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestSearch:
    def test_returns_ranked_hits(self, client: TestClient) -> None:
        response = client.post("/api/v1/search", json={"query": "30 days notice"})
        assert response.status_code == 200
        hits = response.json()["hits"]
        assert len(hits) == 1
        assert hits[0]["filename"] == "contract.pdf"
        assert hits[0]["page_number"] == 4

    def test_snippet_highlights_matching_terms(self, client: TestClient) -> None:
        response = client.post("/api/v1/search", json={"query": "30 days notice"})
        snippet = response.json()["hits"][0]["snippet"]
        assert "**" in snippet

    def test_empty_query_returns_400(self, client: TestClient) -> None:
        response = client.post("/api/v1/search", json={"query": "   "})
        assert response.status_code == 400

    def test_missing_query_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/search", json={})
        assert response.status_code == 422

    def test_top_k_out_of_range_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/search", json={"query": "notice", "top_k": 100})
        assert response.status_code == 422

    def test_no_matches_returns_empty_list_not_error(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/search", json={"query": "totally unrelated xyz", "document_ids": ["nonexistent"]}
        )
        assert response.status_code == 200
        assert response.json()["hits"] == []
