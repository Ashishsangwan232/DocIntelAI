"""
Unit tests for src/vectorstore/chroma_manager.py

Uses a real, temporary ChromaDB instance (no mocking) — ChromaDB
itself requires no network access when embeddings are supplied
explicitly (which `ChromaManager` always does), so these tests
exercise the genuine vector storage and similarity search behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.database.models import Chunk, Document, FileType
from src.vectorstore.chroma_manager import ChromaManager
from src.utils.exceptions import VectorStoreError


@pytest.fixture()
def chroma(tmp_path: Path) -> ChromaManager:
    return ChromaManager(
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
    )


@pytest.fixture()
def sample_document() -> Document:
    return Document(
        filename="policy.pdf", file_type=FileType.PDF,
        file_size_bytes=100, file_hash="hash-1",
    )


def _make_chunks(document: Document, contents: list[str]) -> list[Chunk]:
    return [
        Chunk(document_id=document.id, chunk_index=i, content=text, page_number=1)
        for i, text in enumerate(contents)
    ]


class TestAddChunks:
    def test_add_and_count(self, chroma: ChromaManager, sample_document: Document) -> None:
        chunks = _make_chunks(sample_document, ["alpha", "beta", "gamma"])
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        chroma.add_chunks(sample_document, chunks, embeddings)
        assert chroma.count() == 3

    def test_empty_chunks_is_noop(self, chroma: ChromaManager, sample_document: Document) -> None:
        chroma.add_chunks(sample_document, [], [])
        assert chroma.count() == 0

    def test_mismatched_lengths_raise(self, chroma: ChromaManager, sample_document: Document) -> None:
        chunks = _make_chunks(sample_document, ["alpha", "beta"])
        with pytest.raises(VectorStoreError):
            chroma.add_chunks(sample_document, chunks, [[1.0, 0.0]])  # only 1 embedding for 2 chunks


class TestQuery:
    def test_returns_most_similar_first(
        self, chroma: ChromaManager, sample_document: Document
    ) -> None:
        chunks = _make_chunks(
            sample_document,
            ["The cat sat on the mat.", "Dogs are loyal animals.", "Quantum computing uses qubits."],
        )
        embeddings = [
            [1.0, 0.9, 0.1, 0.0],
            [0.9, 1.0, 0.1, 0.0],
            [0.0, 0.0, 0.1, 1.0],
        ]
        chroma.add_chunks(sample_document, chunks, embeddings)

        results = chroma.query(query_embedding=[1.0, 0.95, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        assert "cat" in results[0].content or "Dogs" in results[0].content
        assert results[0].similarity_score >= results[1].similarity_score

    def test_empty_collection_returns_empty_list(self, chroma: ChromaManager) -> None:
        results = chroma.query(query_embedding=[1.0, 0.0], top_k=5)
        assert results == []

    def test_similarity_score_in_valid_range(
        self, chroma: ChromaManager, sample_document: Document
    ) -> None:
        chunks = _make_chunks(sample_document, ["some content"])
        chroma.add_chunks(sample_document, chunks, [[1.0, 0.0]])
        results = chroma.query(query_embedding=[1.0, 0.0], top_k=1)
        assert 0.0 <= results[0].similarity_score <= 1.0

    def test_filter_by_document_ids(self, chroma: ChromaManager) -> None:
        doc_a = Document(filename="a.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="ha")
        doc_b = Document(filename="b.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="hb")
        chroma.add_chunks(doc_a, _make_chunks(doc_a, ["content from doc A"]), [[1.0, 0.0]])
        chroma.add_chunks(doc_b, _make_chunks(doc_b, ["content from doc B"]), [[0.0, 1.0]])

        results = chroma.query(query_embedding=[1.0, 0.0], top_k=5, document_ids=[doc_a.id])
        assert len(results) == 1
        assert results[0].document_id == doc_a.id

    def test_page_number_preserved(self, chroma: ChromaManager, sample_document: Document) -> None:
        chunk = Chunk(document_id=sample_document.id, chunk_index=0, content="text", page_number=7)
        chroma.add_chunks(sample_document, [chunk], [[1.0, 0.0]])
        results = chroma.query(query_embedding=[1.0, 0.0], top_k=1)
        assert results[0].page_number == 7

    def test_none_page_number_preserved_for_docx(
        self, chroma: ChromaManager, sample_document: Document
    ) -> None:
        chunk = Chunk(document_id=sample_document.id, chunk_index=0, content="text", page_number=None)
        chroma.add_chunks(sample_document, [chunk], [[1.0, 0.0]])
        results = chroma.query(query_embedding=[1.0, 0.0], top_k=1)
        assert results[0].page_number is None


class TestDeleteByDocument:
    def test_deletes_only_target_document_vectors(self, chroma: ChromaManager) -> None:
        doc_a = Document(filename="a.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="ha")
        doc_b = Document(filename="b.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="hb")
        chroma.add_chunks(doc_a, _make_chunks(doc_a, ["content A"]), [[1.0, 0.0]])
        chroma.add_chunks(doc_b, _make_chunks(doc_b, ["content B"]), [[0.0, 1.0]])

        chroma.delete_by_document(doc_a.id)
        assert chroma.count() == 1
        remaining = chroma.query(query_embedding=[0.0, 1.0], top_k=5)
        assert remaining[0].document_id == doc_b.id


class TestCount:
    def test_zero_on_empty_collection(self, chroma: ChromaManager) -> None:
        assert chroma.count() == 0
