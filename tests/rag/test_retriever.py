"""Unit tests for src/rag/retriever.py"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.database.models import Chunk, Document, FileType
from src.embeddings.embedding_service import EmbeddingService
from src.rag.retriever import Retriever
from src.vectorstore.chroma_manager import ChromaManager
from src.utils.exceptions import RetrievalError


class DirectionalFakeModel:
    """Maps specific keywords to specific directions for controllable similarity tests."""

    def encode(self, sentences, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
        vectors = []
        for text in sentences:
            lowered = text.lower()
            if "termination" in lowered:
                vectors.append(np.array([1.0, 0.0, 0.0], dtype="float32"))
            elif "warranty" in lowered:
                vectors.append(np.array([0.0, 1.0, 0.0], dtype="float32"))
            elif "unrelated" in lowered:
                vectors.append(np.array([-1.0, 0.0, 0.0], dtype="float32"))
            else:
                vectors.append(np.array([0.5, 0.5, 0.5], dtype="float32"))
        return np.array(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        return 3


@pytest.fixture()
def retriever(tmp_path: Path) -> Retriever:
    embedding_service = EmbeddingService(model=DirectionalFakeModel())
    chroma = ChromaManager(persist_directory=tmp_path / "chroma", collection_name="test")

    document = Document(filename="contract.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h1")
    chunks = [
        Chunk(document_id=document.id, chunk_index=0, content="termination clause details", page_number=1),
        Chunk(document_id=document.id, chunk_index=1, content="warranty period details", page_number=2),
    ]
    embeddings = embedding_service.embed_texts([c.content for c in chunks])
    chroma.add_chunks(document, chunks, embeddings)

    return Retriever(embedding_service=embedding_service, chroma_manager=chroma, score_threshold=0.6)


class TestRetrieve:
    def test_returns_relevant_chunks_above_threshold(self, retriever: Retriever) -> None:
        result = retriever.retrieve("tell me about termination")
        assert not result.is_empty
        assert result.chunks[0].content == "termination clause details"

    def test_filters_out_chunks_below_threshold(self, retriever: Retriever) -> None:
        result = retriever.retrieve("something completely unrelated")
        assert result.is_empty
        assert result.chunks == []

    def test_empty_query_raises(self, retriever: Retriever) -> None:
        with pytest.raises(RetrievalError):
            retriever.retrieve("")

    def test_whitespace_query_raises(self, retriever: Retriever) -> None:
        with pytest.raises(RetrievalError):
            retriever.retrieve("   ")

    def test_top_score_property(self, retriever: Retriever) -> None:
        result = retriever.retrieve("termination")
        assert result.top_score == result.chunks[0].similarity_score

    def test_top_score_none_when_empty(self, retriever: Retriever) -> None:
        result = retriever.retrieve("unrelated")
        assert result.top_score is None

    def test_document_id_filter(self, tmp_path: Path) -> None:
        embedding_service = EmbeddingService(model=DirectionalFakeModel())
        chroma = ChromaManager(persist_directory=tmp_path / "chroma2", collection_name="test2")

        doc_a = Document(filename="a.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="ha")
        doc_b = Document(filename="b.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="hb")
        chunk_a = Chunk(document_id=doc_a.id, chunk_index=0, content="termination info")
        chunk_b = Chunk(document_id=doc_b.id, chunk_index=0, content="termination info")

        embeddings = embedding_service.embed_texts(["termination info", "termination info"])
        chroma.add_chunks(doc_a, [chunk_a], [embeddings[0]])
        chroma.add_chunks(doc_b, [chunk_b], [embeddings[1]])

        r = Retriever(embedding_service=embedding_service, chroma_manager=chroma, score_threshold=0.5)
        result = r.retrieve("termination", document_ids=[doc_a.id])
        assert len(result.chunks) == 1
        assert result.chunks[0].document_id == doc_a.id


class TestRetrievalResult:
    def test_is_empty_true_for_no_chunks(self) -> None:
        from src.rag.retriever import RetrievalResult

        result = RetrievalResult(query="q", chunks=[])
        assert result.is_empty is True

    def test_is_empty_false_when_chunks_present(self, retriever: Retriever) -> None:
        result = retriever.retrieve("termination")
        assert result.is_empty is False
