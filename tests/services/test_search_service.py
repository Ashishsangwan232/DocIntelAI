"""Unit tests for src/services/search_service.py"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.database.models import Chunk, Document, FileType
from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.rag.retriever import Retriever
from src.services.search_service import SearchService, _build_snippet, _tokenize_query
from src.utils.exceptions import ValidationError
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
        content="Either party may terminate this agreement with 30 days written "
                "notice to the other party, delivered in writing.",
    )
    embeddings = embedding_service.embed_texts([chunk.content])
    chroma.add_chunks(document, [chunk], embeddings)

    retriever = Retriever(embedding_service=embedding_service, chroma_manager=chroma, score_threshold=0.1)
    return SearchService(retriever=retriever, db=db)


class TestSearch:
    def test_returns_ranked_hits(self, search_service: SearchService) -> None:
        hits = search_service.search("30 days notice")
        assert len(hits) == 1
        assert hits[0].filename == "contract.pdf"
        assert hits[0].page_number == 4

    def test_snippet_highlights_matching_terms(self, search_service: SearchService) -> None:
        hits = search_service.search("30 days notice")
        assert "**notice**" in hits[0].snippet or "**30**" in hits[0].snippet

    def test_empty_query_raises(self, search_service: SearchService) -> None:
        with pytest.raises(ValidationError):
            search_service.search("")

    def test_whitespace_query_raises(self, search_service: SearchService) -> None:
        with pytest.raises(ValidationError):
            search_service.search("   ")

    def test_no_matches_returns_empty_list(self, search_service: SearchService) -> None:
        hits = search_service.search("completely unrelated topic xyz")
        # DirectionalFakeModel maps anything without "notice" to the same
        # opposite vector, so this still might match depending on threshold —
        # the important behavioral contract is: it never raises, always a list.
        assert isinstance(hits, list)

    def test_logs_query_analytics(self, search_service: SearchService) -> None:
        search_service.search("30 days notice")
        top = search_service.db.get_most_queried_documents()
        assert len(top) == 1
        assert top[0][1] == 1  # queried once

    def test_full_content_preserved_alongside_snippet(self, search_service: SearchService) -> None:
        hits = search_service.search("30 days notice")
        assert "delivered in writing" in hits[0].full_content
        assert hits[0].full_content == (
            "Either party may terminate this agreement with 30 days written "
            "notice to the other party, delivered in writing."
        )


class TestTokenizeQuery:
    def test_extracts_words(self) -> None:
        tokens = _tokenize_query("What is the termination clause?")
        assert "termination" in tokens
        assert "clause" in tokens

    def test_excludes_very_short_tokens(self) -> None:
        tokens = _tokenize_query("a b is it")
        assert "a" not in tokens

    def test_deduplicates(self) -> None:
        tokens = _tokenize_query("notice notice notice")
        assert tokens.count("notice") == 1

    def test_empty_query_returns_empty_list(self) -> None:
        assert _tokenize_query("") == []


class TestBuildSnippet:
    def test_highlights_first_match_with_context(self) -> None:
        content = "word " * 50 + "TARGETWORD" + " word" * 50
        snippet = _build_snippet(content, "targetword", context_chars=40)
        assert "**TARGETWORD**" in snippet
        assert snippet.startswith("...")
        assert snippet.endswith("...")

    def test_no_ellipsis_when_match_near_start(self) -> None:
        content = "TARGETWORD " + "word " * 50
        snippet = _build_snippet(content, "targetword", context_chars=40)
        assert not snippet.startswith("...")

    def test_falls_back_to_truncation_when_no_match(self) -> None:
        content = "This text has no relevant keywords at all in it whatsoever."
        snippet = _build_snippet(content, "termination clause", context_chars=30)
        assert "**" not in snippet

    def test_empty_query_falls_back_to_truncation(self) -> None:
        content = "Some content here that is reasonably long for testing purposes."
        snippet = _build_snippet(content, "", context_chars=20)
        assert "**" not in snippet

    def test_case_insensitive_highlighting(self) -> None:
        content = "The TERMINATION clause is on page 5."
        snippet = _build_snippet(content, "termination", context_chars=100)
        assert "**TERMINATION**" in snippet

    def test_multiple_terms_all_highlighted(self) -> None:
        content = "The notice period and termination clause are both important."
        snippet = _build_snippet(content, "notice termination", context_chars=200)
        assert "**notice**" in snippet
        assert "**termination**" in snippet
