"""Unit tests for src/rag/prompt_builder.py"""

from __future__ import annotations

from src.rag.prompt_builder import NO_CONTEXT_MESSAGE, PromptBuilder
from src.rag.retriever import RetrievalResult
from src.vectorstore.chroma_manager import VectorSearchResult


def _make_result(content: str, filename: str = "doc.pdf", page: int | None = 1, score: float = 0.9):
    return VectorSearchResult(
        chunk_id="chunk-1", document_id="doc-1", filename=filename,
        chunk_index=0, page_number=page, content=content, similarity_score=score,
    )


class TestBuildPrompt:
    def setup_method(self) -> None:
        self.builder = PromptBuilder()

    def test_empty_retrieval_produces_no_context_instruction(self) -> None:
        retrieval = RetrievalResult(query="what is X?", chunks=[])
        system_prompt, user_prompt = self.builder.build("what is X?", retrieval)
        assert NO_CONTEXT_MESSAGE in user_prompt
        assert "what is X?" in user_prompt

    def test_populated_retrieval_includes_context_and_question(self) -> None:
        retrieval = RetrievalResult(
            query="termination?",
            chunks=[_make_result("30 days notice required.", filename="contract.pdf", page=4)],
        )
        system_prompt, user_prompt = self.builder.build("termination?", retrieval)
        assert "30 days notice required." in user_prompt
        assert "contract.pdf" in user_prompt
        assert "page 4" in user_prompt
        assert "termination?" in user_prompt

    def test_multiple_chunks_numbered_as_sources(self) -> None:
        retrieval = RetrievalResult(
            query="q",
            chunks=[
                _make_result("first chunk", filename="a.pdf"),
                _make_result("second chunk", filename="b.pdf"),
            ],
        )
        _, user_prompt = self.builder.build("q", retrieval)
        assert "[Source 1: a.pdf" in user_prompt
        assert "[Source 2: b.pdf" in user_prompt

    def test_none_page_number_omits_page_suffix(self) -> None:
        retrieval = RetrievalResult(
            query="q", chunks=[_make_result("content", filename="doc.docx", page=None)]
        )
        _, user_prompt = self.builder.build("q", retrieval)
        assert "doc.docx]" in user_prompt or "doc.docx\n" in user_prompt
        assert "page None" not in user_prompt

    def test_system_prompt_instructs_citation_and_grounding(self) -> None:
        retrieval = RetrievalResult(query="q", chunks=[_make_result("content")])
        system_prompt, _ = self.builder.build("q", retrieval)
        assert "ONLY" in system_prompt
        assert "cite" in system_prompt.lower()


class TestBuildCitations:
    def setup_method(self) -> None:
        self.builder = PromptBuilder()

    def test_empty_retrieval_produces_no_citations(self) -> None:
        retrieval = RetrievalResult(query="q", chunks=[])
        assert self.builder.build_citations(retrieval) == []

    def test_citation_fields_mapped_correctly(self) -> None:
        retrieval = RetrievalResult(
            query="q",
            chunks=[_make_result("some excerpt text", filename="policy.pdf", page=9, score=0.83)],
        )
        citations = self.builder.build_citations(retrieval)
        assert len(citations) == 1
        c = citations[0]
        assert c.filename == "policy.pdf"
        assert c.page_number == 9
        assert c.similarity_score == 0.83
        assert c.chunk_id == "chunk-1"
        assert "some excerpt text" in c.excerpt

    def test_long_excerpt_is_truncated(self) -> None:
        long_text = "word " * 200
        retrieval = RetrievalResult(query="q", chunks=[_make_result(long_text)])
        citations = self.builder.build_citations(retrieval)
        assert len(citations[0].excerpt) <= 230  # truncation length + suffix buffer
        assert citations[0].excerpt.endswith("...")

    def test_citation_count_matches_chunk_count(self) -> None:
        retrieval = RetrievalResult(
            query="q",
            chunks=[_make_result("a"), _make_result("b"), _make_result("c")],
        )
        assert len(self.builder.build_citations(retrieval)) == 3
