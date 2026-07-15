"""
Unit tests for src/rag/pipeline.py

Uses a minimal `FakeLLM(BaseLLM)` test double rather than
`OllamaCloudLLM` + fake HTTP session — `RAGPipeline` only depends on
the `BaseLLM` interface, so testing against that interface directly
(rather than a specific implementation's transport details) is the
correct unit-test boundary.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from src.database.models import Chunk, Document, FileType
from src.llm.base import BaseLLM, LLMResponse
from src.rag.pipeline import RAGPipeline
from src.rag.prompt_builder import NO_CONTEXT_MESSAGE, PromptBuilder
from src.rag.retriever import RetrievalResult, Retriever
from src.utils.exceptions import LLMResponseError
from src.vectorstore.chroma_manager import VectorSearchResult


class FakeLLM(BaseLLM):
    """Minimal `BaseLLM` test double with call tracking."""

    def __init__(self, response_text: str = "Fake answer.", stream_chunks: list[str] | None = None):
        self.response_text = response_text
        self.stream_chunks = stream_chunks or ["Fake ", "streamed ", "answer."]
        self.generate_call_count = 0
        self.stream_call_count = 0
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None, model=None) -> LLMResponse:
        self.generate_call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return LLMResponse(content=self.response_text, model="fake-model", latency_ms=10)

    def stream(self, prompt, system_prompt=None, temperature=None, max_tokens=None, model=None) -> Iterator[str]:
        self.stream_call_count += 1
        self.last_prompt = prompt
        yield from self.stream_chunks


class FakeRetriever:
    """Test double for Retriever returning a pre-configured RetrievalResult."""

    def __init__(self, result: RetrievalResult):
        self._result = result
        self.retrieve_call_count = 0

    def retrieve(self, query, top_k=None, document_ids=None) -> RetrievalResult:
        self.retrieve_call_count += 1
        return self._result


def _make_chunk_result(content: str = "relevant content", score: float = 0.9) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id="c1", document_id="d1", filename="doc.pdf",
        chunk_index=0, page_number=1, content=content, similarity_score=score,
    )


class TestAskWithContext:
    def test_calls_llm_and_returns_answer_with_citations(self) -> None:
        retrieval = RetrievalResult(query="q", chunks=[_make_chunk_result()])
        fake_llm = FakeLLM(response_text="The answer is in the document.")
        pipeline = RAGPipeline(
            retriever=FakeRetriever(retrieval), prompt_builder=PromptBuilder(), llm=fake_llm
        )

        result = pipeline.ask("what is the answer?")

        assert result.answer == "The answer is in the document."
        assert result.used_context is True
        assert result.retrieved_chunk_count == 1
        assert len(result.citations) == 1
        assert fake_llm.generate_call_count == 1

    def test_prompt_passed_to_llm_includes_context(self) -> None:
        retrieval = RetrievalResult(query="q", chunks=[_make_chunk_result("special unique phrase")])
        fake_llm = FakeLLM()
        pipeline = RAGPipeline(retriever=FakeRetriever(retrieval), prompt_builder=PromptBuilder(), llm=fake_llm)

        pipeline.ask("test question")
        assert "special unique phrase" in fake_llm.last_prompt
        assert fake_llm.last_system_prompt is not None


class TestAskWithoutContext:
    def test_skips_llm_call_entirely(self) -> None:
        retrieval = RetrievalResult(query="q", chunks=[])
        fake_llm = FakeLLM()
        pipeline = RAGPipeline(retriever=FakeRetriever(retrieval), prompt_builder=PromptBuilder(), llm=fake_llm)

        result = pipeline.ask("unrelated question")

        assert result.answer == NO_CONTEXT_MESSAGE
        assert result.used_context is False
        assert result.citations == []
        assert result.retrieved_chunk_count == 0
        assert fake_llm.generate_call_count == 0  # LLM never called


class TestAskErrorPropagation:
    def test_llm_errors_propagate_to_caller(self) -> None:
        class FailingLLM(FakeLLM):
            def generate(self, *args, **kwargs):
                raise LLMResponseError("simulated failure")

        retrieval = RetrievalResult(query="q", chunks=[_make_chunk_result()])
        pipeline = RAGPipeline(
            retriever=FakeRetriever(retrieval), prompt_builder=PromptBuilder(), llm=FailingLLM()
        )
        with pytest.raises(LLMResponseError):
            pipeline.ask("question")


class TestAskStreaming:
    def test_returns_citations_immediately_and_streams_tokens(self) -> None:
        retrieval = RetrievalResult(query="q", chunks=[_make_chunk_result()])
        fake_llm = FakeLLM(stream_chunks=["Hello", " world"])
        pipeline = RAGPipeline(retriever=FakeRetriever(retrieval), prompt_builder=PromptBuilder(), llm=fake_llm)

        stream, citations = pipeline.ask_streaming("question")

        assert len(citations) == 1  # available before consuming the stream
        collected = list(stream)
        assert collected == ["Hello", " world"]
        assert fake_llm.stream_call_count == 1

    def test_no_context_returns_fallback_stream_and_empty_citations(self) -> None:
        retrieval = RetrievalResult(query="q", chunks=[])
        fake_llm = FakeLLM()
        pipeline = RAGPipeline(retriever=FakeRetriever(retrieval), prompt_builder=PromptBuilder(), llm=fake_llm)

        stream, citations = pipeline.ask_streaming("unrelated")

        assert citations == []
        assert list(stream) == [NO_CONTEXT_MESSAGE]
        assert fake_llm.stream_call_count == 0  # LLM never called
