"""
pipeline.py
===========
Orchestrates the full RAG flow: retrieve -> build prompt -> call LLM.

This is the single entry point `chat_service.py` (Phase 6) and
`search_service.py`'s AI-summary features call — neither needs to know
about `Retriever`, `PromptBuilder`, or `BaseLLM` individually.

Design notes
------------
- When retrieval returns zero chunks above the similarity threshold,
  the LLM call is skipped entirely and a deterministic "no information
  found" response is returned. This avoids spending tokens/latency on
  a call we already know shouldn't proceed, and avoids the model
  hallucinating an answer from outside the document set.
- `ask_streaming()` returns citations *before* streaming begins,
  because citations depend only on retrieval (which completes before
  any LLM call), not on the LLM's output. This lets the chat UI render
  citation chips immediately while tokens are still streaming in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from src.database.models import Citation
from src.llm.base import BaseLLM
from src.llm.ollama_cloud import OllamaCloudLLM
from src.rag.prompt_builder import NO_CONTEXT_MESSAGE, PromptBuilder
from src.rag.retriever import Retriever
from src.utils.exceptions import LLMError, RetrievalError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Complete result of a single RAG query — answer plus supporting metadata."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved_chunk_count: int = 0
    response_time_ms: int = 0
    used_context: bool = False


class RAGPipeline:
    """Coordinates retrieval, prompt construction, and LLM generation."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm = llm or OllamaCloudLLM()

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> RAGResponse:
        """
        Answer a question using retrieval-augmented generation (non-streaming).

        Args:
            query: The user's natural-language question.
            top_k: Number of chunks to retrieve.
            document_ids: Optional filter to scope retrieval to specific documents.
            temperature: Overrides the configured LLM temperature.
            max_tokens: Overrides the configured LLM max token count.
            model: Overrides the configured LLM model (e.g. a user
                preference from the Settings panel).

        Returns:
            A complete `RAGResponse`.

        Raises:
            RetrievalError: If query embedding or vector search fails outright.
            LLMError: If the LLM call fails (timeout, auth, malformed response).
        """
        start = time.monotonic()
        retrieval = self.retriever.retrieve(query, top_k=top_k, document_ids=document_ids)

        if retrieval.is_empty:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info("No relevant context found for query: '%s'", query[:80])
            return RAGResponse(
                answer=NO_CONTEXT_MESSAGE,
                citations=[],
                retrieved_chunk_count=0,
                response_time_ms=elapsed_ms,
                used_context=False,
            )

        system_prompt, user_prompt = self.prompt_builder.build(query, retrieval)

        try:
            llm_response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            )
        except LLMError:
            logger.error("LLM generation failed for query: '%s'", query[:80])
            raise

        citations = self.prompt_builder.build_citations(retrieval)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "RAG ask() completed in %dms (%d chunks, %d citations)",
            elapsed_ms, len(retrieval.chunks), len(citations),
        )
        return RAGResponse(
            answer=llm_response.content,
            citations=citations,
            retrieved_chunk_count=len(retrieval.chunks),
            response_time_ms=elapsed_ms,
            used_context=True,
        )

    def ask_streaming(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[Iterator[str], list[Citation]]:
        """
        Answer a question using RAG, streaming the answer token-by-token.

        Returns:
            A tuple of `(token_stream, citations)`. Citations are
            available immediately (retrieval completes before this
            method returns); `token_stream` yields text chunks as the
            LLM generates them.

        Raises:
            RetrievalError: If query embedding or vector search fails outright.
        """
        retrieval = self.retriever.retrieve(query, top_k=top_k, document_ids=document_ids)

        if retrieval.is_empty:
            logger.info("No relevant context found for streaming query: '%s'", query[:80])

            def _no_context_stream() -> Iterator[str]:
                yield NO_CONTEXT_MESSAGE

            return _no_context_stream(), []

        system_prompt, user_prompt = self.prompt_builder.build(query, retrieval)
        citations = self.prompt_builder.build_citations(retrieval)

        token_stream = self.llm.stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        return token_stream, citations
