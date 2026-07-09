"""
base.py
=======
Abstract interface for LLM providers.

`OllamaCloudLLM` implements this interface today. Swapping to a
different provider (OpenAI, Anthropic, a local Ollama instance) later
means writing one new `BaseLLM` subclass — `rag/pipeline.py` and every
other caller depends only on this interface, never on a specific
provider's SDK or request/response shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class LLMResponse:
    """Unified response shape returned by any `BaseLLM` implementation."""

    content: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class BaseLLM(ABC):
    """Abstract interface every LLM provider implementation must satisfy."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """
        Generate a complete response for a prompt (non-streaming).

        Args:
            prompt: The user-facing prompt (typically includes RAG context).
            system_prompt: Optional system-level instruction.
            temperature: Overrides the configured default if provided.
            max_tokens: Overrides the configured default if provided.
            model: Overrides the configured default model if provided
                (e.g. from a user-selected Settings preference).

        Returns:
            An `LLMResponse` with the full generated content.

        Raises:
            LLMTimeoutError: If the provider does not respond in time.
            LLMAuthenticationError: If the API key is rejected.
            LLMResponseError: If the provider returns a malformed response.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        """
        Generate a response as an incremental stream of text tokens/chunks.

        Yields:
            Successive text chunks as they are generated. The caller is
            responsible for concatenating them for the full response.

        Raises:
            LLMTimeoutError: If the provider does not respond in time.
            LLMAuthenticationError: If the API key is rejected.
            LLMResponseError: If the provider returns a malformed response.
        """
        raise NotImplementedError
