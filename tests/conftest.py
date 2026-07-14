"""
conftest.py
===========
Shared pytest fixtures.

`FakeEmbeddingModel` stands in for the real Sentence-Transformers
model in tests: it's deterministic (same text -> same vector),
respects the same `.encode()` signature, and requires no network
access or multi-hundred-MB model download. This is what keeps the
test suite fast (seconds, not minutes) while still exercising every
real line of `EmbeddingService`, `ChromaManager`, and
`DocumentService` orchestration logic.

`FakeOllamaChatClient` plays the same role for `OllamaCloudLLM` — it
satisfies the exact `ChatClient` structural type the LLM class depends
on, with zero real network calls to Ollama Cloud.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pytest


class FakeEmbeddingModel:
    """
    Deterministic stand-in for `sentence_transformers.SentenceTransformer`.

    Produces a fixed-size vector per input string derived from a hash
    of the text, so identical inputs always produce identical vectors
    (mirrors real embedding model determinism) without loading any
    real weights.
    """

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def encode(
        self,
        sentences: list[str],
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ):
        vectors = []
        for text in sentences:
            seed = abs(hash(text)) % (2**31)
            rng = np.random.RandomState(seed)
            vector = rng.rand(self.dimension).astype("float32")
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vectors.append(vector)
        return np.array(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


@pytest.fixture()
def fake_embedding_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel(dimension=16)


class FakeOllamaResponseError(Exception):
    """Stand-in for `ollama.ResponseError` — carries a real status code."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.error = message
        self.status_code = status_code


class FakeOllamaChatClient:
    """
    Fake stand-in for `ollama.Client` used by `OllamaCloudLLM`.

    Records the last call's arguments (`last_call`) so tests can assert
    on exactly what model/messages/options were sent, and can be
    configured with either a fixed response (dict, for non-streaming)
    or a list of chunk dicts (for streaming) or an exception to raise.
    """

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        stream_chunks: Iterable[dict[str, Any]] | None = None,
        raise_exception: Exception | None = None,
    ) -> None:
        self.response = response
        self._stream_chunks = list(stream_chunks) if stream_chunks is not None else []
        self.raise_exception = raise_exception
        self.last_call: dict[str, Any] | None = None
        self.call_count = 0

    def chat(
        self, *, model: str, messages: list[dict[str, str]], options: dict[str, Any], stream: bool = False
    ):
        self.call_count += 1
        self.last_call = {"model": model, "messages": messages, "options": options, "stream": stream}
        if self.raise_exception is not None:
            raise self.raise_exception
        if stream:
            return iter(self._stream_chunks)
        return self.response


def make_stream_chunks(*chunks: str, final_done: bool = True) -> list[dict[str, Any]]:
    """Helper to build Ollama-chat-style streaming response chunks."""
    parts: list[dict[str, Any]] = [
        {"message": {"role": "assistant", "content": chunk}, "done": False} for chunk in chunks
    ]
    if final_done:
        parts.append({"message": {"role": "assistant", "content": ""}, "done": True})
    return parts

