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

`FakeHTTPSession`/`FakeHTTPResponse` play the same role for
`OllamaCloudLLM` — they satisfy the exact `HTTPSession`/`HTTPResponse`
structural types the LLM client depends on, with zero real network
calls to Ollama Cloud.
"""

from __future__ import annotations

import json
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


class FakeHTTPResponse:
    """Fake stand-in for a `requests.Response` used by `OllamaCloudLLM`."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        lines: Iterable[bytes] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self._lines = list(lines) if lines is not None else []
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._json_data

    def iter_lines(self):
        return iter(self._lines)


class FakeHTTPSession:
    """
    Fake stand-in for `requests.Session` used by `OllamaCloudLLM`.

    Records the last call's arguments (`last_call`) so tests can assert
    on exactly what payload/headers/URL were sent, and can be
    configured with either a fixed response or an exception to raise.
    """

    def __init__(
        self,
        response: FakeHTTPResponse | None = None,
        raise_exception: Exception | None = None,
    ) -> None:
        self.response = response
        self.raise_exception = raise_exception
        self.last_call: dict[str, Any] | None = None
        self.call_count = 0

    def post(self, url: str, *, headers: dict, json: dict, timeout: int, stream: bool = False):
        self.call_count += 1
        self.last_call = {
            "url": url, "headers": headers, "json": json,
            "timeout": timeout, "stream": stream,
        }
        if self.raise_exception is not None:
            raise self.raise_exception
        return self.response


def make_stream_lines(*chunks: str, final_done: bool = True) -> list[bytes]:
    """Helper to build Ollama-style newline-delimited JSON stream chunks."""
    lines = [
        json.dumps({"response": chunk, "done": False}).encode() for chunk in chunks
    ]
    if final_done:
        lines.append(json.dumps({"response": "", "done": True}).encode())
    return lines

