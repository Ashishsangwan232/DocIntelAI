"""
Unit tests for src/embeddings/embedding_service.py

Uses `FakeEmbeddingModel` (see tests/conftest.py) instead of a real
Sentence-Transformers model — this environment has no network access
to Hugging Face, and even with access, downloading real weights on
every test run would be slow. The fake preserves the exact `.encode()`
contract the real model exposes, so every line of orchestration logic
in `EmbeddingService` is still genuinely exercised.
"""

from __future__ import annotations

import pytest

from src.embeddings.embedding_service import EmbeddingService
from src.utils.exceptions import EmbeddingError
from tests.conftest import FakeEmbeddingModel


@pytest.fixture()
def service(fake_embedding_model: FakeEmbeddingModel) -> EmbeddingService:
    return EmbeddingService(model=fake_embedding_model)


class TestEmbedTexts:
    def test_returns_one_vector_per_input(self, service: EmbeddingService) -> None:
        vectors = service.embed_texts(["one", "two", "three"])
        assert len(vectors) == 3
        assert all(len(v) == 16 for v in vectors)

    def test_empty_list_returns_empty_list(self, service: EmbeddingService) -> None:
        assert service.embed_texts([]) == []

    def test_same_text_produces_same_vector(self, service: EmbeddingService) -> None:
        v1 = service.embed_texts(["contract termination clause"])[0]
        v2 = service.embed_texts(["contract termination clause"])[0]
        assert v1 == v2

    def test_different_text_produces_different_vector(self, service: EmbeddingService) -> None:
        v1 = service.embed_texts(["contract termination"])[0]
        v2 = service.embed_texts(["completely unrelated topic"])[0]
        assert v1 != v2

    def test_blank_strings_do_not_crash(self, service: EmbeddingService) -> None:
        vectors = service.embed_texts(["", "   ", "real text"])
        assert len(vectors) == 3

    def test_wraps_encoder_failures_as_embedding_error(self) -> None:
        class FailingModel:
            def encode(self, *args, **kwargs):
                raise RuntimeError("backend unavailable")

            def get_sentence_embedding_dimension(self) -> int:
                return 16

        svc = EmbeddingService(model=FailingModel())
        with pytest.raises(EmbeddingError):
            svc.embed_texts(["some text"])


class TestEmbedQuery:
    def test_returns_single_vector(self, service: EmbeddingService) -> None:
        vector = service.embed_query("what is the termination clause?")
        assert isinstance(vector, list)
        assert len(vector) == 16

    def test_empty_query_raises(self, service: EmbeddingService) -> None:
        with pytest.raises(EmbeddingError):
            service.embed_query("")

    def test_whitespace_only_query_raises(self, service: EmbeddingService) -> None:
        with pytest.raises(EmbeddingError):
            service.embed_query("   ")


class TestEmbeddingDimension:
    def test_matches_fake_model_dimension(self, service: EmbeddingService) -> None:
        assert service.embedding_dimension == 16

    def test_dimension_failure_wrapped_as_embedding_error(self) -> None:
        class BrokenDimensionModel:
            def encode(self, *args, **kwargs):
                return []

            def get_sentence_embedding_dimension(self):
                raise RuntimeError("model not loaded")

        svc = EmbeddingService(model=BrokenDimensionModel())
        with pytest.raises(EmbeddingError):
            _ = svc.embedding_dimension


class TestConfiguration:
    def test_defaults_come_from_settings(self) -> None:
        svc = EmbeddingService(model=FakeEmbeddingModel())
        assert svc.model_name  # non-empty, sourced from config
        assert svc.batch_size > 0

    def test_explicit_overrides_take_precedence(self) -> None:
        svc = EmbeddingService(
            model=FakeEmbeddingModel(), model_name="custom-model", batch_size=7, normalize=False
        )
        assert svc.model_name == "custom-model"
        assert svc.batch_size == 7
        assert svc.normalize is False
