"""
embedding_service.py
=====================
Wraps a Sentence-Transformers model (default: BAAI/bge-base-en-v1.5)
to convert text into dense vector embeddings for semantic search.

Design notes
------------
- **Lazy model loading**: the transformer model is only loaded from
  disk/network the first time `embed_texts()` or `embed_query()` is
  called, not at construction time. In a Streamlit app this matters —
  instantiating `EmbeddingService()` happens on every page render, and
  eagerly loading a ~400MB model on every rerun would make the UI
  unusable.
- **Dependency injection for the model**: `EmbeddingService(model=...)`
  accepts a pre-built model object (anything with a Sentence-
  Transformers-compatible `.encode()` signature). This is what makes
  the service unit-testable without downloading real weights, and is
  also how a future swap to a different embedding backend (OpenAI
  embeddings, Cohere, etc.) would be wired in without touching
  `document_service.py` or `retriever.py`.
- **Batching**: `embed_texts()` always calls the model's own internal
  batching via `batch_size`, rather than looping chunk-by-chunk in
  Python — this is the difference between embedding 500 chunks in a
  few seconds vs. one API/model call per chunk.
"""

from __future__ import annotations

from typing import Any, Protocol

from config import settings
from src.utils.exceptions import EmbeddingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EncoderModel(Protocol):
    """
    Structural type describing the subset of the Sentence-Transformers
    `SentenceTransformer` API this service depends on. Any object
    matching this shape (real model, fake test double, or a future
    alternative embedding backend) can be injected.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> Any: ...

    def get_sentence_embedding_dimension(self) -> int: ...


class EmbeddingService:
    """Generates dense vector embeddings for document chunks and queries."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
        model: EncoderModel | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding.model_name
        self.device = device or settings.embedding.device
        self.batch_size = batch_size or settings.embedding.batch_size
        self.normalize = settings.embedding.normalize_embeddings if normalize is None else normalize

        # Injected model bypasses lazy loading entirely — used in tests
        # and for swapping embedding backends without code changes.
        self._model: EncoderModel | None = model

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def _get_model(self) -> EncoderModel:
        """Lazily load the Sentence-Transformers model on first use."""
        if self._model is None:
            try:
                # Imported here, not at module level, so importing this
                # module never requires `sentence-transformers` (and its
                # heavy `torch` dependency) to be installed unless a
                # real model is actually needed at runtime.
                from sentence_transformers import SentenceTransformer

                logger.info(
                    "Loading embedding model '%s' on device '%s'...",
                    self.model_name, self.device,
                )
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info("Embedding model '%s' loaded successfully", self.model_name)
            except Exception as exc:
                logger.error("Failed to load embedding model '%s': %s", self.model_name, exc)
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Return the output vector dimension of the loaded model."""
        try:
            return self._get_model().get_sentence_embedding_dimension()
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Could not determine embedding dimension: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts (e.g. document chunks).

        Args:
            texts: List of text strings to embed. Empty strings are
                rejected upstream by the splitter, but this method
                defends against them anyway.

        Returns:
            A list of embedding vectors (list of floats), same length
            and order as `texts`.

        Raises:
            EmbeddingError: If the model fails to load or encode.
        """
        if not texts:
            return []

        cleaned = [t if t and t.strip() else " " for t in texts]

        try:
            model = self._get_model()
            vectors = model.encode(
                cleaned,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embeddings = [vector.tolist() for vector in vectors]
            logger.info(
                "Embedded %d texts (dim=%d, batch_size=%d)",
                len(embeddings),
                len(embeddings[0]) if embeddings else 0,
                self.batch_size,
            )
            return embeddings
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc, exc_info=True)
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        """
        Generate a single embedding for a user query string.

        Kept as a distinct method from `embed_texts` (even though it
        delegates to it) because query embedding is a distinct concept
        in the RAG pipeline — some embedding models use different
        prefixes/instructions for queries vs. documents (BGE models in
        particular support a query instruction prefix), giving us a
        single seam to add that later without touching call sites.
        """
        if not query or not query.strip():
            raise EmbeddingError("Cannot embed an empty query string.")

        return self.embed_texts([query])[0]
