"""
exceptions.py
=============
Structured exception hierarchy for DocIntel AI.

Every layer raises a specific, typed exception rather than a bare
`Exception` or `ValueError`. This lets the service layer catch narrow
exception types and translate them into user-friendly UI messages
(Phase 13 error-handling pass) instead of leaking stack traces.
"""

from __future__ import annotations


class DocIntelError(Exception):
    """Base class for all application-specific exceptions."""


# --- Upload / validation ----------------------------------------------------
class ValidationError(DocIntelError):
    """Raised when user input fails validation (e.g. bad file type)."""


class UnsupportedFileTypeError(ValidationError):
    """Raised when an uploaded file extension is not supported."""


class FileTooLargeError(ValidationError):
    """Raised when an uploaded file exceeds the configured size limit."""


class DuplicateDocumentError(ValidationError):
    """Raised when an uploaded file's content hash already exists."""


# --- Document processing ----------------------------------------------------
class DocumentProcessingError(DocIntelError):
    """Raised when text extraction or cleaning fails for a document."""


class CorruptedFileError(DocumentProcessingError):
    """Raised when a file cannot be parsed because it is corrupted."""


# --- Embeddings / vector store ----------------------------------------------
class EmbeddingError(DocIntelError):
    """Raised when embedding generation fails."""


class VectorStoreError(DocIntelError):
    """Raised when a ChromaDB operation fails."""


# --- RAG / LLM ---------------------------------------------------------------
class RetrievalError(DocIntelError):
    """Raised when semantic retrieval fails or returns no usable results."""


class SummaryGenerationError(DocIntelError):
    """Raised when AI summary generation fails or returns unparseable output."""


class LLMError(DocIntelError):
    """Base class for LLM-related failures."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM provider does not respond within the timeout."""


class LLMAuthenticationError(LLMError):
    """Raised when the LLM provider rejects the configured API key."""


class LLMResponseError(LLMError):
    """Raised when the LLM provider returns a malformed or error response."""


# --- Persistence -------------------------------------------------------------
class DatabaseError(DocIntelError):
    """Raised when a SQLite operation fails."""


class RecordNotFoundError(DatabaseError):
    """Raised when a requested record does not exist."""
