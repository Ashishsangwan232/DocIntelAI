"""
models.py
=========
Typed dataclass entities shared across every layer of the application
(loaders, services, RAG pipeline, database, UI).

These are plain Python dataclasses — deliberately not tied to SQLite,
ChromaDB, or Streamlit — so the same model can be constructed by a
loader, persisted by `sqlite_manager.py`, embedded by
`embedding_service.py`, and rendered by `ui/*.py` without any layer
depending on another layer's implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.utils.helpers import generate_uuid, utc_now


class DocumentStatus(str, Enum):
    """Lifecycle states of an uploaded document."""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class FileType(str, Enum):
    """Supported document formats."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class MessageRole(str, Enum):
    """Role of a message within a chat session."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Collection:
    """A named grouping of related documents (e.g. 'Q3 Contracts')."""

    name: str
    description: str | None = None
    id: str = field(default_factory=generate_uuid)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Document:
    """Metadata for an uploaded document, mirrored in SQLite."""

    filename: str
    file_type: FileType
    file_size_bytes: int
    file_hash: str
    id: str = field(default_factory=generate_uuid)
    collection_id: str | None = None
    page_count: int | None = None
    status: DocumentStatus = DocumentStatus.PROCESSING
    uploaded_at: datetime = field(default_factory=utc_now)
    processed_at: datetime | None = None

    def mark_ready(self, page_count: int | None = None) -> None:
        """Transition the document into the READY state."""
        self.status = DocumentStatus.READY
        self.processed_at = utc_now()
        if page_count is not None:
            self.page_count = page_count

    def mark_failed(self) -> None:
        """Transition the document into the FAILED state."""
        self.status = DocumentStatus.FAILED
        self.processed_at = utc_now()


@dataclass
class Chunk:
    """
    A single retrievable unit of text extracted from a document.

    The `id` here is intentionally the same ID used as the vector ID
    in ChromaDB, so the relational row and the vector can always be
    joined without a separate mapping table.
    """

    document_id: str
    chunk_index: int
    content: str
    id: str = field(default_factory=generate_uuid)
    page_number: int | None = None
    token_count: int | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Citation:
    """
    A reference back to the source chunk that supported part of an
    LLM answer. Returned alongside `ChatMessage.content` and rendered
    as an expandable citation chip in the UI.
    """

    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    similarity_score: float
    excerpt: str
    page_number: int | None = None


@dataclass
class ChatSession:
    """A conversation thread, optionally scoped to a collection."""

    id: str = field(default_factory=generate_uuid)
    collection_id: str | None = None
    title: str = "New Conversation"
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class ChatMessage:
    """A single message (user or assistant) within a chat session."""

    session_id: str
    role: MessageRole
    content: str
    id: str = field(default_factory=generate_uuid)
    citations: list[Citation] = field(default_factory=list)
    response_time_ms: int | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class QueryLog:
    """A record of a single semantic search / chat query for analytics."""

    query_text: str
    id: str = field(default_factory=generate_uuid)
    document_id: str | None = None
    top_score: float | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class SearchResult:
    """A single ranked result returned by the semantic search service."""

    chunk: Chunk
    document: Document
    similarity_score: float
