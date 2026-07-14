"""
schemas/chat.py
================
Request/response models for the Chat API. `CitationResponse` and
`ChatMessageResponse` mirror `src.database.models.Citation`/
`ChatMessage` field-for-field — the frontend's citation panel should
be able to render directly from these without any renaming.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.database.models import ChatMessage, ChatSession, Citation, MessageRole


class CitationResponse(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    similarity_score: float
    excerpt: str
    page_number: int | None = None

    @classmethod
    def from_citation(cls, citation: Citation) -> "CitationResponse":
        return cls(
            document_id=citation.document_id,
            filename=citation.filename,
            chunk_id=citation.chunk_id,
            chunk_index=citation.chunk_index,
            similarity_score=citation.similarity_score,
            excerpt=citation.excerpt,
            page_number=citation.page_number,
        )


class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    role: MessageRole
    content: str
    citations: list[CitationResponse] = Field(default_factory=list)
    response_time_ms: int | None = None
    created_at: datetime

    @classmethod
    def from_message(cls, message: ChatMessage) -> "ChatMessageResponse":
        return cls(
            id=message.id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            citations=[CitationResponse.from_citation(c) for c in message.citations],
            response_time_ms=message.response_time_ms,
            created_at=message.created_at,
        )


class ChatSessionResponse(BaseModel):
    id: str
    collection_id: str | None = None
    title: str
    created_at: datetime

    @classmethod
    def from_session(cls, session: ChatSession) -> "ChatSessionResponse":
        return cls(
            id=session.id,
            collection_id=session.collection_id,
            title=session.title,
            created_at=session.created_at,
        )


class CreateSessionRequest(BaseModel):
    title: str = "New Conversation"
    collection_id: str | None = None


class SendMessageRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    document_ids: list[str] | None = Field(
        default=None, description="Optional filter to scope retrieval to specific documents."
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8000)
    model: str | None = None


class RegenerateRequest(BaseModel):
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8000)
    model: str | None = None
