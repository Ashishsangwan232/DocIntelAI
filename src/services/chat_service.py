"""
chat_service.py
================
Orchestrates the chat experience: wires `RAGPipeline` to persisted
`ChatSession`/`ChatMessage` records and query analytics logging.

`ui/chat.py` (Streamlit) calls this class exclusively — it never talks
to `RAGPipeline`, `Retriever`, or the LLM directly. This keeps the UI
layer free of business logic and is what makes a future Streamlit ->
FastAPI migration a matter of wrapping these same methods in HTTP
routes, not rewriting them.

Every user question and assistant answer is persisted immediately
(not just held in Streamlit session state), so conversation history
survives page reloads and app restarts.
"""

from __future__ import annotations

import time
from typing import Iterator

from src.database.models import ChatMessage, ChatSession, MessageRole, QueryLog
from src.database.sqlite_manager import SQLiteManager
from src.rag.pipeline import RAGPipeline
from src.utils.exceptions import ValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_SESSION_TITLE = "New Conversation"
_TITLE_PREVIEW_LENGTH = 50


class ChatService:
    """Coordinates chat sessions, RAG-backed responses, and analytics logging."""

    def __init__(
        self,
        db: SQLiteManager | None = None,
        rag_pipeline: RAGPipeline | None = None,
    ) -> None:
        self.db = db or SQLiteManager()
        self.rag_pipeline = rag_pipeline or RAGPipeline()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def create_session(
        self, title: str = _DEFAULT_SESSION_TITLE, collection_id: str | None = None
    ) -> ChatSession:
        """Start a new, empty chat session."""
        session = ChatSession(title=title, collection_id=collection_id)
        return self.db.create_chat_session(session)

    def get_session(self, session_id: str) -> ChatSession:
        """
        Fetch a single chat session by ID.

        Streamlit's chat page never needed this — it only ever acts on
        a session ID it created and holds in `st.session_state`. The
        API layer accepts a session ID from an arbitrary request, so it
        uses this to turn an invalid ID into a clean 404 before calling
        `send_message`/`regenerate_last_response`/etc., rather than
        letting a bad ID surface as a raw FOREIGN KEY constraint error
        from the messages INSERT.

        Raises:
            RecordNotFoundError: If no session with this ID exists.
        """
        return self.db.get_chat_session(session_id)

    def list_sessions(self) -> list[ChatSession]:
        """Return all chat sessions, most recently created first."""
        return self.db.list_chat_sessions()

    def get_messages(self, session_id: str) -> list[ChatMessage]:
        """Return the full message history for a session, chronological order."""
        return self.db.get_messages_for_session(session_id)

    def clear_conversation(self, session_id: str) -> None:
        """Delete all messages in a session but keep the session itself."""
        self.db.clear_session_messages(session_id)

    def delete_session(self, session_id: str) -> None:
        """Permanently delete a chat session and all its messages."""
        self.db.delete_chat_session(session_id)

    # ------------------------------------------------------------------
    # Sending messages
    # ------------------------------------------------------------------
    def send_message(
        self,
        session_id: str,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> ChatMessage:
        """
        Send a user message and return the persisted assistant reply
        (non-streaming — the full RAG round trip completes before
        returning).

        Args:
            session_id: The chat session this message belongs to.
            query: The user's question.
            document_ids: Optional filter to scope retrieval to specific
                documents (e.g. chatting with a single document).
            top_k: Overrides the configured retrieval top-K.
            temperature: Overrides the configured LLM temperature.
            max_tokens: Overrides the configured LLM max token count.
            model: Overrides the configured LLM model.

        Returns:
            The persisted assistant `ChatMessage`, including citations.

        Raises:
            ValidationError: If `query` is empty.
            RetrievalError: If query embedding or vector search fails outright.
            LLMError: If the LLM call fails.
        """
        self._validate_query(query)
        self._persist_user_message(session_id, query)

        rag_response = self.rag_pipeline.ask(
            query,
            top_k=top_k,
            document_ids=document_ids,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

        assistant_message = ChatMessage(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=rag_response.answer,
            citations=rag_response.citations,
            response_time_ms=rag_response.response_time_ms,
        )
        self.db.add_chat_message(assistant_message)
        self._log_query_analytics(query, rag_response.citations)

        logger.info(
            "Chat message answered in session %s (%d citations, %dms)",
            session_id, len(rag_response.citations), rag_response.response_time_ms,
        )
        return assistant_message

    def send_message_streaming(
        self,
        session_id: str,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[Iterator[str], list]:
        """
        Send a user message and stream the assistant reply token-by-token.

        The user message is persisted immediately. The assistant message
        is persisted automatically once the returned stream is fully
        consumed — `ui/chat.py` should iterate the stream to completion
        (e.g. via `st.write_stream`) rather than partially consuming it,
        or the assistant's turn will never be saved.

        Returns:
            A tuple of `(token_stream, citations)`. Citations are
            available immediately; `token_stream` yields text chunks.

        Raises:
            ValidationError: If `query` is empty.
            RetrievalError: If query embedding or vector search fails outright.
        """
        self._validate_query(query)
        self._persist_user_message(session_id, query)

        token_stream, citations = self.rag_pipeline.ask_streaming(
            query,
            top_k=top_k,
            document_ids=document_ids,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

        def _persisting_stream() -> Iterator[str]:
            start = time.monotonic()
            collected: list[str] = []
            for chunk in token_stream:
                collected.append(chunk)
                yield chunk

            full_answer = "".join(collected)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            assistant_message = ChatMessage(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=full_answer,
                citations=citations,
                response_time_ms=elapsed_ms,
            )
            self.db.add_chat_message(assistant_message)
            self._log_query_analytics(query, citations)
            logger.info(
                "Streamed chat message answered in session %s (%d citations, %dms)",
                session_id, len(citations), elapsed_ms,
            )

        return _persisting_stream(), citations

    def regenerate_last_response(
        self,
        session_id: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> ChatMessage:
        """
        Re-run the last user question in a session, replacing the
        previous assistant answer with a freshly generated one.

        Raises:
            ValidationError: If the session has no user message to regenerate.
        """
        messages = self.db.get_messages_for_session(session_id)
        if not messages:
            raise ValidationError("Cannot regenerate: this conversation has no messages yet.")

        last_message = messages[-1]
        if last_message.role == MessageRole.ASSISTANT:
            self.db.delete_chat_message(last_message.id)
            remaining = messages[:-1]
        else:
            remaining = messages

        if not remaining or remaining[-1].role != MessageRole.USER:
            raise ValidationError("Cannot regenerate: no prior user question found.")

        last_user_query = remaining[-1].content

        rag_response = self.rag_pipeline.ask(
            last_user_query,
            top_k=top_k,
            document_ids=document_ids,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        assistant_message = ChatMessage(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=rag_response.answer,
            citations=rag_response.citations,
            response_time_ms=rag_response.response_time_ms,
        )
        self.db.add_chat_message(assistant_message)
        self._log_query_analytics(last_user_query, rag_response.citations)

        logger.info("Regenerated response for session %s", session_id)
        return assistant_message

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_query(self, query: str) -> None:
        if not query or not query.strip():
            raise ValidationError("Message cannot be empty.")

    def _persist_user_message(self, session_id: str, query: str) -> ChatMessage:
        user_message = ChatMessage(session_id=session_id, role=MessageRole.USER, content=query)
        return self.db.add_chat_message(user_message)

    def _log_query_analytics(self, query: str, citations: list) -> None:
        top_citation = citations[0] if citations else None
        self.db.log_query(
            QueryLog(
                query_text=query,
                document_id=top_citation.document_id if top_citation else None,
                top_score=top_citation.similarity_score if top_citation else None,
            )
        )

    @staticmethod
    def derive_session_title(first_message: str) -> str:
        """
        Generate a short, human-readable session title from the first
        user message — used by the UI to auto-title new conversations
        (e.g. in a session list sidebar) instead of showing "New
        Conversation" for every entry.
        """
        cleaned = " ".join(first_message.strip().split())
        if len(cleaned) <= _TITLE_PREVIEW_LENGTH:
            return cleaned
        return cleaned[:_TITLE_PREVIEW_LENGTH].rstrip() + "..."
