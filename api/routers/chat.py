"""
routers/chat.py
================
REST + SSE endpoints for chat sessions and RAG-backed messaging — a
thin wrapper around `ChatService`, which already does 100% of the
real work (persistence, RAG orchestration, analytics logging).

Streaming design
-----------------
`POST /sessions/{id}/messages/stream` uses Server-Sent Events rather
than a WebSocket: chat is one-directional (server -> client) once the
question is sent, and SSE is trivial to consume with a hand-rolled
`fetch` + `ReadableStream` reader in vanilla JS — no handshake/framing
complexity a WebSocket would add for no benefit here.

`ChatService.send_message_streaming()` returns `(token_stream,
citations)` where citations are already known *before* the first
token is yielded (retrieval happens before generation starts). So the
event order is:

    event: citations   (sent once, immediately)
    event: token        (repeated, one per generated chunk)
    event: error         (only if the LLM call fails mid-stream)
    event: done          (always sent last, whether or not an error occurred)

Once the HTTP response has started (status 200, `text/event-stream`),
the status code can no longer change — so any `DocIntelError` raised
while iterating `token_stream` (e.g. an LLM timeout partway through)
is caught *inside* the generator and reported as an `event: error`
frame instead of propagating to the global exception handler, which
would otherwise just kill the connection with no client-visible
reason.
"""

from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_chat_service, get_settings_service
from api.schemas.chat import (
    ChatMessageResponse,
    ChatSessionResponse,
    CitationResponse,
    CreateSessionRequest,
    RegenerateRequest,
    SendMessageRequest,
)
from src.services.chat_service import ChatService
from src.services.settings_service import RuntimeSettingsService
from src.utils.exceptions import DocIntelError
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _apply_preference_defaults(
    settings_service: RuntimeSettingsService,
    top_k: int | None,
    temperature: float | None,
    max_tokens: int | None,
    model: str | None,
) -> dict:
    """
    Fill in any field the request left unset from the user's persisted
    Settings-panel preferences — mirrors `_preference_overrides` in
    `src/ui/chat.py`, so a request that omits these fields behaves
    identically to a Streamlit chat message sent with no manual
    override.
    """
    prefs = settings_service.get_preferences()
    return {
        "top_k": top_k if top_k is not None else prefs.top_k,
        "temperature": temperature if temperature is not None else prefs.temperature,
        "max_tokens": max_tokens if max_tokens is not None else prefs.max_tokens,
        "model": model if model is not None else prefs.model,
    }


def _sse_event(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


# ------------------------------------------------------------------
# Session management
# ------------------------------------------------------------------
@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    body: CreateSessionRequest, chat_service: ChatService = Depends(get_chat_service)
) -> ChatSessionResponse:
    session = chat_service.create_session(title=body.title, collection_id=body.collection_id)
    return ChatSessionResponse.from_session(session)


@router.get("/sessions", response_model=list[ChatSessionResponse])
def list_sessions(chat_service: ChatService = Depends(get_chat_service)) -> list[ChatSessionResponse]:
    return [ChatSessionResponse.from_session(s) for s in chat_service.list_sessions()]


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
def get_messages(
    session_id: str, chat_service: ChatService = Depends(get_chat_service)
) -> list[ChatMessageResponse]:
    chat_service.get_session(session_id)  # 404s cleanly on a bad ID
    return [ChatMessageResponse.from_message(m) for m in chat_service.get_messages(session_id)]


@router.delete("/sessions/{session_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
def clear_conversation(
    session_id: str, chat_service: ChatService = Depends(get_chat_service)
) -> None:
    """Deletes every message in the session but keeps the session itself."""
    chat_service.get_session(session_id)
    chat_service.clear_conversation(session_id)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, chat_service: ChatService = Depends(get_chat_service)) -> None:
    chat_service.get_session(session_id)
    chat_service.delete_session(session_id)


# ------------------------------------------------------------------
# Sending messages
# ------------------------------------------------------------------
@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
def send_message(
    session_id: str,
    body: SendMessageRequest,
    chat_service: ChatService = Depends(get_chat_service),
    settings_service: RuntimeSettingsService = Depends(get_settings_service),
) -> ChatMessageResponse:
    chat_service.get_session(session_id)  # 404s cleanly on a bad ID, rather than a raw FK error
    overrides = _apply_preference_defaults(
        settings_service, body.top_k, body.temperature, body.max_tokens, body.model
    )
    assistant_message = chat_service.send_message(
        session_id=session_id, query=body.query, document_ids=body.document_ids, **overrides
    )
    return ChatMessageResponse.from_message(assistant_message)


@router.post("/sessions/{session_id}/messages/stream")
def send_message_stream(
    session_id: str,
    body: SendMessageRequest,
    chat_service: ChatService = Depends(get_chat_service),
    settings_service: RuntimeSettingsService = Depends(get_settings_service),
) -> StreamingResponse:
    chat_service.get_session(session_id)  # validated here, before the response starts —
    # once StreamingResponse begins, the status code can no longer change.
    overrides = _apply_preference_defaults(
        settings_service, body.top_k, body.temperature, body.max_tokens, body.model
    )

    def _event_stream() -> Iterator[str]:
        try:
            token_stream, citations = chat_service.send_message_streaming(
                session_id=session_id, query=body.query, document_ids=body.document_ids, **overrides
            )
        except DocIntelError as exc:
            # Failed before any bytes were sent (e.g. empty query) —
            # still reported as an SSE error frame rather than raised,
            # since a route returning StreamingResponse commits to a
            # 200 status the instant it's called.
            yield _sse_event("error", {"type": type(exc).__name__, "message": str(exc)})
            yield _sse_event("done", {})
            return

        citation_payload = [
            CitationResponse.from_citation(c).model_dump(mode="json") for c in citations
        ]
        yield _sse_event("citations", citation_payload)

        try:
            for chunk in token_stream:
                yield _sse_event("token", chunk)
        except DocIntelError as exc:
            logger.error("Streaming chat response failed mid-stream: %s", exc)
            yield _sse_event("error", {"type": type(exc).__name__, "message": str(exc)})

        yield _sse_event("done", {})

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sessions/{session_id}/regenerate", response_model=ChatMessageResponse)
def regenerate_last_response(
    session_id: str,
    body: RegenerateRequest,
    chat_service: ChatService = Depends(get_chat_service),
    settings_service: RuntimeSettingsService = Depends(get_settings_service),
) -> ChatMessageResponse:
    chat_service.get_session(session_id)
    overrides = _apply_preference_defaults(
        settings_service, body.top_k, body.temperature, body.max_tokens, body.model
    )
    assistant_message = chat_service.regenerate_last_response(
        session_id=session_id, document_ids=body.document_ids, **overrides
    )
    return ChatMessageResponse.from_message(assistant_message)
