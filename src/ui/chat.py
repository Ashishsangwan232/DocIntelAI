"""
ui/chat.py
==========
Streamlit chat page — the ChatGPT-style interface for talking to
uploaded documents.

This module contains ONLY presentation logic. Every piece of business
logic (sending messages, regenerating, clearing history, retrieval
scoping) is delegated to `ChatService` / `DocumentService`. This
separation is what lets the Streamlit UI be swapped for a React
frontend later without touching how chat actually works.
"""

from __future__ import annotations

import streamlit as st

from src.database.models import ChatMessage, Citation, DocumentStatus, MessageRole
from src.services.chat_service import ChatService
from src.services.document_service import DocumentService
from src.services.export_service import ExportService
from src.services.settings_service import RuntimeSettingsService
from src.ui import theme
from src.utils.exceptions import DocIntelError

# Session-state keys namespaced to this page to avoid collisions with
# other pages (dashboard, upload, etc.) sharing the same Streamlit run.
_STATE_SESSION_ID = "chat_active_session_id"
_STATE_DOCUMENT_SCOPE = "chat_document_scope"


def render(
    chat_service: ChatService,
    document_service: DocumentService | None = None,
    settings_service: RuntimeSettingsService | None = None,
    export_service: ExportService | None = None,
) -> None:
    """
    Render the full chat page.

    Args:
        chat_service: Handles sending/regenerating/persisting messages.
        document_service: Optional — if provided, enables scoping chat
            to specific documents via a multiselect. If omitted, chat
            runs against the entire document library.
        settings_service: Optional — if provided, the user's persisted
            Top-K/temperature/max-tokens/model preferences (Settings
            panel) are applied to every message sent from this page.
            If omitted, `ChatService`/`RAGPipeline` config defaults apply.
        export_service: Optional — if provided, enables exporting the
            current conversation as PDF/Markdown/plain text. If
            omitted, the Export controls are simply not shown.
    """
    _ensure_active_session(chat_service)

    session_id = st.session_state[_STATE_SESSION_ID]
    messages = chat_service.get_messages(session_id)

    _render_toolbar(chat_service, document_service, export_service, session_id, messages)

    if not messages:
        st.info(
            "Ask a question about your uploaded documents to get started — "
            "answers are grounded in your documents, with citations."
        )

    for index, message in enumerate(messages):
        _render_message(
            message,
            is_last_assistant_message=(
                message.role == MessageRole.ASSISTANT and index == len(messages) - 1
            ),
            chat_service=chat_service,
            session_id=session_id,
            settings_service=settings_service,
        )

    _render_chat_input(chat_service, session_id, settings_service)


# ------------------------------------------------------------------
# Toolbar: new chat, clear conversation, document scope, export
# ------------------------------------------------------------------
def _render_toolbar(
    chat_service: ChatService,
    document_service: DocumentService | None,
    export_service: ExportService | None,
    session_id: str,
    messages: list[ChatMessage],
) -> None:
    show_export = export_service is not None and len(messages) > 0

    if show_export:
        col_new, col_clear, col_export, col_scope = st.columns([1, 1, 1, 2])
    else:
        col_new, col_clear, col_scope = st.columns([1, 1, 2])

    with col_new:
        if st.button("New Chat", use_container_width=True):
            new_session = chat_service.create_session()
            st.session_state[_STATE_SESSION_ID] = new_session.id
            st.session_state[_STATE_DOCUMENT_SCOPE] = []
            st.rerun()

    with col_clear:
        if st.button("Clear Conversation", use_container_width=True):
            chat_service.clear_conversation(st.session_state[_STATE_SESSION_ID])
            st.rerun()

    if show_export:
        with col_export:
            _render_export_menu(export_service, session_id, messages)

    with col_scope:
        if document_service is not None:
            _render_document_scope_selector(document_service)


def _render_export_menu(export_service: ExportService, session_id: str, messages: list[ChatMessage]) -> None:
    with st.popover("📤 Export", use_container_width=True):
        session = _get_session_for_export(export_service, session_id)

        st.download_button(
            "Download as PDF",
            data=export_service.export_to_pdf(session_id),
            file_name=f"{_safe_export_filename(session.title)}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.download_button(
            "Download as Markdown",
            data=export_service.export_to_markdown(session_id),
            file_name=f"{_safe_export_filename(session.title)}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.caption("Copy conversation:")
        st.code(export_service.export_to_text(session_id), language=None)


def _get_session_for_export(export_service: ExportService, session_id: str):
    # ExportService methods each fetch the session internally; this
    # thin helper avoids fetching it a fourth time just for the title
    # used in the download filename.
    return export_service.db.get_chat_session(session_id)


def _safe_export_filename(title: str) -> str:
    from src.utils.helpers import sanitize_filename

    return sanitize_filename(title.replace(" ", "_")) or "conversation"


def _render_document_scope_selector(document_service: DocumentService) -> None:
    ready_documents = document_service.list_documents(status=DocumentStatus.READY)
    if not ready_documents:
        st.caption("No processed documents yet — upload one to chat with it.")
        st.session_state[_STATE_DOCUMENT_SCOPE] = []
        return

    options = {doc.id: doc.filename for doc in ready_documents}
    selected_ids = st.multiselect(
        "Chat scope",
        options=list(options.keys()),
        default=st.session_state.get(_STATE_DOCUMENT_SCOPE, []),
        format_func=lambda doc_id: options[doc_id],
        placeholder="All documents",
        label_visibility="collapsed",
    )
    st.session_state[_STATE_DOCUMENT_SCOPE] = selected_ids


def _current_document_scope() -> list[str] | None:
    selected = st.session_state.get(_STATE_DOCUMENT_SCOPE, [])
    return selected or None  # empty selection means "search all documents"


# ------------------------------------------------------------------
# Message rendering
# ------------------------------------------------------------------
def _render_message(
    message: ChatMessage,
    is_last_assistant_message: bool,
    chat_service: ChatService,
    session_id: str,
    settings_service: RuntimeSettingsService | None,
) -> None:
    with st.chat_message(message.role.value):
        st.markdown(message.content)

        if message.role == MessageRole.ASSISTANT:
            if message.citations:
                _render_citations(message.citations)
            _render_message_actions(
                message, is_last_assistant_message, chat_service, session_id, settings_service
            )


def _render_citations(citations: list[Citation]) -> None:
    with st.expander(f"📎 {len(citations)} source{'s' if len(citations) != 1 else ''}"):
        for i, citation in enumerate(citations, start=1):
            page_info = f", page {citation.page_number}" if citation.page_number else ""
            dial_html = theme.render_match_dial(citation.similarity_score)
            st.markdown(
                f"**[Source {i}] {citation.filename}{page_info}** · {dial_html}",
                unsafe_allow_html=True,
            )
            st.caption(citation.excerpt)
            if i < len(citations):
                st.divider()


def _render_message_actions(
    message: ChatMessage,
    is_last_assistant_message: bool,
    chat_service: ChatService,
    session_id: str,
    settings_service: RuntimeSettingsService | None,
) -> None:
    cols = st.columns([1, 1, 10])

    with cols[0]:
        with st.popover("📋", help="Copy answer"):
            st.code(message.content, language=None)

    if is_last_assistant_message:
        with cols[1]:
            if st.button("🔄", key=f"regenerate_{message.id}", help="Regenerate response"):
                with st.spinner("Regenerating..."):
                    try:
                        overrides = _preference_overrides(settings_service)
                        chat_service.regenerate_last_response(
                            session_id, document_ids=_current_document_scope(), **overrides
                        )
                    except DocIntelError as exc:
                        st.error(f"Couldn't regenerate the response: {exc}")
                st.rerun()


# ------------------------------------------------------------------
# Chat input + streaming response
# ------------------------------------------------------------------
def _render_chat_input(
    chat_service: ChatService,
    session_id: str,
    settings_service: RuntimeSettingsService | None,
) -> None:
    query = st.chat_input("Ask a question about your documents...")
    if not query:
        return

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            overrides = _preference_overrides(settings_service)
            stream, citations = chat_service.send_message_streaming(
                session_id, query, document_ids=_current_document_scope(), **overrides
            )
            st.write_stream(stream)
            if citations:
                _render_citations(citations)
        except DocIntelError as exc:
            st.error(f"Something went wrong answering that question: {exc}")

    st.rerun()


def _preference_overrides(settings_service: RuntimeSettingsService | None) -> dict:
    """
    Translate the persisted Settings-panel preferences into the
    keyword overrides `ChatService.send_message*`/`regenerate_last_response`
    already accept. Returns an empty dict (i.e. use config defaults)
    if no settings service was provided.
    """
    if settings_service is None:
        return {}
    prefs = settings_service.get_preferences()
    return {
        "top_k": prefs.top_k,
        "temperature": prefs.temperature,
        "max_tokens": prefs.max_tokens,
        "model": prefs.model,
    }


# ------------------------------------------------------------------
# Session bootstrapping
# ------------------------------------------------------------------
def _ensure_active_session(chat_service: ChatService) -> None:
    if _STATE_SESSION_ID not in st.session_state:
        session = chat_service.create_session()
        st.session_state[_STATE_SESSION_ID] = session.id
        st.session_state[_STATE_DOCUMENT_SCOPE] = []
