"""
ui/search.py
============
Streamlit "Semantic Search" page — search the document library by
meaning rather than keywords, with similarity scores and highlighted
matching terms. No LLM call is involved; this is `SearchService`
(retrieval only) rendered directly.

Distinct from `ui/chat.py`: chat synthesizes an answer from retrieved
context, search just shows you the ranked source material itself —
useful for quickly locating a clause/fact without waiting on
generation, and for verifying what the chat feature is drawing from.
"""

from __future__ import annotations

import streamlit as st

from src.database.models import DocumentStatus
from src.services.document_service import DocumentService
from src.services.search_service import SearchHit, SearchService
from src.ui import theme
from src.utils.exceptions import DocIntelError

_STATE_SEARCH_SCOPE = "search_document_scope"
_STATE_LAST_QUERY = "search_last_query"
_STATE_LAST_RESULTS = "search_last_results"


def render(search_service: SearchService, document_service: DocumentService | None = None) -> None:
    """Render the full semantic search page."""
    st.subheader("Semantic Search")
    st.caption(
        "Search your documents by meaning, not just keywords. "
        "Results are ranked by similarity — no AI-generated answer, just the source material."
    )

    _render_scope_selector(document_service)
    query = st.text_input(
        "Search query", placeholder="Search your documents...", label_visibility="collapsed"
    )

    if query and query.strip():
        _run_search(search_service, query)

    _render_results()


def _render_scope_selector(document_service: DocumentService | None) -> None:
    if document_service is None:
        return

    try:
        ready_documents = document_service.list_documents(status=DocumentStatus.READY)
    except DocIntelError as exc:
        st.error(f"Couldn't load document list: {exc}")
        return

    if not ready_documents:
        st.info("No processed documents yet — upload one to start searching.")
        return

    options = {doc.id: doc.filename for doc in ready_documents}
    selected_ids = st.multiselect(
        "Search scope",
        options=list(options.keys()),
        default=st.session_state.get(_STATE_SEARCH_SCOPE, []),
        format_func=lambda doc_id: options[doc_id],
        placeholder="All documents",
        label_visibility="collapsed",
    )
    st.session_state[_STATE_SEARCH_SCOPE] = selected_ids


def _current_scope() -> list[str] | None:
    selected = st.session_state.get(_STATE_SEARCH_SCOPE, [])
    return selected or None


def _run_search(search_service: SearchService, query: str) -> None:
    # Avoid re-running the search on every rerun if the query hasn't
    # changed (e.g. when a scope multiselect interaction triggers a
    # rerun but the search box value is unchanged).
    if st.session_state.get(_STATE_LAST_QUERY) == query:
        return

    try:
        with st.spinner("Searching..."):
            hits = search_service.search(query, document_ids=_current_scope())
        st.session_state[_STATE_LAST_QUERY] = query
        st.session_state[_STATE_LAST_RESULTS] = hits
    except DocIntelError as exc:
        st.error(f"Search failed: {exc}")
        st.session_state[_STATE_LAST_QUERY] = query
        st.session_state[_STATE_LAST_RESULTS] = []


def _render_results() -> None:
    hits: list[SearchHit] = st.session_state.get(_STATE_LAST_RESULTS, [])
    query_ran = st.session_state.get(_STATE_LAST_QUERY)

    if query_ran is None:
        return

    if not hits:
        st.info(f"No results found for '{query_ran}'. Try rephrasing your search.")
        return

    st.caption(f"{len(hits)} result{'s' if len(hits) != 1 else ''} for '{query_ran}'")

    for hit in hits:
        _render_hit(hit)


def _render_hit(hit: SearchHit) -> None:
    with st.container(border=True):
        col_title, col_score = st.columns([4, 1])
        with col_title:
            page_suffix = f", page {hit.page_number}" if hit.page_number else ""
            st.markdown(f"**{hit.filename}**{page_suffix}")
        with col_score:
            st.markdown(theme.render_match_dial(hit.similarity_score), unsafe_allow_html=True)
            st.caption(f"{hit.similarity_score:.0%} match")

        st.markdown(hit.snippet)
