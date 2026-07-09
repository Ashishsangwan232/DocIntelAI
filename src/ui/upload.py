"""
ui/upload.py
============
Streamlit "Documents" page: drag-and-drop upload with per-file
progress/validation feedback, plus the document library (search,
sort, metadata, preview, delete-with-confirmation).

Design notes
------------
- The actual file-processing loop (`_process_uploaded_files`) is a
  plain function operating on `(filename, bytes)` tuples rather than
  Streamlit's `UploadedFile` objects directly. This keeps it directly
  unit-testable without needing to simulate `st.file_uploader` (which
  Streamlit's `AppTest` framework does not support) — the widget layer
  is a thin adapter that unwraps `UploadedFile.name`/`.getvalue()` and
  calls this function.
- Library search/sort are client-side (filter/sort an already-fetched
  list) since the library size is expected to stay in the hundreds,
  not millions, of documents — no need for a dedicated DB query per
  keystroke.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import streamlit as st

from src.database.models import Document, DocumentStatus
from src.services.document_service import DocumentService
from src.services.settings_service import RuntimeSettingsService
from src.services.summary_service import SummaryService
from src.utils.exceptions import DocIntelError
from src.utils.helpers import format_bytes

_STATE_PREVIEW_DOC_ID = "upload_preview_document_id"
_STATE_PENDING_DELETE_ID = "upload_pending_delete_document_id"
_STATE_SUMMARY_DOC_ID = "upload_summary_document_id"
_STATE_SUMMARY_CACHE = "upload_summary_cache"

_SORT_OPTIONS = {
    "Newest first": lambda d: d.uploaded_at,
    "Oldest first": lambda d: d.uploaded_at,
    "Name (A-Z)": lambda d: d.filename.lower(),
    "Size (largest first)": lambda d: d.file_size_bytes,
}

_STATUS_BADGES = {
    DocumentStatus.READY: "🟢 Ready",
    DocumentStatus.PROCESSING: "🟡 Processing",
    DocumentStatus.FAILED: "🔴 Failed",
}


@dataclass
class UploadOutcome:
    """Result of attempting to process a single uploaded file."""

    filename: str
    success: bool
    message: str
    document: Document | None = None


def render(
    document_service: DocumentService,
    summary_service: SummaryService | None = None,
    settings_service: RuntimeSettingsService | None = None,
) -> None:
    """
    Render the full Documents page: upload section + library section.

    `summary_service` is optional — if omitted, the "Generate Summary"
    action is simply not shown. `settings_service` is optional — if
    provided, the persisted chunk size/overlap preferences from the
    Settings panel apply to files uploaded from this page.
    """
    _render_upload_section(document_service, settings_service)
    st.divider()
    _render_library_section(document_service, summary_service)


# ------------------------------------------------------------------
# Upload section
# ------------------------------------------------------------------
def _render_upload_section(
    document_service: DocumentService, settings_service: RuntimeSettingsService | None
) -> None:
    st.subheader("Upload Documents")
    st.caption(
        "Drag and drop files here, or click to browse. "
        "Supported formats: PDF, DOCX, TXT, Markdown."
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        return

    chunk_size, chunk_overlap = _current_chunking_preferences(settings_service)
    files_as_bytes = [(f.name, f.getvalue()) for f in uploaded_files]
    outcomes = _process_uploaded_files(files_as_bytes, document_service, chunk_size, chunk_overlap)
    _render_upload_outcomes(outcomes)


def _current_chunking_preferences(
    settings_service: RuntimeSettingsService | None,
) -> tuple[int | None, int | None]:
    """Read persisted chunk size/overlap preferences, or (None, None) to use config defaults."""
    if settings_service is None:
        return None, None
    prefs = settings_service.get_preferences()
    return prefs.chunk_size, prefs.chunk_overlap


def _process_uploaded_files(
    files: list[tuple[str, bytes]],
    document_service: DocumentService,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> list[UploadOutcome]:
    """
    Process a batch of (filename, content) uploads, showing progress
    and per-file status in the UI. Delegates the actual outcome logic
    to `_attempt_process`, which has no Streamlit dependency and is
    unit tested directly in tests/ui/test_upload_processing.py.
    """
    outcomes: list[UploadOutcome] = []
    progress_bar = st.progress(0.0, text="Starting upload...") if len(files) > 1 else None

    for i, (filename, content) in enumerate(files, start=1):
        if progress_bar is not None:
            progress_bar.progress(i / len(files), text=f"Processing {filename} ({i}/{len(files)})...")

        with st.status(f"Processing {filename}...", expanded=False) as status:
            outcome = _attempt_process(filename, content, document_service, chunk_size, chunk_overlap)
            if outcome.success:
                status.update(label=f"✅ {filename} processed successfully", state="complete")
            else:
                status.update(label=f"❌ {filename} failed", state="error")
        outcomes.append(outcome)

    if progress_bar is not None:
        progress_bar.empty()

    return outcomes


def _attempt_process(
    filename: str,
    content: bytes,
    document_service: DocumentService,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> UploadOutcome:
    """
    Attempt to process a single uploaded file, with no Streamlit
    dependency whatsoever — this is the actual unit of business logic
    the UI wraps with progress/status widgets.
    """
    try:
        document = document_service.process_upload(
            filename, content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return UploadOutcome(
            filename=filename, success=True, message="Processed successfully", document=document
        )
    except DocIntelError as exc:
        return UploadOutcome(filename=filename, success=False, message=str(exc))


def _render_upload_outcomes(outcomes: list[UploadOutcome]) -> None:
    successes = [o for o in outcomes if o.success]
    failures = [o for o in outcomes if not o.success]

    if successes:
        st.success(f"{len(successes)} file(s) uploaded and processed successfully.")
    for outcome in failures:
        st.error(f"**{outcome.filename}**: {outcome.message}")


# ------------------------------------------------------------------
# Library section
# ------------------------------------------------------------------
def _render_library_section(
    document_service: DocumentService, summary_service: SummaryService | None
) -> None:
    st.subheader("Document Library")

    documents = document_service.list_documents()
    if not documents:
        st.info("No documents uploaded yet. Upload a file above to get started.")
        return

    col_search, col_sort = st.columns([3, 1])
    with col_search:
        search_query = st.text_input(
            "Search documents", placeholder="Search by filename...", label_visibility="collapsed"
        )
    with col_sort:
        sort_choice = st.selectbox("Sort by", options=list(_SORT_OPTIONS.keys()), label_visibility="collapsed")

    filtered = _filter_documents(documents, search_query)
    sorted_docs = _sort_documents(filtered, sort_choice)

    if not sorted_docs:
        st.info(f"No documents match '{search_query}'.")
        return

    for document in sorted_docs:
        _render_document_row(document, document_service, summary_service)

    _maybe_render_preview_dialog(document_service)
    _maybe_render_delete_dialog(document_service)
    _maybe_render_summary_dialog(document_service, summary_service)


def _filter_documents(documents: list[Document], query: str) -> list[Document]:
    if not query or not query.strip():
        return documents
    needle = query.strip().lower()
    return [d for d in documents if needle in d.filename.lower()]


def _sort_documents(documents: list[Document], sort_choice: str) -> list[Document]:
    key_fn = _SORT_OPTIONS[sort_choice]
    reverse = sort_choice in ("Newest first", "Size (largest first)")
    return sorted(documents, key=key_fn, reverse=reverse)


def _render_document_row(
    document: Document,
    document_service: DocumentService,
    summary_service: SummaryService | None,
) -> None:
    with st.container(border=True):
        col_info, col_badges, col_actions = st.columns([4, 3, 2])

        with col_info:
            st.markdown(f"**{document.filename}**")
            st.caption(_format_uploaded_at(document.uploaded_at))

        with col_badges:
            status_label = _STATUS_BADGES.get(document.status, document.status.value)
            st.caption(status_label)
            size_label = format_bytes(document.file_size_bytes)
            type_label = document.file_type.value.upper()
            if document.status == DocumentStatus.READY:
                chunk_count = document_service.get_chunk_count(document.id)
                st.caption(f"{type_label} · {size_label} · {chunk_count} chunks")
            else:
                st.caption(f"{type_label} · {size_label}")

        with col_actions:
            show_summarize = summary_service is not None and document.status == DocumentStatus.READY
            num_buttons = 3 if show_summarize else 2
            action_cols = st.columns(num_buttons)

            with action_cols[0]:
                if st.button("👁", key=f"preview_{document.id}", help="Preview"):
                    st.session_state[_STATE_PREVIEW_DOC_ID] = document.id
                    st.rerun()

            next_col = 1
            if show_summarize:
                with action_cols[next_col]:
                    if st.button("📝", key=f"summarize_{document.id}", help="Generate AI Summary"):
                        st.session_state[_STATE_SUMMARY_DOC_ID] = document.id
                        st.rerun()
                next_col += 1

            with action_cols[next_col]:
                if st.button("🗑", key=f"delete_{document.id}", help="Delete"):
                    st.session_state[_STATE_PENDING_DELETE_ID] = document.id
                    st.rerun()


def _format_uploaded_at(uploaded_at: datetime) -> str:
    return uploaded_at.strftime("Uploaded %b %d, %Y at %H:%M")


@st.dialog("Delete document")
def _confirm_delete_dialog(document: Document, document_service: DocumentService) -> None:
    st.write(f"Delete **{document.filename}**? This will remove its extracted "
             f"text, embeddings, and the original file. This cannot be undone.")
    col_cancel, col_delete = st.columns(2)
    with col_cancel:
        if st.button("Cancel", use_container_width=True, key=f"delete_cancel_{document.id}"):
            # No explicit st.rerun() — the button click itself already
            # triggers Streamlit's automatic rerun. An extra explicit
            # rerun here caused a `st.dialog` widget-lifecycle crash in
            # Streamlit's AppTest framework (confirmed via a minimal,
            # app-independent reproduction); it's also simply redundant.
            st.session_state[_STATE_PENDING_DELETE_ID] = None
    with col_delete:
        if st.button("Delete", type="primary", use_container_width=True, key=f"delete_confirm_{document.id}"):
            document_service.delete_document(document.id)
            st.session_state[_STATE_PENDING_DELETE_ID] = None


def _maybe_render_delete_dialog(document_service: DocumentService) -> None:
    document_id = st.session_state.get(_STATE_PENDING_DELETE_ID)
    if not document_id:
        return

    try:
        document = document_service.get_document(document_id)
    except DocIntelError:
        st.session_state[_STATE_PENDING_DELETE_ID] = None
        return

    # Called unconditionally as long as the pending-delete state is
    # set — NOT gated on the row's button being clicked *this* rerun.
    # A dialog invoked only inside a momentary `if st.button(...)`
    # block would never see its own Cancel/Delete clicks handled,
    # since that outer condition resets to False on the very next
    # rerun (the one triggered by clicking inside the dialog itself).
    _confirm_delete_dialog(document, document_service)


def _maybe_render_preview_dialog(document_service: DocumentService) -> None:
    document_id = st.session_state.get(_STATE_PREVIEW_DOC_ID)
    if not document_id:
        return

    try:
        document = document_service.get_document(document_id)
    except DocIntelError:
        st.session_state[_STATE_PREVIEW_DOC_ID] = None
        return

    _preview_dialog(document, document_service)


@st.dialog("Document preview", width="large")
def _preview_dialog(document: Document, document_service: DocumentService) -> None:
    st.markdown(f"**{document.filename}**")
    preview_text = document_service.get_preview_text(document.id, max_chars=2000)
    st.text_area(
        "Preview", value=preview_text, height=300, disabled=True,
        label_visibility="collapsed", key=f"preview_text_{document.id}",
    )
    if st.button("Close", key=f"preview_close_{document.id}"):
        st.session_state[_STATE_PREVIEW_DOC_ID] = None


def _maybe_render_summary_dialog(
    document_service: DocumentService, summary_service: SummaryService | None
) -> None:
    if summary_service is None:
        return

    document_id = st.session_state.get(_STATE_SUMMARY_DOC_ID)
    if not document_id:
        return

    try:
        document = document_service.get_document(document_id)
    except DocIntelError:
        st.session_state[_STATE_SUMMARY_DOC_ID] = None
        return

    _summary_dialog(document, summary_service)


@st.dialog("AI Summary", width="large")
def _summary_dialog(document: Document, summary_service: SummaryService) -> None:
    st.markdown(f"**{document.filename}**")
    cache = st.session_state.setdefault(_STATE_SUMMARY_CACHE, {})
    summary = cache.get(document.id)

    if summary is None:
        with st.spinner("Generating summary..."):
            try:
                summary = summary_service.generate_summary(document.id)
                cache[document.id] = summary
            except DocIntelError as exc:
                st.error(f"Couldn't generate a summary: {exc}")
                if st.button("Close", key=f"summary_close_error_{document.id}"):
                    st.session_state[_STATE_SUMMARY_DOC_ID] = None
                return

    st.markdown("**Executive Summary**")
    st.write(summary.executive_summary)

    if summary.key_insights:
        st.markdown("**Key Insights**")
        for insight in summary.key_insights:
            st.markdown(f"- {insight}")

    if summary.topics:
        st.markdown("**Topics**")
        st.write(", ".join(summary.topics))

    col_regenerate, col_close = st.columns(2)
    with col_regenerate:
        if st.button("🔄 Regenerate", key=f"summary_regenerate_{document.id}", use_container_width=True):
            cache.pop(document.id, None)
    with col_close:
        if st.button("Close", key=f"summary_close_{document.id}", use_container_width=True):
            st.session_state[_STATE_SUMMARY_DOC_ID] = None
