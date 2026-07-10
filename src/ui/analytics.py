"""
ui/analytics.py
================
Streamlit "Analytics" page: usage dashboard (documents, chunks,
embeddings, response time, storage, most-queried documents, upload
activity chart) plus a Settings panel for chunking, retrieval, and LLM
parameters.

Presented as two tabs on one page (Analytics / Settings) rather than
two separate top-level files — both are small enough that a dedicated
`ui/settings.py` would be more ceremony than value, and both consume
the same underlying `AnalyticsService`/`RuntimeSettingsService` pair.
"""

from __future__ import annotations

import streamlit as st

from src.services.analytics_service import AnalyticsService
from src.services.settings_service import RuntimeSettingsService
from src.utils.exceptions import DocIntelError, ValidationError


def render(analytics_service: AnalyticsService, settings_service: RuntimeSettingsService) -> None:
    """Render the full Analytics + Settings page."""
    tab_analytics, tab_settings = st.tabs(["📊 Analytics", "⚙️ Settings"])

    with tab_analytics:
        _render_analytics(analytics_service)

    with tab_settings:
        _render_settings(settings_service)


# ------------------------------------------------------------------
# Analytics dashboard
# ------------------------------------------------------------------
def _render_analytics(analytics_service: AnalyticsService) -> None:
    st.subheader("Usage Analytics")

    try:
        summary = analytics_service.get_summary()
    except DocIntelError as exc:
        st.error(f"Couldn't load analytics: {exc}")
        return

    row1 = st.columns(4)
    row1[0].metric("Total Documents", summary.total_documents)
    row1[1].metric("Total Chunks", summary.total_chunks)
    row1[2].metric("Total Embeddings", summary.total_embeddings)
    row1[3].metric("Storage Used", summary.total_storage_display)

    row2 = st.columns(4)
    row2[0].metric("Chat Sessions", summary.total_chat_sessions)
    row2[1].metric("Total Queries", summary.total_queries)
    row2[2].metric("Avg. Response Time", summary.average_response_time_display)
    row2[3].metric(
        "Most Queried",
        summary.most_queried_documents[0].filename if summary.most_queried_documents else "N/A",
    )

    st.divider()

    col_activity, col_ranking = st.columns(2)
    with col_activity:
        st.markdown("**Upload Activity (last 14 days)**")
        if summary.uploads_per_day:
            chart_data = {day: count for day, count in summary.uploads_per_day}
            st.bar_chart(chart_data)
        else:
            st.caption("No uploads yet in this period.")

    with col_ranking:
        st.markdown("**Most Queried Documents**")
        if summary.most_queried_documents:
            for entry in summary.most_queried_documents:
                st.markdown(f"**{entry.filename}**")
                st.caption(f"{entry.query_count} quer{'y' if entry.query_count == 1 else 'ies'}")
        else:
            st.caption("No queries logged yet.")


# ------------------------------------------------------------------
# Settings panel
# ------------------------------------------------------------------
def _render_settings(settings_service: RuntimeSettingsService) -> None:
    st.subheader("Settings")
    st.caption(
        "Changes to Top-K, Temperature, Max Tokens, and Model apply to your "
        "next chat message. Chunk Size/Overlap apply to documents uploaded "
        "after saving — already-processed documents are unaffected."
    )

    try:
        current = settings_service.get_preferences()
    except DocIntelError as exc:
        st.error(f"Couldn't load current settings: {exc}")
        return

    with st.form("settings_form"):
        st.markdown("**Document Chunking**")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size (characters)", min_value=100, max_value=4000, value=current.chunk_size, step=50)
        with col2:
            chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=max(chunk_size - 1, 1), value=min(current.chunk_overlap, max(chunk_size - 1, 1)), step=10)

        st.markdown("**Retrieval**")
        top_k = st.slider("Top-K Retrieval", min_value=1, max_value=20, value=current.top_k)

        st.markdown("**LLM Generation**")
        col3, col4 = st.columns(2)
        with col3:
            temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=current.temperature, step=0.1)
        with col4:
            max_tokens = st.slider("Maximum Tokens", min_value=64, max_value=8192, value=current.max_tokens, step=64)

        model_options = RuntimeSettingsService.available_models()
        model_index = model_options.index(current.model) if current.model in model_options else 0
        model = st.selectbox("Model", options=model_options, index=model_index)

        col_save, col_reset = st.columns(2)
        with col_save:
            save_clicked = st.form_submit_button("Save Settings", type="primary", use_container_width=True)
        with col_reset:
            reset_clicked = st.form_submit_button("Reset to Defaults", use_container_width=True)

    if save_clicked:
        try:
            settings_service.update_preferences(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
            )
            st.success("Settings saved.")
        except ValidationError as exc:
            st.error(f"Couldn't save settings: {exc}")

    if reset_clicked:
        try:
            settings_service.reset_to_defaults()
            st.success("Settings reset to defaults.")
            st.rerun()
        except DocIntelError as exc:
            st.error(f"Couldn't reset settings: {exc}")
