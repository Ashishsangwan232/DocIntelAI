"""
app.py
======
Streamlit application entrypoint.

Currently renders the chat page directly; full sidebar navigation
across Dashboard / Chat / Search / Documents / Analytics / Settings
pages is added incrementally as each of those pages is built
(Phases 7-10). This file is intentionally thin — it wires services
together and delegates all rendering to `src/ui/*.py` modules.

Services are instantiated once via `st.cache_resource` rather than on
every Streamlit rerun, since `EmbeddingService` lazily loads a
multi-hundred-MB model on first use — recreating it every rerun would
make the app unusable.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.rag.retriever import Retriever
from src.rag.pipeline import RAGPipeline
from src.services.analytics_service import AnalyticsService
from src.services.chat_service import ChatService
from src.services.document_service import DocumentService
from src.services.export_service import ExportService
from src.services.search_service import SearchService
from src.services.settings_service import RuntimeSettingsService
from src.services.summary_service import SummaryService
from src.ui import analytics as analytics_page
from src.ui import chat as chat_page
from src.ui import search as search_page
from src.ui import theme
from src.ui import upload as upload_page
from src.vectorstore.chroma_manager import ChromaManager

st.set_page_config(
    page_title=settings.app.app_name,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def _get_db() -> SQLiteManager:
    return SQLiteManager()


@st.cache_resource(show_spinner=False)
def _get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@st.cache_resource(show_spinner=False)
def _get_chroma_manager() -> ChromaManager:
    return ChromaManager()


@st.cache_resource(show_spinner=False)
def _get_retriever() -> Retriever:
    return Retriever(embedding_service=_get_embedding_service(), chroma_manager=_get_chroma_manager())


@st.cache_resource(show_spinner=False)
def _get_document_service() -> DocumentService:
    return DocumentService(
        db=_get_db(),
        embedding_service=_get_embedding_service(),
        chroma_manager=_get_chroma_manager(),
    )


@st.cache_resource(show_spinner=False)
def _get_chat_service() -> ChatService:
    return ChatService(db=_get_db(), rag_pipeline=RAGPipeline(retriever=_get_retriever()))


@st.cache_resource(show_spinner=False)
def _get_search_service() -> SearchService:
    return SearchService(retriever=_get_retriever(), db=_get_db())


@st.cache_resource(show_spinner=False)
def _get_summary_service() -> SummaryService:
    return SummaryService(db=_get_db())


@st.cache_resource(show_spinner=False)
def _get_analytics_service() -> AnalyticsService:
    return AnalyticsService(db=_get_db(), chroma_manager=_get_chroma_manager())


@st.cache_resource(show_spinner=False)
def _get_settings_service() -> RuntimeSettingsService:
    return RuntimeSettingsService(db=_get_db())


@st.cache_resource(show_spinner=False)
def _get_export_service() -> ExportService:
    return ExportService(db=_get_db())


def _render_config_warnings() -> None:
    issues = settings.validate()
    if issues:
        with st.sidebar:
            for issue in issues:
                st.warning(issue, icon="⚠️")


def main() -> None:
    theme.inject_custom_css()
    _render_config_warnings()

    st.title(f"🧠 {settings.app.app_name}")
    st.caption("AI-Powered Intelligent Document Analysis & Retrieval Platform")

    chat_service = _get_chat_service()
    document_service = _get_document_service()
    search_service = _get_search_service()
    summary_service = _get_summary_service()
    analytics_service = _get_analytics_service()
    settings_service = _get_settings_service()
    export_service = _get_export_service()

    # Simple tab-based navigation for now — a dedicated sidebar.py with
    # richer top-level navigation lands in a later phase and will
    # replace this without touching how any page itself works.
    tab_chat, tab_search, tab_documents, tab_analytics = st.tabs(
        ["💬 Chat", "🔍 Search", "📁 Documents", "📈 Analytics"]
    )

    with tab_chat:
        chat_page.render(
            chat_service=chat_service,
            document_service=document_service,
            settings_service=settings_service,
            export_service=export_service,
        )

    with tab_search:
        search_page.render(search_service=search_service, document_service=document_service)

    with tab_documents:
        upload_page.render(
            document_service=document_service,
            summary_service=summary_service,
            settings_service=settings_service,
        )

    with tab_analytics:
        analytics_page.render(analytics_service=analytics_service, settings_service=settings_service)


if __name__ == "__main__":
    main()
