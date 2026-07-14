"""
dependencies.py
================
FastAPI's equivalent of `app.py`'s `@st.cache_resource` singletons.

Streamlit's `@st.cache_resource` and `@functools.lru_cache` solve the
identical problem here: build each service/model exactly once per
process and hand back the same instance on every subsequent call,
because `EmbeddingService` lazily loads a multi-hundred-MB
Sentence-Transformers model on first use — rebuilding it per request
would make the API unusable.

The object graph is intentionally identical to `app.py`'s, built
bottom-up in the same order (db -> embedding/vectorstore -> retriever
-> services), so a future retirement of the Streamlit app is just
deleting `app.py`/`src/ui/` — nothing here changes.

Route handlers take these as `Depends(get_document_service)` etc.
rather than importing the module-level singletons directly, which is
what keeps this same code trivially testable: tests override these
dependencies via `app.dependency_overrides[...]` with fakes, exactly
like the Streamlit test harnesses inject fakes through constructor
arguments.
"""

from __future__ import annotations

from functools import lru_cache

from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import Retriever
from src.services.analytics_service import AnalyticsService
from src.services.chat_service import ChatService
from src.services.document_service import DocumentService
from src.services.export_service import ExportService
from src.services.search_service import SearchService
from src.services.settings_service import RuntimeSettingsService
from src.services.summary_service import SummaryService
from src.vectorstore.chroma_manager import ChromaManager


@lru_cache
def get_db() -> SQLiteManager:
    return SQLiteManager()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_chroma_manager() -> ChromaManager:
    return ChromaManager()


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(embedding_service=get_embedding_service(), chroma_manager=get_chroma_manager())


@lru_cache
def get_document_service() -> DocumentService:
    return DocumentService(
        db=get_db(),
        embedding_service=get_embedding_service(),
        chroma_manager=get_chroma_manager(),
    )


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(db=get_db(), rag_pipeline=RAGPipeline(retriever=get_retriever()))


@lru_cache
def get_search_service() -> SearchService:
    return SearchService(retriever=get_retriever(), db=get_db())


@lru_cache
def get_summary_service() -> SummaryService:
    return SummaryService(db=get_db())


@lru_cache
def get_analytics_service() -> AnalyticsService:
    return AnalyticsService(db=get_db(), chroma_manager=get_chroma_manager())


@lru_cache
def get_settings_service() -> RuntimeSettingsService:
    return RuntimeSettingsService(db=get_db())


@lru_cache
def get_export_service() -> ExportService:
    return ExportService(db=get_db())
