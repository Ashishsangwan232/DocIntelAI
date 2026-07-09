"""
search_app_harness.py
======================
A minimal, self-contained Streamlit script used only by
`tests/ui/test_search_ui.py` via Streamlit's `AppTest` framework.

Seeds a document + chunk directly into ChromaDB via a fake embedding
model (deterministic, no network), then renders `ui/search.py` wired
to a real `SearchService`/`Retriever` — only the embedding model is
faked, all retrieval/highlighting logic is real.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st

from src.database.models import Chunk, Document, DocumentStatus, FileType
from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.rag.retriever import Retriever
from src.services.document_service import DocumentService
from src.services.search_service import SearchService
from src.ui import search as search_page
from src.vectorstore.chroma_manager import ChromaManager

# Each test run gets a unique base directory (set via env var by the
# test fixture, using pytest's tmp_path). ChromaDB's Rust backend
# maintains an internal by-path client registry that does not clean
# up correctly when a directory is deleted and recreated at the same
# path within the same process — reusing a fixed path across multiple
# AppTest runs in one pytest session intermittently raised "attempt to
# write a readonly database" on the second and later runs. A fresh
# path per test sidesteps this entirely (the same reason every other
# test module in this project uses pytest's `tmp_path` instead of a
# fixed /tmp path).
_BASE_DIR = Path(os.environ.get("SEARCH_UI_HARNESS_DIR", "/tmp/search_ui_harness_default"))
_DB_PATH = _BASE_DIR / "harness.db"
_CHROMA_DIR = _BASE_DIR / "chroma"
_FILES_DIR = _BASE_DIR / "files"


class DirectionalFakeModel:
    """Maps text containing 'notice' to one direction, everything else to another."""

    def encode(self, sentences, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
        vectors = []
        for text in sentences:
            if "notice" in text.lower():
                vectors.append(np.array([1.0, 0.0, 0.0], dtype="float32"))
            elif "warranty" in text.lower():
                vectors.append(np.array([0.0, 1.0, 0.0], dtype="float32"))
            elif "unrelated" in text.lower() or "gibberish" in text.lower():
                vectors.append(np.array([-1.0, 0.0, 0.0], dtype="float32"))  # opposite of "notice"
            else:
                vectors.append(np.array([0.5, 0.5, 0.5], dtype="float32"))
        return np.array(vectors)

    def get_sentence_embedding_dimension(self) -> int:
        return 3


@st.cache_resource(show_spinner=False)
def _harness_db() -> SQLiteManager:
    return SQLiteManager(db_path=_DB_PATH)


@st.cache_resource(show_spinner=False)
def _harness_embedding_service() -> EmbeddingService:
    return EmbeddingService(model=DirectionalFakeModel())


@st.cache_resource(show_spinner=False)
def _harness_chroma() -> ChromaManager:
    return ChromaManager(persist_directory=_CHROMA_DIR, collection_name="test")


@st.cache_resource(show_spinner=False)
def _harness_document_service() -> DocumentService:
    return DocumentService(
        db=_harness_db(),
        upload_dir=_FILES_DIR,
        embedding_service=_harness_embedding_service(),
        chroma_manager=_harness_chroma(),
    )


@st.cache_resource(show_spinner=False)
def _harness_search_service() -> SearchService:
    retriever = Retriever(
        embedding_service=_harness_embedding_service(),
        chroma_manager=_harness_chroma(),
        score_threshold=0.6,
    )
    return SearchService(retriever=retriever, db=_harness_db())


@st.cache_resource(show_spinner=False)
def _harness_bootstrap() -> bool:
    """
    Single cached entrypoint that seeds the document/chunk exactly
    once. Consolidating seeding behind one cache_resource call (rather
    than calling `_harness_chroma()` independently from multiple other
    cached functions in the same run) avoids a first-population race
    across ChromaDB's PersistentClient that otherwise intermittently
    raised 'attempt to write a readonly database' in this environment.
    """
    db = _harness_db()
    if db.count_documents() > 0:
        return True

    document = Document(
        filename="contract.pdf", file_type=FileType.PDF,
        file_size_bytes=1024, file_hash="harness-hash",
    )
    db.create_document(document)
    chunk = Chunk(
        document_id=document.id, chunk_index=0, page_number=4,
        content="Either party may terminate this agreement with 30 days written "
                "notice to the other party, delivered in writing.",
    )
    db.bulk_create_chunks([chunk])
    db.update_document_status(document.id, DocumentStatus.READY, page_count=1)

    embeddings = _harness_embedding_service().embed_texts([chunk.content])
    _harness_chroma().add_chunks(document, [chunk], embeddings)
    return True


_FILES_DIR.mkdir(parents=True, exist_ok=True)
_harness_bootstrap()
search_page.render(
    search_service=_harness_search_service(),
    document_service=_harness_document_service(),
)
