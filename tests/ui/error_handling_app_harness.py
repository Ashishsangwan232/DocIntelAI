"""
error_handling_app_harness.py
===============================
A minimal Streamlit script used only by
`tests/ui/test_error_handling_ui.py`, wiring `ui/upload.py` to a fake
`DocumentService` whose methods can be configured to raise —
verifying the Phase 13 error-handling hardening actually shows clean
`st.error` messages instead of crashing the page.
"""

from __future__ import annotations

import os

import streamlit as st

from src.database.models import Document, DocumentStatus, FileType
from src.ui import upload as upload_page
from src.utils.exceptions import DocIntelError

# Which operation should simulate a failure, set via env var by the
# test (mirrors the SEARCH_UI_HARNESS_DIR pattern in search_app_harness.py).
_FAIL_ON = os.environ.get("ERROR_HARNESS_FAIL_ON", "")


class ConfigurableFailingDocumentService:
    """Fake DocumentService where exactly one operation can be configured to fail."""

    def __init__(self, doc: Document):
        self.doc = doc

    def _maybe_fail(self, operation: str) -> None:
        if operation == _FAIL_ON:
            raise DocIntelError(f"Simulated failure during {operation}")

    def list_documents(self, collection_id=None, status=None):
        self._maybe_fail("list_documents")
        return [self.doc]

    def get_document(self, document_id):
        self._maybe_fail("get_document")
        return self.doc

    def get_chunk_count(self, document_id):
        return 3

    def get_preview_text(self, document_id, max_chars=2000):
        self._maybe_fail("get_preview_text")
        return "preview text content"

    def delete_document(self, document_id):
        self._maybe_fail("delete_document")

    def process_upload(self, *args, **kwargs):
        raise NotImplementedError


@st.cache_resource(show_spinner=False)
def _get_doc() -> Document:
    doc = Document(filename="test.pdf", file_type=FileType.PDF, file_size_bytes=100, file_hash="h1")
    doc.status = DocumentStatus.READY
    return doc


document_service = ConfigurableFailingDocumentService(_get_doc())
upload_page.render(document_service=document_service)
