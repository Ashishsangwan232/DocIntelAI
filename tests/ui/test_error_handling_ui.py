"""
Unit tests for the Phase 13 error-handling hardening in src/ui/upload.py.

Verifies that failures from the service layer surface as clean
`st.error` messages rather than crashing the page with a raw
traceback — using a fake `DocumentService` configured (via the
`ERROR_HARNESS_FAIL_ON` env var) to raise on exactly one operation
per test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

_HARNESS_PATH = str(Path(__file__).parent / "error_handling_app_harness.py")


@pytest.fixture(autouse=True)
def _clear_state():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
    os.environ.pop("ERROR_HARNESS_FAIL_ON", None)


def _run_with_failure(operation: str) -> AppTest:
    os.environ["ERROR_HARNESS_FAIL_ON"] = operation
    at = AppTest.from_file(_HARNESS_PATH, default_timeout=30)
    at.run()
    return at


class TestDeleteFailureHandling:
    def test_delete_failure_shows_clean_error_not_crash(self) -> None:
        at = _run_with_failure("delete_document")
        delete_buttons = [b for b in at.button if b.label == "🗑"]
        delete_buttons[0].click().run()
        confirm_button = [b for b in at.button if b.label == "Delete"][0]
        confirm_button.click().run()

        assert not at.exception
        assert any("Simulated failure during delete_document" in e.value for e in at.error)


class TestPreviewFailureHandling:
    def test_preview_failure_shows_message_in_text_area_not_crash(self) -> None:
        at = _run_with_failure("get_preview_text")
        preview_buttons = [b for b in at.button if b.label == "👁"]
        preview_buttons[0].click().run()

        assert not at.exception
        assert any("Preview unavailable" in ta.value for ta in at.text_area)


class TestListDocumentsFailureHandling:
    def test_library_load_failure_shows_error_not_crash(self) -> None:
        at = _run_with_failure("list_documents")
        assert not at.exception
        assert any("couldn't" in e.value.lower() or "error" in e.value.lower() for e in at.error)


class TestHappyPathUnaffected:
    def test_no_failures_configured_renders_normally(self) -> None:
        at = _run_with_failure("")  # empty string never matches any operation name
        assert not at.exception
        assert any(m.value == "**test.pdf**" for m in at.markdown)
        assert len(at.error) == 0
