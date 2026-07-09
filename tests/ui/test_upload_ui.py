"""
Unit tests for src/ui/upload.py's document library section (search,
sort, preview, delete-with-confirmation).

Uses Streamlit's official `AppTest` framework against
`tests/ui/upload_app_harness.py`, which seeds real documents directly
into the database (real processing/embedding is already covered by
`tests/ui/test_upload_processing.py` and
`tests/services/test_document_service.py`).

Note: dialog button handlers (Cancel/Delete/Close) intentionally do
NOT call `st.rerun()` explicitly — the button click itself already
triggers Streamlit's automatic rerun, and an extra explicit rerun call
was found to crash Streamlit's `AppTest` framework specifically when a
`st.dialog`'s widgets unmount (confirmed via a minimal, app-independent
reproduction — a framework limitation, not an application bug). Tests
below call `.run()` once after each interaction to reach the stable
post-rerun state, matching the pattern used in test_chat_ui.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_HARNESS_PATH = str(Path(__file__).parent / "upload_app_harness.py")
_HARNESS_DB_PATH = Path("/tmp/upload_ui_harness.db")
_HARNESS_FILES_DIR = Path("/tmp/upload_ui_harness_files")
_HARNESS_CHROMA_DIR = Path("/tmp/upload_ui_harness_chroma")


def _settle(at: AppTest) -> AppTest:
    at.run()
    return at


@pytest.fixture(autouse=True)
def _fresh_harness_state():
    """Fresh DB and cleared Streamlit resource cache for every test (see test_chat_ui.py for why)."""
    import streamlit as st

    st.cache_resource.clear()
    _HARNESS_DB_PATH.unlink(missing_ok=True)
    yield
    st.cache_resource.clear()
    _HARNESS_DB_PATH.unlink(missing_ok=True)


@pytest.fixture()
def app() -> AppTest:
    at = AppTest.from_file(_HARNESS_PATH, default_timeout=30)
    at.run()
    return at


class TestInitialRender:
    def test_renders_without_exceptions(self, app: AppTest) -> None:
        assert not app.exception

    def test_shows_all_seeded_documents(self, app: AppTest) -> None:
        titles = [md.value for md in app.markdown]
        assert any("contract_alpha.pdf" in t for t in titles)
        assert any("policy_beta.docx" in t for t in titles)
        assert any("broken_gamma.pdf" in t for t in titles)

    def test_shows_status_badges(self, app: AppTest) -> None:
        captions = [c.value for c in app.caption]
        assert any("Ready" in c for c in captions)
        assert any("Failed" in c for c in captions)

    def test_shows_chunk_counts_for_ready_documents(self, app: AppTest) -> None:
        captions = [c.value for c in app.caption]
        assert any("chunks" in c for c in captions)

    def test_shows_search_and_sort_controls(self, app: AppTest) -> None:
        assert len(app.text_input) == 1
        assert len(app.selectbox) == 1


class TestSearch:
    def test_filters_to_matching_filename(self, app: AppTest) -> None:
        app.text_input[0].set_value("alpha").run()
        titles = [md.value for md in app.markdown]
        assert titles == ["**contract_alpha.pdf**"]

    def test_empty_search_shows_all_documents(self, app: AppTest) -> None:
        app.text_input[0].set_value("").run()
        assert len(app.markdown) == 3

    def test_no_match_shows_info_message(self, app: AppTest) -> None:
        app.text_input[0].set_value("nonexistent_file_xyz").run()
        assert not app.exception
        assert any("no documents match" in i.value.lower() for i in app.info)

    def test_search_is_case_insensitive(self, app: AppTest) -> None:
        app.text_input[0].set_value("ALPHA").run()
        titles = [md.value for md in app.markdown]
        assert titles == ["**contract_alpha.pdf**"]


class TestSort:
    def test_sort_by_name_ascending(self, app: AppTest) -> None:
        app.selectbox[0].set_value("Name (A-Z)").run()
        titles = [md.value for md in app.markdown]
        assert titles == ["**broken_gamma.pdf**", "**contract_alpha.pdf**", "**policy_beta.docx**"]

    def test_sort_by_size_largest_first(self, app: AppTest) -> None:
        app.selectbox[0].set_value("Size (largest first)").run()
        titles = [md.value for md in app.markdown]
        # contract_alpha.pdf (200KB) > policy_beta.docx (50KB) > broken_gamma.pdf (1KB)
        assert titles[0] == "**contract_alpha.pdf**"
        assert titles[-1] == "**broken_gamma.pdf**"


class TestPreview:
    def test_clicking_preview_shows_content(self, app: AppTest) -> None:
        app.selectbox[0].set_value("Name (A-Z)").run()  # deterministic order
        preview_buttons = [b for b in app.button if b.label == "👁"]
        preview_buttons[1].click().run()  # index 1 = contract_alpha.pdf after sort

        assert not app.exception
        assert len(app.text_area) == 1
        assert "chunk" in app.text_area[0].value.lower()

    def test_closing_preview_removes_text_area(self, app: AppTest) -> None:
        app.selectbox[0].set_value("Name (A-Z)").run()
        preview_buttons = [b for b in app.button if b.label == "👁"]
        preview_buttons[1].click().run()

        close_button = [b for b in app.button if b.label == "Close"][0]
        close_button.click().run()
        _settle(app)

        assert not app.exception
        assert len(app.text_area) == 0


class TestSummarize:
    def test_summarize_button_only_shown_for_ready_documents(self, app: AppTest) -> None:
        # 2 READY documents (contract_alpha, policy_beta) + 1 FAILED (broken_gamma)
        summarize_buttons = [b for b in app.button if b.label == "📝"]
        assert len(summarize_buttons) == 2

    def test_clicking_summarize_generates_and_shows_summary(self, app: AppTest) -> None:
        summarize_buttons = [b for b in app.button if b.label == "📝"]
        summarize_buttons[0].click().run()

        assert not app.exception
        markdown_values = [m.value for m in app.markdown]
        assert "**Executive Summary**" in markdown_values
        assert "**Key Insights**" in markdown_values
        assert "**Topics**" in markdown_values
        assert any("termination notice" in m for m in markdown_values)

    def test_closing_summary_dialog_hides_content(self, app: AppTest) -> None:
        summarize_buttons = [b for b in app.button if b.label == "📝"]
        summarize_buttons[0].click().run()

        close_button = [b for b in app.button if b.label == "Close"][0]
        close_button.click().run()
        _settle(app)

        assert not app.exception
        markdown_values = [m.value for m in app.markdown]
        assert "**Executive Summary**" not in markdown_values

    def test_regenerate_still_shows_summary(self, app: AppTest) -> None:
        summarize_buttons = [b for b in app.button if b.label == "📝"]
        summarize_buttons[0].click().run()

        regenerate_button = [b for b in app.button if "Regenerate" in b.label][0]
        regenerate_button.click().run()

        assert not app.exception
        markdown_values = [m.value for m in app.markdown]
        assert "**Executive Summary**" in markdown_values


class TestDeleteConfirmation:
    def test_clicking_delete_opens_confirmation_dialog(self, app: AppTest) -> None:
        delete_buttons = [b for b in app.button if b.label == "🗑"]
        delete_buttons[0].click().run()

        assert not app.exception
        assert any(b.label == "Cancel" for b in app.button)
        assert any(b.label == "Delete" for b in app.button)

    def test_cancel_does_not_delete(self, app: AppTest) -> None:
        delete_buttons = [b for b in app.button if b.label == "🗑"]
        delete_buttons[0].click().run()
        cancel_button = [b for b in app.button if b.label == "Cancel"][0]
        cancel_button.click().run()
        _settle(app)

        assert not app.exception
        assert len(app.markdown) == 3  # still all 3 documents

    def test_confirming_delete_removes_document(self, app: AppTest) -> None:
        delete_buttons = [b for b in app.button if b.label == "🗑"]
        delete_buttons[0].click().run()
        confirm_button = [b for b in app.button if b.label == "Delete"][0]
        confirm_button.click().run()
        _settle(app)

        assert not app.exception
        assert len(app.markdown) == 2

    def test_deleted_document_gone_after_deletion(self, app: AppTest) -> None:
        app.selectbox[0].set_value("Name (A-Z)").run()  # broken_gamma, contract_alpha, policy_beta
        delete_buttons = [b for b in app.button if b.label == "🗑"]
        delete_buttons[0].click().run()  # deletes broken_gamma (first in this sort order)
        confirm_button = [b for b in app.button if b.label == "Delete"][0]
        confirm_button.click().run()
        _settle(app)

        titles = [md.value for md in app.markdown]
        assert "**broken_gamma.pdf**" not in titles
        assert len(titles) == 2
