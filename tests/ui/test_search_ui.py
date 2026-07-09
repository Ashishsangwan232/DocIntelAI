"""
Unit tests for src/ui/search.py

Uses Streamlit's official `AppTest` framework against
`tests/ui/search_app_harness.py`, with a fake embedding model
(deterministic, no network) but real `SearchService`/`Retriever`/
`ChromaManager` logic underneath.

Each test gets a unique storage directory via the
`SEARCH_UI_HARNESS_DIR` env var (backed by pytest's `tmp_path`) rather
than a fixed path. This is required, not just tidy: ChromaDB's Rust
backend maintains an internal by-path client registry that does not
clean up correctly when a directory is deleted and recreated at the
same path within one process — reusing a fixed path across sequential
`AppTest` runs in the same pytest session intermittently raised
"attempt to write a readonly database" starting on the second test
(confirmed via minimal reproduction, independent of this app's code).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

_HARNESS_PATH = str(Path(__file__).parent / "search_app_harness.py")


@pytest.fixture(autouse=True)
def _fresh_harness_state(tmp_path: Path):
    """Unique storage directory + cleared Streamlit resource cache for every test."""
    st.cache_resource.clear()
    os.environ["SEARCH_UI_HARNESS_DIR"] = str(tmp_path / "search_harness")
    yield
    st.cache_resource.clear()
    os.environ.pop("SEARCH_UI_HARNESS_DIR", None)


@pytest.fixture()
def app() -> AppTest:
    at = AppTest.from_file(_HARNESS_PATH, default_timeout=30)
    at.run()
    return at


class TestInitialRender:
    def test_renders_without_exceptions(self, app: AppTest) -> None:
        assert not app.exception

    def test_shows_search_box_and_scope_selector(self, app: AppTest) -> None:
        assert len(app.text_input) == 1
        assert len(app.multiselect) == 1

    def test_no_results_shown_before_searching(self, app: AppTest) -> None:
        assert len(app.info) == 0  # no "no results" message until a search runs


class TestSearching:
    def test_relevant_query_returns_result_with_score_and_page(self, app: AppTest) -> None:
        app.text_input[0].set_value("30 days notice").run()

        assert not app.exception
        captions = [c.value for c in app.caption]
        assert any("100% match" in c or "match" in c for c in captions)
        markdown_values = [m.value for m in app.markdown]
        assert any("contract.pdf" in m and "page 4" in m for m in markdown_values)

    def test_relevant_query_highlights_matching_terms(self, app: AppTest) -> None:
        app.text_input[0].set_value("30 days notice").run()
        markdown_values = [m.value for m in app.markdown]
        combined = " ".join(markdown_values)
        assert "**notice**" in combined or "**30**" in combined or "**days**" in combined

    def test_unrelated_query_shows_no_results_message(self, app: AppTest) -> None:
        app.text_input[0].set_value("completely unrelated gibberish topic").run()

        assert not app.exception
        assert len(app.info) == 1
        assert "no results found" in app.info[0].value.lower()

    def test_result_count_caption_shown(self, app: AppTest) -> None:
        app.text_input[0].set_value("30 days notice").run()
        captions = [c.value for c in app.caption]
        assert any("1 result" in c for c in captions)


class TestScopeSelector:
    def test_scope_selector_shows_ready_documents(self, app: AppTest) -> None:
        assert "contract.pdf" in app.multiselect[0].options

    def test_scope_defaults_to_empty_meaning_all_documents(self, app: AppTest) -> None:
        assert app.multiselect[0].value == []
