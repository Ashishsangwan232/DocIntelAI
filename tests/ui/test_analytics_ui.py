"""
Unit tests for src/ui/analytics.py

Uses Streamlit's official `AppTest` framework against
`tests/ui/analytics_app_harness.py`. No LLM/embedding calls are
involved — analytics and settings are pure read/write over data
already captured by other services — so this harness seeds the
database directly with real rows.

Each test gets a unique storage directory via the
`ANALYTICS_UI_HARNESS_DIR` env var (backed by pytest's `tmp_path`),
consistent with the pattern established in test_search_ui.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

_HARNESS_PATH = str(Path(__file__).parent / "analytics_app_harness.py")


@pytest.fixture(autouse=True)
def _fresh_harness_state(tmp_path: Path):
    st.cache_resource.clear()
    os.environ["ANALYTICS_UI_HARNESS_DIR"] = str(tmp_path / "analytics_harness")
    yield
    st.cache_resource.clear()
    os.environ.pop("ANALYTICS_UI_HARNESS_DIR", None)


@pytest.fixture()
def app() -> AppTest:
    at = AppTest.from_file(_HARNESS_PATH, default_timeout=30)
    at.run()
    return at


def _metric(app: AppTest, label: str) -> str:
    matches = [m.value for m in app.metric if m.label == label]
    assert matches, f"No metric found with label '{label}'"
    return matches[0]


class TestAnalyticsDashboard:
    def test_renders_without_exceptions(self, app: AppTest) -> None:
        assert not app.exception

    def test_shows_correct_document_and_chunk_counts(self, app: AppTest) -> None:
        assert _metric(app, "Total Documents") == "2"
        assert _metric(app, "Total Chunks") == "6"
        assert _metric(app, "Total Embeddings") == "6"

    def test_shows_storage_used(self, app: AppTest) -> None:
        storage = _metric(app, "Storage Used")
        assert "KB" in storage or "MB" in storage or "B" in storage

    def test_shows_chat_session_and_query_counts(self, app: AppTest) -> None:
        assert _metric(app, "Chat Sessions") == "1"
        assert _metric(app, "Total Queries") == "4"

    def test_shows_average_response_time(self, app: AppTest) -> None:
        assert _metric(app, "Avg. Response Time") == "1.5s"

    def test_shows_most_queried_document(self, app: AppTest) -> None:
        assert _metric(app, "Most Queried") == "contract.pdf"

    def test_shows_most_queried_ranking_list(self, app: AppTest) -> None:
        markdown_values = [m.value for m in app.markdown]
        assert any("contract.pdf" in m for m in markdown_values)
        assert any("policy.docx" in m for m in markdown_values)


class TestSettingsPanel:
    def test_shows_all_controls_with_current_values(self, app: AppTest) -> None:
        sliders = {s.label: s.value for s in app.slider}
        assert sliders["Chunk Size (characters)"] == 800
        assert sliders["Chunk Overlap (characters)"] == 120
        assert sliders["Top-K Retrieval"] == 5
        assert sliders["Temperature"] == 0.3
        assert sliders["Maximum Tokens"] == 1024

        selectbox = app.selectbox[0]
        assert selectbox.value == "gpt-oss:120b"

    def test_saving_settings_shows_success_message(self, app: AppTest) -> None:
        top_k_slider = [s for s in app.slider if s.label == "Top-K Retrieval"][0]
        top_k_slider.set_value(12)
        save_button = [b for b in app.button if b.label == "Save Settings"][0]
        save_button.click().run()

        assert not app.exception
        assert any("saved" in s.value.lower() for s in app.success)

    def test_saved_settings_persist_across_fresh_app_load(
        self, app: AppTest, tmp_path: Path
    ) -> None:
        top_k_slider = [s for s in app.slider if s.label == "Top-K Retrieval"][0]
        top_k_slider.set_value(9)
        save_button = [b for b in app.button if b.label == "Save Settings"][0]
        save_button.click().run()

        fresh_app = AppTest.from_file(_HARNESS_PATH, default_timeout=30)
        fresh_app.run()
        fresh_top_k = [s for s in fresh_app.slider if s.label == "Top-K Retrieval"][0]
        assert fresh_top_k.value == 9

    def test_reset_to_defaults_restores_config_values(self, app: AppTest) -> None:
        top_k_slider = [s for s in app.slider if s.label == "Top-K Retrieval"][0]
        top_k_slider.set_value(18)
        save_button = [b for b in app.button if b.label == "Save Settings"][0]
        save_button.click().run()

        reset_button = [b for b in app.button if b.label == "Reset to Defaults"][0]
        reset_button.click().run()
        app.run()

        assert not app.exception
        top_k_after = [s for s in app.slider if s.label == "Top-K Retrieval"][0]
        assert top_k_after.value == 5
