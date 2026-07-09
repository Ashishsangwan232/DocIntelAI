"""
Unit tests for src/ui/chat.py

Uses Streamlit's official `AppTest` framework to run `chat.py` as a
real, live Streamlit script (not a mock of Streamlit) against
`tests/ui/chat_app_harness.py`, which wires in a fake LLM/retriever so
no network calls are made. This verifies actual widget behavior
(button clicks, chat input, reruns) rather than just checking that the
render function doesn't raise in isolation.

Note: after any interaction that triggers an internal `st.rerun()`
(sending a message, clicking Regenerate/Clear/New Chat), a follow-up
`.run()` is needed to reach the final stable DOM — `AppTest` surfaces
the intermediate frame otherwise. This is a testing-framework quirk,
not application behavior; `_settle()` encapsulates it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

_HARNESS_PATH = str(Path(__file__).parent / "chat_app_harness.py")
_HARNESS_DB_PATH = Path("/tmp/chat_ui_harness.db")


def _settle(at: AppTest) -> AppTest:
    """Run once more to reach the stable post-rerun DOM state."""
    at.run()
    return at


@pytest.fixture(autouse=True)
def _fresh_harness_db():
    """
    Ensure each test starts with a genuinely fresh harness database.

    `st.cache_resource` caches across the entire pytest process (it is
    not scoped per-`AppTest` instance), so merely deleting the DB file
    is not enough — a stale cached `SQLiteManager` from a previous test
    would still be returned, now pointing at a file that no longer
    exists. Clearing the resource cache forces the harness script to
    reconstruct every cached object from scratch on the next `.run()`.
    """
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

    def test_shows_empty_state_message(self, app: AppTest) -> None:
        assert len(app.info) == 1
        assert "ask a question" in app.info[0].value.lower()

    def test_shows_toolbar_buttons(self, app: AppTest) -> None:
        labels = [b.label for b in app.button]
        assert "New Chat" in labels
        assert "Clear Conversation" in labels

    def test_shows_document_scope_selector(self, app: AppTest) -> None:
        assert len(app.multiselect) == 1
        assert "contract.pdf" in app.multiselect[0].options

    def test_shows_chat_input(self, app: AppTest) -> None:
        assert len(app.chat_input) == 1


class TestSendingMessages:
    def test_sending_message_renders_user_and_assistant_bubbles(self, app: AppTest) -> None:
        app.chat_input[0].set_value("How do I terminate this contract?").run()
        _settle(app)

        assert not app.exception
        roles = [m.name for m in app.chat_message]
        assert roles == ["user", "assistant"]
        assert "terminate" in app.chat_message[0].markdown[0].value.lower()

    def test_assistant_response_includes_citation_content(self, app: AppTest) -> None:
        app.chat_input[0].set_value("termination question").run()
        _settle(app)

        assistant = [m for m in app.chat_message if m.name == "assistant"][0]
        combined_text = " ".join(md.value for md in assistant.markdown)
        assert "contract.pdf" in combined_text
        assert "90% match" in combined_text

    def test_empty_state_disappears_after_first_message(self, app: AppTest) -> None:
        app.chat_input[0].set_value("a question").run()
        _settle(app)
        assert len(app.info) == 0


class TestRegenerate:
    def test_regenerate_button_appears_only_on_last_assistant_message(
        self, app: AppTest
    ) -> None:
        app.chat_input[0].set_value("question one").run()
        _settle(app)
        regen_buttons = [b for b in app.button if b.label == "🔄"]
        assert len(regen_buttons) == 1

    def test_clicking_regenerate_produces_new_response(self, app: AppTest) -> None:
        app.chat_input[0].set_value("question one").run()
        _settle(app)

        regen_button = [b for b in app.button if b.label == "🔄"][0]
        regen_button.click().run()
        _settle(app)

        assert not app.exception
        assistant = [m for m in app.chat_message if m.name == "assistant"][0]
        assert "Answer #2" in assistant.markdown[0].value

    def test_regenerate_does_not_duplicate_messages(self, app: AppTest) -> None:
        app.chat_input[0].set_value("question one").run()
        _settle(app)
        regen_button = [b for b in app.button if b.label == "🔄"][0]
        regen_button.click().run()
        _settle(app)

        assert len(app.chat_message) == 2  # still just user + assistant


class TestClearConversation:
    def test_clear_removes_all_messages(self, app: AppTest) -> None:
        app.chat_input[0].set_value("a question").run()
        _settle(app)

        clear_button = [b for b in app.button if b.label == "Clear Conversation"][0]
        clear_button.click().run()
        _settle(app)

        assert not app.exception
        assert len(app.chat_message) == 0
        assert len(app.info) == 1  # empty-state message reappears


class TestExport:
    def test_export_menu_not_shown_with_no_messages(self, app: AppTest) -> None:
        assert len(app.download_button) == 0

    def test_export_menu_appears_after_first_message(self, app: AppTest) -> None:
        app.chat_input[0].set_value("How do I terminate this contract?").run()
        _settle(app)

        assert not app.exception
        labels = [b.label for b in app.download_button]
        assert "Download as PDF" in labels
        assert "Download as Markdown" in labels

    def test_copy_conversation_code_block_contains_transcript(self, app: AppTest) -> None:
        app.chat_input[0].set_value("How do I terminate this contract?").run()
        _settle(app)

        code_values = [c.value for c in app.code]
        assert any("How do I terminate this contract?" in c for c in code_values)
        assert any("30 days notice required" in c for c in code_values)

    def test_export_persists_after_regenerate(self, app: AppTest) -> None:
        app.chat_input[0].set_value("question one").run()
        _settle(app)
        regen_button = [b for b in app.button if b.label == "🔄"][0]
        regen_button.click().run()
        _settle(app)

        assert not app.exception
        labels = [b.label for b in app.download_button]
        assert "Download as PDF" in labels

    def test_export_menu_disappears_after_clear_conversation(self, app: AppTest) -> None:
        app.chat_input[0].set_value("a question").run()
        _settle(app)
        clear_button = [b for b in app.button if b.label == "Clear Conversation"][0]
        clear_button.click().run()
        _settle(app)

        assert not app.exception
        assert len(app.download_button) == 0


class TestNewChat:
    def test_new_chat_starts_empty_session(self, app: AppTest) -> None:
        app.chat_input[0].set_value("a question").run()
        _settle(app)
        assert len(app.chat_message) == 2

        new_chat_button = [b for b in app.button if b.label == "New Chat"][0]
        new_chat_button.click().run()
        _settle(app)

        assert not app.exception
        assert len(app.chat_message) == 0
