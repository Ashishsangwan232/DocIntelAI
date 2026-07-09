"""Unit tests for src/preprocessing/cleaner.py"""

from __future__ import annotations

from src.preprocessing.cleaner import TextCleaner


class TestTextCleaner:
    def setup_method(self) -> None:
        self.cleaner = TextCleaner()

    def test_empty_input_returns_empty(self) -> None:
        assert self.cleaner.clean("") == ""
        assert self.cleaner.clean("   \n\n  ") == ""

    def test_collapses_excess_newlines(self) -> None:
        result = self.cleaner.clean("Para one.\n\n\n\n\nPara two.")
        assert "\n\n\n" not in result
        assert "Para one." in result and "Para two." in result

    def test_collapses_excess_spaces(self) -> None:
        result = self.cleaner.clean("Word1     Word2")
        assert "Word1 Word2" == result

    def test_strips_control_characters(self) -> None:
        result = self.cleaner.clean("Hello\x00World\x0b!")
        assert "\x00" not in result
        assert "\x0b" not in result

    def test_normalizes_ligatures(self) -> None:
        result = self.cleaner.clean("e\ufb03cient o\ufb04ine")
        assert "efficient" in result
        assert "offline" in result

    def test_normalizes_smart_quotes(self) -> None:
        result = self.cleaner.clean("\u201cHello\u201d \u2018world\u2019")
        assert result == '"Hello" \'world\''

    def test_normalizes_non_breaking_space(self) -> None:
        result = self.cleaner.clean("Hello\u00a0World")
        assert "\u00a0" not in result

    def test_preserves_meaningful_content(self) -> None:
        original = "This is a normal sentence with punctuation, and numbers 123."
        result = self.cleaner.clean(original)
        assert result == original

    def test_strips_leading_trailing_whitespace_per_line(self) -> None:
        result = self.cleaner.clean("  Line one  \n  Line two  ")
        lines = result.split("\n")
        assert all(line == line.strip() for line in lines)
