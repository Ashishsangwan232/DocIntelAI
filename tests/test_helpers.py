"""Unit tests for src/utils/helpers.py"""

from __future__ import annotations

import pytest

from src.utils.helpers import (
    compute_file_hash,
    format_bytes,
    get_file_extension,
    sanitize_filename,
    truncate_text,
)


class TestSanitizeFilename:
    def test_replaces_spaces_with_underscores(self) -> None:
        assert sanitize_filename("My Report.pdf") == "My_Report.pdf"

    def test_strips_unsafe_characters(self) -> None:
        result = sanitize_filename("My Report (Final) v2.pdf")
        assert result == "My_Report_Final_v2.pdf"
        assert all(c.isalnum() or c in "_.-" for c in result)

    def test_collapses_multiple_underscores(self) -> None:
        assert sanitize_filename("a___b.txt") == "a_b.txt"

    def test_falls_back_for_empty_result(self) -> None:
        assert sanitize_filename("***").startswith("unnamed_file") or sanitize_filename("***") == "unnamed_file"


class TestFileHash:
    def test_same_content_same_hash(self) -> None:
        h1 = compute_file_hash(b"hello world")
        h2 = compute_file_hash(b"hello world")
        assert h1 == h2

    def test_different_content_different_hash(self) -> None:
        assert compute_file_hash(b"a") != compute_file_hash(b"b")

    def test_hash_is_sha256_hex_length(self) -> None:
        assert len(compute_file_hash(b"anything")) == 64


class TestFileExtension:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("report.PDF", "pdf"),
            ("notes.md", "md"),
            ("archive.tar.gz", "gz"),
            ("noextension", ""),
        ],
    )
    def test_extraction(self, filename: str, expected: str) -> None:
        assert get_file_extension(filename) == expected


class TestFormatBytes:
    @pytest.mark.parametrize(
        "size,expected_unit",
        [(500, "B"), (2048, "KB"), (5 * 1024 * 1024, "MB")],
    )
    def test_units(self, size: int, expected_unit: str) -> None:
        assert expected_unit in format_bytes(size)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            format_bytes(-1)


class TestTruncateText:
    def test_short_text_unchanged(self) -> None:
        assert truncate_text("hello", max_length=100) == "hello"

    def test_long_text_truncated_with_suffix(self) -> None:
        result = truncate_text("a" * 300, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")
