"""
helpers.py
==========
Small, pure, reusable utility functions shared across the codebase.

Kept deliberately free of business logic and external dependencies
(besides the standard library) so it can be imported anywhere without
risk of circular imports.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone

# Characters considered unsafe in filenames across common filesystems.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")
_MULTIPLE_UNDERSCORES = re.compile(r"_{2,}")
_UNDERSCORE_BEFORE_DOT = re.compile(r"_+\.")


def sanitize_filename(filename: str) -> str:
    """
    Produce a filesystem-safe filename while preserving the extension
    and enough of the original name to remain human-readable.

    Example:
        "My Report (Final) v2.pdf" -> "My_Report_Final_v2.pdf"
    """
    name = filename.strip().replace(" ", "_")
    name = _UNSAFE_FILENAME_CHARS.sub("_", name)
    name = _MULTIPLE_UNDERSCORES.sub("_", name)
    name = _UNDERSCORE_BEFORE_DOT.sub(".", name)
    name = name.strip("._") or "unnamed_file"
    return name


def generate_uuid() -> str:
    """Generate a UUID4 string used as a primary key across the app."""
    return str(uuid.uuid4())


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute a SHA-256 hash of file contents, used for duplicate
    detection at upload time. Two uploads with identical content
    hash to the same value regardless of filename.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def get_file_extension(filename: str) -> str:
    """Return the lowercase extension of a filename, without the dot."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def format_bytes(size_bytes: int) -> str:
    """Format a byte count into a human-readable string (KB, MB, GB)."""
    if size_bytes < 0:
        raise ValueError("size_bytes must be non-negative")

    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < step:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= step
    return f"{size:.1f} PB"


def utc_now() -> datetime:
    """Return the current UTC timestamp (timezone-aware)."""
    return datetime.now(timezone.utc)


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to a maximum length, appending a suffix if cut."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)].rstrip() + suffix
