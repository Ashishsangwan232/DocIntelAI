"""
txt_loader.py
=============
Plain-text extraction for `.txt` and `.md` files.

Markdown is intentionally handled by the same loader as plain text
rather than a separate `MarkdownLoader`: at the text-extraction stage
Markdown syntax (`#`, `**`, `-`) is just characters in the text, and
stripping it is a *cleaning* concern (handled later in `cleaner.py`),
not a loading concern. Splitting this into two loader classes would
duplicate identical file-reading logic for no benefit.
"""

from __future__ import annotations

from pathlib import Path

from src.loaders.base_loader import BaseLoader, LoadedDocument, PageContent
from src.utils.exceptions import CorruptedFileError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Encodings tried in order — covers the vast majority of real-world
# plain-text files without pulling in a heavier dependency like chardet.
_FALLBACK_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")


class TextLoader(BaseLoader):
    """Extracts raw text from `.txt` and `.md` files."""

    supported_extensions = ("txt", "md")

    def load(self, file_path: Path) -> LoadedDocument:
        text = self._read_with_fallback_encoding(file_path)
        logger.info(
            "Loaded text file '%s' (%d characters)", file_path.name, len(text)
        )
        return LoadedDocument(
            source_filename=file_path.name,
            pages=[PageContent(text=text, page_number=None)],
        )

    @staticmethod
    def _read_with_fallback_encoding(file_path: Path) -> str:
        last_error: Exception | None = None
        for encoding in _FALLBACK_ENCODINGS:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError) as exc:
                last_error = exc
                continue
            except OSError as exc:
                logger.error("Failed to read file %s: %s", file_path.name, exc)
                raise CorruptedFileError(
                    f"Could not read '{file_path.name}': {exc}"
                ) from exc

        logger.error(
            "All encodings failed for %s: %s", file_path.name, last_error
        )
        raise CorruptedFileError(
            f"'{file_path.name}' could not be decoded with any supported encoding."
        )
