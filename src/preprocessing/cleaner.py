"""
cleaner.py
==========
Normalizes raw extracted text before it is split into chunks and
embedded. Garbage in the cleaning stage becomes garbage embeddings,
so this step directly affects retrieval quality.

Deliberately conservative: it removes noise (control characters,
excessive whitespace, encoding artifacts) without rewriting or
summarizing content, since the RAG pipeline must answer from the
document's actual wording.
"""

from __future__ import annotations

import re
import unicodedata

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Three or more consecutive newlines collapse to exactly two
# (preserves paragraph breaks, removes excessive vertical whitespace
# common in PDF extraction around page margins/headers).
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Runs of horizontal whitespace (spaces/tabs) collapse to a single space.
_EXCESS_SPACES = re.compile(r"[ \t]{2,}")

# Control characters (except \n and \t) that sometimes leak in from
# PDF extraction and break downstream tokenization.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Common ligature / typography artifacts from PDF font encoding.
_LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",  # non-breaking space
}


class TextCleaner:
    """Normalizes raw document text for chunking and embedding."""

    def clean(self, text: str) -> str:
        """
        Apply the full cleaning pipeline to a block of text.

        Args:
            text: Raw text extracted by a loader.

        Returns:
            Cleaned text, safe to hand to the splitter. Returns an
            empty string if input is empty or whitespace-only.
        """
        if not text or not text.strip():
            return ""

        cleaned = unicodedata.normalize("NFKC", text)
        cleaned = self._replace_ligatures(cleaned)
        cleaned = _CONTROL_CHARS.sub("", cleaned)
        cleaned = _EXCESS_SPACES.sub(" ", cleaned)
        cleaned = _EXCESS_NEWLINES.sub("\n\n", cleaned)
        cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))

        return cleaned.strip()

    @staticmethod
    def _replace_ligatures(text: str) -> str:
        for artifact, replacement in _LIGATURE_MAP.items():
            if artifact in text:
                text = text.replace(artifact, replacement)
        return text
