"""
base_loader.py
==============
Abstract interface for document loaders.

Every format-specific loader (`PDFLoader`, `DocxLoader`, `TextLoader`)
implements `BaseLoader.load()` and returns a format-agnostic
`LoadedDocument`. This is the seam that lets `document_service.py`
process any supported file type through identical downstream code
(cleaning, splitting, embedding) without ever knowing whether the
source was a PDF, DOCX, or plain text file.

Adding a new format later (e.g. `.pptx`, `.html`) means writing one
new `BaseLoader` subclass — no other file in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageContent:
    """
    Raw extracted text for a single logical unit of a document.

    For PDFs, a unit is a real page. For DOCX/TXT/Markdown, which have
    no reliable page boundaries, the entire document is represented as
    a single `PageContent` with `page_number=None` — downstream code
    treats "unknown page" as a first-class, expected case rather than
    an error.
    """

    text: str
    page_number: int | None = None


@dataclass
class LoadedDocument:
    """The unified output of any loader, ready for cleaning + splitting."""

    source_filename: str
    pages: list[PageContent] = field(default_factory=list)

    @property
    def page_count(self) -> int | None:
        """
        Number of real pages, or None when the format has no page
        concept (DOCX/TXT/MD). Distinguishing "1 page" from "no page
        concept" matters for accurate UI metadata.
        """
        if not self.pages:
            return 0
        if all(p.page_number is None for p in self.pages):
            return None
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """Concatenate all page text, double-newline separated."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    def is_empty(self) -> bool:
        """True if extraction produced no usable text at all."""
        return len(self.full_text.strip()) == 0


class BaseLoader(ABC):
    """Abstract interface every format-specific document loader implements."""

    #: File extensions (lowercase, no dot) this loader supports.
    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, file_path: Path) -> LoadedDocument:
        """
        Extract text content from a file on disk.

        Args:
            file_path: Absolute path to the file to load.

        Returns:
            A `LoadedDocument` with one or more `PageContent` entries.

        Raises:
            CorruptedFileError: If the file exists but cannot be parsed.
        """
        raise NotImplementedError
