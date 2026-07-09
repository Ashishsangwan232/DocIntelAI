"""
pdf_loader.py
=============
PDF text extraction using PyMuPDF (fitz).

PyMuPDF was chosen over pypdf/PyPDF2 for speed and significantly more
reliable text extraction on real-world PDFs (multi-column layouts,
embedded fonts), which directly improves downstream RAG answer quality.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from src.loaders.base_loader import BaseLoader, LoadedDocument, PageContent
from src.utils.exceptions import CorruptedFileError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    """Extracts per-page text from PDF files."""

    supported_extensions = ("pdf",)

    def load(self, file_path: Path) -> LoadedDocument:
        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:  # PyMuPDF raises its own broad exceptions
            logger.error("Failed to open PDF %s: %s", file_path.name, exc)
            raise CorruptedFileError(
                f"'{file_path.name}' could not be opened — the PDF may be "
                f"corrupted, password-protected, or not a valid PDF."
            ) from exc

        try:
            pages: list[PageContent] = []
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                pages.append(PageContent(text=text.strip(), page_number=page_number))
        except Exception as exc:
            logger.error("Failed to extract text from PDF %s: %s", file_path.name, exc)
            raise CorruptedFileError(
                f"Text extraction failed for '{file_path.name}': {exc}"
            ) from exc
        finally:
            doc.close()

        logger.info("Loaded PDF '%s' with %d pages", file_path.name, len(pages))
        return LoadedDocument(source_filename=file_path.name, pages=pages)
