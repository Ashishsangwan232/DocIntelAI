"""
splitter.py
===========
Wraps LangChain's `RecursiveCharacterTextSplitter` to turn a cleaned
`LoadedDocument` into a list of typed `Chunk` dataclasses ready for
persistence and embedding.

Chunk size/overlap are read from `config.settings.rag` by default but
can be overridden per call — this is what powers the "Chunking"
section of the Settings page in Phase 10, without the splitter needing
to know anything about Streamlit.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.database.models import Chunk
from src.loaders.base_loader import LoadedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Separator priority: paragraph breaks first, then lines, then
# sentences, then words — falls back to hard character split only as
# a last resort, which keeps chunks semantically coherent.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class TextSplitter:
    """Splits cleaned document text into overlapping, size-bounded chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.rag.chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=_SEPARATORS,
            length_function=len,
        )

    def split_document(self, document: LoadedDocument, document_id: str) -> list[Chunk]:
        """
        Split every page of a `LoadedDocument` into `Chunk` records.

        Chunk indices are assigned continuously across the whole
        document (not reset per page), so `chunk_index` always
        reflects a chunk's position in the full document — required
        for stable citation ordering in the chat UI.

        Args:
            document: The cleaned, loaded document to split.
            document_id: The `Document.id` these chunks belong to.

        Returns:
            A list of `Chunk` objects, empty if the document has no
            usable text.
        """
        chunks: list[Chunk] = []
        running_index = 0

        for page in document.pages:
            if not page.text.strip():
                continue

            page_splits = self._splitter.split_text(page.text)
            for split_text in page_splits:
                if not split_text.strip():
                    continue
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_index=running_index,
                        content=split_text.strip(),
                        page_number=page.page_number,
                        token_count=self._estimate_token_count(split_text),
                    )
                )
                running_index += 1

        logger.info(
            "Split document %s into %d chunks (size=%d, overlap=%d)",
            document_id, len(chunks), self.chunk_size, self.chunk_overlap,
        )
        return chunks

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """
        Cheap whitespace-based token estimate for storage/analytics.

        Not used for LLM context-window math (the LLM service does its
        own accounting) — this exists purely so the UI can display an
        approximate token count without a tokenizer dependency.
        """
        return len(text.split())
