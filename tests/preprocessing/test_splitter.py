"""Unit tests for src/preprocessing/splitter.py"""

from __future__ import annotations

import pytest

from src.loaders.base_loader import LoadedDocument, PageContent
from src.preprocessing.splitter import TextSplitter


class TestTextSplitter:
    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            TextSplitter(chunk_size=100, chunk_overlap=150)

    def test_empty_document_produces_no_chunks(self) -> None:
        splitter = TextSplitter(chunk_size=200, chunk_overlap=20)
        doc = LoadedDocument(source_filename="empty.txt", pages=[PageContent(text="")])
        chunks = splitter.split_document(doc, document_id="doc-1")
        assert chunks == []

    def test_short_text_produces_single_chunk(self) -> None:
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        doc = LoadedDocument(
            source_filename="short.txt",
            pages=[PageContent(text="A short sentence.", page_number=None)],
        )
        chunks = splitter.split_document(doc, document_id="doc-1")
        assert len(chunks) == 1
        assert chunks[0].content == "A short sentence."
        assert chunks[0].chunk_index == 0

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        long_text = "This is a sentence about contracts. " * 20
        doc = LoadedDocument(
            source_filename="long.txt",
            pages=[PageContent(text=long_text, page_number=None)],
        )
        chunks = splitter.split_document(doc, document_id="doc-1")
        assert len(chunks) > 1
        assert all(c.document_id == "doc-1" for c in chunks)

    def test_chunk_index_is_continuous_across_pages(self) -> None:
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        doc = LoadedDocument(
            source_filename="multi.pdf",
            pages=[
                PageContent(text="Page one content here. " * 5, page_number=1),
                PageContent(text="Page two content here. " * 5, page_number=2),
            ],
        )
        chunks = splitter.split_document(doc, document_id="doc-1")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))  # 0, 1, 2, 3...

    def test_page_number_preserved_per_chunk(self) -> None:
        splitter = TextSplitter(chunk_size=500, chunk_overlap=20)
        doc = LoadedDocument(
            source_filename="paged.pdf",
            pages=[
                PageContent(text="Page one text.", page_number=1),
                PageContent(text="Page two text.", page_number=2),
            ],
        )
        chunks = splitter.split_document(doc, document_id="doc-1")
        assert chunks[0].page_number == 1
        assert chunks[1].page_number == 2

    def test_none_page_number_preserved_for_docx_like_input(self) -> None:
        splitter = TextSplitter(chunk_size=500, chunk_overlap=20)
        doc = LoadedDocument(
            source_filename="doc.docx",
            pages=[PageContent(text="Content with no page concept.", page_number=None)],
        )
        chunks = splitter.split_document(doc, document_id="doc-1")
        assert chunks[0].page_number is None

    def test_token_count_is_populated(self) -> None:
        splitter = TextSplitter(chunk_size=500, chunk_overlap=20)
        doc = LoadedDocument(
            source_filename="doc.txt",
            pages=[PageContent(text="one two three four five", page_number=None)],
        )
        chunks = splitter.split_document(doc, document_id="doc-1")
        assert chunks[0].token_count == 5

    def test_blank_pages_are_skipped(self) -> None:
        splitter = TextSplitter(chunk_size=500, chunk_overlap=20)
        doc = LoadedDocument(
            source_filename="doc.pdf",
            pages=[
                PageContent(text="", page_number=1),
                PageContent(text="   ", page_number=2),
                PageContent(text="Real content.", page_number=3),
            ],
        )
        chunks = splitter.split_document(doc, document_id="doc-1")
        assert len(chunks) == 1
        assert chunks[0].page_number == 3
