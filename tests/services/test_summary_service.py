"""Unit tests for src/services/summary_service.py"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from src.database.models import Chunk, Document, FileType
from src.database.sqlite_manager import SQLiteManager
from src.llm.base import BaseLLM, LLMResponse
from src.services.summary_service import SummaryService, _parse_summary_json
from src.utils.exceptions import LLMResponseError, SummaryGenerationError, ValidationError

_VALID_SUMMARY_JSON = json.dumps({
    "executive_summary": "This document covers vendor obligations and termination terms.",
    "key_insights": ["30 day notice required", "Warranty capped at 12 months"],
    "topics": ["termination", "warranty"],
})


class FakeLLM(BaseLLM):
    """Distinguishes the final structured pass from partial/compress passes by system prompt content."""

    def __init__(self, final_response: str = _VALID_SUMMARY_JSON, partial_response: str = "Partial summary text."):
        self.final_response = final_response
        self.partial_response = partial_response
        self.final_calls = 0
        self.partial_calls = 0
        self.prompts_seen: list[str] = []

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None) -> LLMResponse:
        self.prompts_seen.append(prompt)
        if system_prompt and "JSON object" in system_prompt:
            self.final_calls += 1
            return LLMResponse(content=self.final_response, model="fake", latency_ms=5)
        self.partial_calls += 1
        return LLMResponse(content=self.partial_response, model="fake", latency_ms=5)

    def stream(self, prompt, system_prompt=None, temperature=None, max_tokens=None) -> Iterator[str]:
        yield "unused"


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "test.db")


@pytest.fixture()
def short_document(db: SQLiteManager) -> Document:
    doc = Document(filename="contract.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h1")
    db.create_document(doc)
    db.bulk_create_chunks([
        Chunk(document_id=doc.id, chunk_index=i, content=f"Clause {i} text about obligations.")
        for i in range(3)
    ])
    return doc


class TestGenerateSummaryShortDocument:
    def test_returns_parsed_summary(self, db: SQLiteManager, short_document: Document) -> None:
        fake_llm = FakeLLM()
        service = SummaryService(llm=fake_llm, db=db)

        summary = service.generate_summary(short_document.id)

        assert summary.executive_summary == "This document covers vendor obligations and termination terms."
        assert summary.key_insights == ["30 day notice required", "Warranty capped at 12 months"]
        assert summary.topics == ["termination", "warranty"]
        assert summary.document_id == short_document.id
        assert summary.filename == "contract.pdf"

    def test_does_not_use_map_reduce_for_short_document(
        self, db: SQLiteManager, short_document: Document
    ) -> None:
        fake_llm = FakeLLM()
        service = SummaryService(llm=fake_llm, db=db)
        summary = service.generate_summary(short_document.id)

        assert summary.used_map_reduce is False
        assert fake_llm.final_calls == 1
        assert fake_llm.partial_calls == 0

    def test_source_chunk_count_recorded(self, db: SQLiteManager, short_document: Document) -> None:
        service = SummaryService(llm=FakeLLM(), db=db)
        summary = service.generate_summary(short_document.id)
        assert summary.source_chunk_count == 3

    def test_document_with_no_chunks_raises(self, db: SQLiteManager) -> None:
        empty_doc = Document(filename="empty.txt", file_type=FileType.TXT, file_size_bytes=1, file_hash="h2")
        db.create_document(empty_doc)
        service = SummaryService(llm=FakeLLM(), db=db)

        with pytest.raises(ValidationError):
            service.generate_summary(empty_doc.id)


class TestGenerateSummaryLongDocument:
    def test_uses_map_reduce_when_exceeding_context_budget(self, db: SQLiteManager) -> None:
        doc = Document(filename="huge_report.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h3")
        db.create_document(doc)
        db.bulk_create_chunks([
            Chunk(document_id=doc.id, chunk_index=i, content="word " * 200) for i in range(10)
        ])

        fake_llm = FakeLLM()
        service = SummaryService(llm=fake_llm, db=db, max_context_chars=3000)
        summary = service.generate_summary(doc.id)

        assert summary.used_map_reduce is True
        assert fake_llm.partial_calls >= 2
        assert fake_llm.final_calls == 1

    def test_groups_respect_character_budget(self, db: SQLiteManager) -> None:
        doc = Document(filename="report.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h4")
        db.create_document(doc)
        chunks = [Chunk(document_id=doc.id, chunk_index=i, content="x" * 500) for i in range(20)]
        db.bulk_create_chunks(chunks)

        service = SummaryService(llm=FakeLLM(), db=db, max_context_chars=1000)
        groups = service._group_chunks(chunks)

        for group in groups:
            total_len = sum(len(c.content) for c in group)
            # Each group should be near the budget, and a single
            # chunk's own length should never be split mid-chunk.
            assert total_len <= 1000 or len(group) == 1

    def test_final_summary_still_well_formed_after_map_reduce(self, db: SQLiteManager) -> None:
        doc = Document(filename="huge.pdf", file_type=FileType.PDF, file_size_bytes=1, file_hash="h5")
        db.create_document(doc)
        db.bulk_create_chunks([
            Chunk(document_id=doc.id, chunk_index=i, content="content " * 300) for i in range(8)
        ])
        service = SummaryService(llm=FakeLLM(), db=db, max_context_chars=2000)
        summary = service.generate_summary(doc.id)

        assert summary.executive_summary
        assert isinstance(summary.key_insights, list)
        assert isinstance(summary.topics, list)


class TestJSONParsing:
    def test_parses_clean_json(self) -> None:
        result = _parse_summary_json(_VALID_SUMMARY_JSON)
        assert result["executive_summary"] == "This document covers vendor obligations and termination terms."
        assert len(result["key_insights"]) == 2

    def test_parses_json_wrapped_in_markdown_fences(self) -> None:
        fenced = f"```json\n{_VALID_SUMMARY_JSON}\n```"
        result = _parse_summary_json(fenced)
        assert result["executive_summary"] == "This document covers vendor obligations and termination terms."

    def test_parses_json_with_leading_commentary(self) -> None:
        content = f"Here is the summary:\n\n{_VALID_SUMMARY_JSON}"
        result = _parse_summary_json(content)
        assert "termination" in result["topics"]

    def test_missing_executive_summary_key_raises(self) -> None:
        bad_json = json.dumps({"key_insights": ["a"], "topics": ["b"]})
        with pytest.raises(SummaryGenerationError):
            _parse_summary_json(bad_json)

    def test_non_json_content_raises(self) -> None:
        with pytest.raises(SummaryGenerationError):
            _parse_summary_json("This is not JSON at all, sorry.")

    def test_empty_content_raises(self) -> None:
        with pytest.raises(SummaryGenerationError):
            _parse_summary_json("")

    def test_missing_optional_fields_default_to_empty_lists(self) -> None:
        minimal = json.dumps({"executive_summary": "Just a summary."})
        result = _parse_summary_json(minimal)
        assert result["key_insights"] == []
        assert result["topics"] == []

    def test_blank_insights_filtered_out(self) -> None:
        content = json.dumps({
            "executive_summary": "Summary.",
            "key_insights": ["Real insight", "", "   "],
            "topics": [],
        })
        result = _parse_summary_json(content)
        assert result["key_insights"] == ["Real insight"]


class TestLLMErrorPropagation:
    def test_llm_failure_during_final_pass_propagates(
        self, db: SQLiteManager, short_document: Document
    ) -> None:
        class FailingLLM(BaseLLM):
            def generate(self, *a, **kw):
                raise LLMResponseError("simulated outage")

            def stream(self, *a, **kw):
                yield "unused"

        service = SummaryService(llm=FailingLLM(), db=db)
        with pytest.raises(LLMResponseError):
            service.generate_summary(short_document.id)

    def test_malformed_final_response_raises_summary_generation_error(
        self, db: SQLiteManager, short_document: Document
    ) -> None:
        fake_llm = FakeLLM(final_response="not valid json")
        service = SummaryService(llm=fake_llm, db=db)
        with pytest.raises(SummaryGenerationError):
            service.generate_summary(short_document.id)
