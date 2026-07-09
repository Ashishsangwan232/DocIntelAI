"""
summary_service.py
===================
Generates AI summaries of a document: an executive summary, key
insights, and important topics.

Deliberately distinct from `rag/pipeline.py`: chat/search retrieve the
top-K *relevant* chunks for a specific question, but summarization
needs the *whole* document's content, in order. This service reads
every chunk for a document (already split/cleaned by the Phase 3
pipeline) rather than doing a similarity search.

Long documents (whose full text would exceed a reasonable LLM context
budget) are handled with a simple map-reduce: chunks are grouped into
batches under a character budget, each batch gets a plain-text partial
summary, and the partial summaries are combined into one final
structured summary. Short documents skip straight to the final pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from src.database.models import Chunk, Document
from src.database.sqlite_manager import SQLiteManager
from src.llm.base import BaseLLM
from src.llm.ollama_cloud import OllamaCloudLLM
from src.utils.exceptions import LLMError, SummaryGenerationError, ValidationError
from src.utils.helpers import utc_now
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Character budget per LLM call. Conservative relative to typical
# context windows (roughly 4 chars/token, so ~3000 tokens of source
# text) to leave headroom for the prompt scaffolding and the model's
# own output.
_MAX_CONTEXT_CHARS = 12_000

_FINAL_SYSTEM_PROMPT = """\
You are DocIntel AI, generating a structured summary of a document.

Respond with ONLY a single JSON object — no markdown code fences, no \
commentary before or after — matching exactly this schema:

{
  "executive_summary": "2-4 sentence high-level summary",
  "key_insights": ["insight 1", "insight 2", "..."],
  "topics": ["topic 1", "topic 2", "..."]
}

Rules:
- executive_summary: concise, plain prose, no bullet points.
- key_insights: 3-6 short, specific, factual bullet points from the document.
- topics: 3-8 short topic/keyword phrases (2-4 words each) covering the \
document's main subject areas.
- Base everything strictly on the provided text. Do not invent information.
"""

_PARTIAL_SYSTEM_PROMPT = """\
You are DocIntel AI. Summarize the following excerpt from a longer document \
in 3-5 concise sentences, preserving concrete facts, figures, and named \
entities. Plain prose, no headers, no bullet points. This is an intermediate \
summary that will later be combined with summaries of other excerpts from \
the same document.
"""


@dataclass
class DocumentSummary:
    """The generated summary of a single document."""

    document_id: str
    filename: str
    executive_summary: str
    key_insights: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    source_chunk_count: int = 0
    used_map_reduce: bool = False
    generated_at: datetime = field(default_factory=utc_now)


class SummaryService:
    """Generates executive summaries, key insights, and topics for a document."""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        db: SQLiteManager | None = None,
        max_context_chars: int | None = None,
    ) -> None:
        self.llm = llm or OllamaCloudLLM()
        self.db = db or SQLiteManager()
        self.max_context_chars = max_context_chars or _MAX_CONTEXT_CHARS

    def generate_summary(self, document_id: str) -> DocumentSummary:
        """
        Generate a full summary for a document.

        Args:
            document_id: The document to summarize. Must have at least
                one processed chunk (i.e. status READY).

        Returns:
            A `DocumentSummary` with executive summary, key insights,
            and topics.

        Raises:
            ValidationError: If the document has no processed content.
            LLMError: If the underlying LLM call fails.
            SummaryGenerationError: If the LLM's output can't be parsed
                into the expected schema.
        """
        document = self.db.get_document(document_id)
        chunks = self.db.get_chunks_for_document(document_id)

        if not chunks:
            raise ValidationError(
                f"'{document.filename}' has no processed content to summarize."
            )

        full_text = "\n\n".join(c.content for c in chunks)

        if len(full_text) <= self.max_context_chars:
            summary = self._final_pass(full_text, document)
            summary.used_map_reduce = False
        else:
            summary = self._map_reduce_pass(chunks, document)
            summary.used_map_reduce = True

        summary.source_chunk_count = len(chunks)
        logger.info(
            "Generated summary for '%s' (%d chunks, map_reduce=%s)",
            document.filename, len(chunks), summary.used_map_reduce,
        )
        return summary

    # ------------------------------------------------------------------
    # Map-reduce for long documents
    # ------------------------------------------------------------------
    def _map_reduce_pass(self, chunks: list[Chunk], document: Document) -> DocumentSummary:
        groups = self._group_chunks(chunks)
        logger.info(
            "'%s' exceeds context budget — summarizing in %d groups",
            document.filename, len(groups),
        )

        partial_summaries = [self._partial_summary(group, document) for group in groups]
        combined_text = "\n\n".join(partial_summaries)

        # The combined partial summaries are themselves usually well
        # under budget, but guard against pathological cases (huge
        # number of groups) by recursing — each recursive pass
        # compresses the text further until it fits.
        if len(combined_text) > self.max_context_chars:
            logger.info(
                "Combined partial summaries for '%s' still exceed budget — compressing further",
                document.filename,
            )
            combined_text = self._compress_text(combined_text, document)

        return self._final_pass(combined_text, document)

    def _group_chunks(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Greedily group ordered chunks into batches under the character budget."""
        groups: list[list[Chunk]] = []
        current_group: list[Chunk] = []
        current_length = 0

        for chunk in chunks:
            chunk_length = len(chunk.content)
            if current_group and current_length + chunk_length > self.max_context_chars:
                groups.append(current_group)
                current_group = []
                current_length = 0
            current_group.append(chunk)
            current_length += chunk_length

        if current_group:
            groups.append(current_group)

        return groups

    def _partial_summary(self, group: list[Chunk], document: Document) -> str:
        text = "\n\n".join(c.content for c in group)
        try:
            response = self.llm.generate(prompt=text, system_prompt=_PARTIAL_SYSTEM_PROMPT)
            return response.content.strip()
        except LLMError:
            logger.error("Partial summary generation failed for '%s'", document.filename)
            raise

    def _compress_text(self, text: str, document: Document) -> str:
        try:
            response = self.llm.generate(
                prompt=text,
                system_prompt=(
                    "Condense the following into 4-6 sentences preserving the "
                    "most important facts. Plain prose only."
                ),
            )
            return response.content.strip()
        except LLMError:
            logger.error("Compression pass failed for '%s'", document.filename)
            raise

    # ------------------------------------------------------------------
    # Final structured pass
    # ------------------------------------------------------------------
    def _final_pass(self, text: str, document: Document) -> DocumentSummary:
        try:
            response = self.llm.generate(prompt=text, system_prompt=_FINAL_SYSTEM_PROMPT)
        except LLMError:
            logger.error("Final summary pass failed for '%s'", document.filename)
            raise

        parsed = _parse_summary_json(response.content)
        return DocumentSummary(
            document_id=document.id,
            filename=document.filename,
            executive_summary=parsed["executive_summary"],
            key_insights=parsed["key_insights"],
            topics=parsed["topics"],
        )


# ------------------------------------------------------------------
# JSON parsing (module-private)
# ------------------------------------------------------------------
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_summary_json(raw_content: str) -> dict:
    """
    Parse the LLM's summary response into the expected schema.

    Handles the common case of an LLM wrapping JSON in markdown code
    fences despite instructions not to, by extracting the first
    `{...}` block before giving up.

    Raises:
        SummaryGenerationError: If no valid JSON matching the schema
            can be extracted.
    """
    candidates = [raw_content.strip()]

    match = _JSON_OBJECT_PATTERN.search(raw_content)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(data, dict):
            continue
        if "executive_summary" not in data:
            continue

        return {
            "executive_summary": str(data.get("executive_summary", "")).strip(),
            "key_insights": [str(i).strip() for i in data.get("key_insights", []) if str(i).strip()],
            "topics": [str(t).strip() for t in data.get("topics", []) if str(t).strip()],
        }

    logger.error("Could not parse summary JSON from LLM response: %.200s", raw_content)
    raise SummaryGenerationError(
        "The AI's summary response could not be parsed. Please try regenerating the summary."
    )
