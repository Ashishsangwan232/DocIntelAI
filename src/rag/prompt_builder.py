"""
prompt_builder.py
==================
Builds citation-aware prompts from retrieved chunks, and converts
retrieval results into `Citation` objects for display in the chat UI.

Keeping prompt construction in its own class (rather than inline in
`rag/pipeline.py`) means the prompt template can be tuned/versioned
independently of retrieval or LLM-calling logic, and can be unit
tested without any LLM or vector store involved.
"""

from __future__ import annotations

from src.database.models import Citation
from src.rag.retriever import RetrievalResult
from src.utils.helpers import truncate_text

_SYSTEM_PROMPT = """\
You are DocIntel AI, an assistant that answers questions using ONLY the \
document excerpts provided in the user's message as context.

Rules:
1. Base your answer strictly on the provided context. Do not use outside \
knowledge or make assumptions beyond what the excerpts state.
2. Cite the source of every claim using the bracketed source numbers \
provided, e.g. [Source 1], [Source 2].
3. If the context does not contain enough information to answer the \
question, say so plainly instead of guessing.
4. Be concise and direct. Use plain prose unless the question calls for a \
list or structured output.
"""

NO_CONTEXT_MESSAGE = (
    "I couldn't find any relevant information in the uploaded documents to "
    "answer that question. Try rephrasing, or upload a document that covers "
    "this topic."
)

# Truncation length for citation excerpts shown in the UI — long enough
# to give context, short enough to stay a "preview" rather than a full
# reproduction of the chunk.
_EXCERPT_LENGTH = 220


class PromptBuilder:
    """Constructs RAG prompts and citation metadata from retrieval results."""

    def build(self, query: str, retrieval: RetrievalResult) -> tuple[str, str]:
        """
        Build the (system_prompt, user_prompt) pair to send to the LLM.

        Args:
            query: The user's original question.
            retrieval: The `RetrievalResult` from `Retriever.retrieve()`.

        Returns:
            A tuple of `(system_prompt, user_prompt)`. When `retrieval`
            is empty, the user prompt instructs the model to state that
            no relevant context was found, rather than silently
            answering from general knowledge.
        """
        if retrieval.is_empty:
            user_prompt = (
                f"Question: {query}\n\n"
                f"No relevant document excerpts were found for this question. "
                f"Respond with: \"{NO_CONTEXT_MESSAGE}\""
            )
            return _SYSTEM_PROMPT, user_prompt

        context = self._format_context(retrieval)
        user_prompt = (
            f"Context (document excerpts):\n\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer the question using only the context above, and cite "
            f"sources using [Source N] notation."
        )
        return _SYSTEM_PROMPT, user_prompt

    def build_citations(self, retrieval: RetrievalResult) -> list[Citation]:
        """
        Convert a `RetrievalResult` into `Citation` objects for the UI's
        citation chips/panel. Excerpts are truncated so the citation
        panel shows a preview, not the full chunk.
        """
        return [
            Citation(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                similarity_score=chunk.similarity_score,
                excerpt=truncate_text(chunk.content, max_length=_EXCERPT_LENGTH),
                page_number=chunk.page_number,
            )
            for chunk in retrieval.chunks
        ]

    @staticmethod
    def _format_context(retrieval: RetrievalResult) -> str:
        blocks: list[str] = []
        for i, chunk in enumerate(retrieval.chunks, start=1):
            page_suffix = f", page {chunk.page_number}" if chunk.page_number else ""
            blocks.append(
                f"[Source {i}: {chunk.filename}{page_suffix}]\n{chunk.content}"
            )
        return "\n\n".join(blocks)
