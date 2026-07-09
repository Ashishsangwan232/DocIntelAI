"""
export_service.py
==================
Exports a chat conversation as Markdown, plain text (for copy-to-
clipboard), or PDF.

PDF generation uses reportlab's Platypus API (`SimpleDocTemplate` +
`Paragraph`), per the project's PDF-creation guidelines — reportlab is
the recommended library for building PDFs from scratch, as opposed to
reading/merging/splitting existing ones. All user content is HTML-
escaped before being placed in a `Paragraph`, since reportlab's
Paragraph markup is itself a small XML-like dialect and unescaped
special characters (`<`, `&`, etc.) in chat content would otherwise
break rendering or be silently dropped.
"""

from __future__ import annotations

import html
import io

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.database.models import ChatMessage, ChatSession, MessageRole
from src.database.sqlite_manager import SQLiteManager
from src.utils.exceptions import ValidationError
from src.utils.helpers import utc_now
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExportService:
    """Generates downloadable exports of a chat conversation."""

    def __init__(self, db: SQLiteManager | None = None) -> None:
        self.db = db or SQLiteManager()

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------
    def export_to_markdown(self, session_id: str) -> str:
        """
        Render a conversation as a Markdown document, including
        citations as a sub-list under each assistant message.

        Raises:
            ValidationError: If the session has no messages to export.
        """
        session, messages = self._load_conversation(session_id)

        lines: list[str] = [
            f"# {session.title}",
            "",
            f"_Exported from DocIntel AI on {self._formatted_timestamp()}_",
            "",
            "---",
            "",
        ]

        for message in messages:
            speaker = self._speaker_label(message)
            lines.append(f"**{speaker}:**")
            lines.append("")
            lines.append(message.content)
            lines.append("")

            if message.citations:
                lines.append("*Sources:*")
                for citation in message.citations:
                    lines.append(f"- {self._citation_label(citation)}")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    # ------------------------------------------------------------------
    # Plain-text export (copy to clipboard)
    # ------------------------------------------------------------------
    def export_to_text(self, session_id: str) -> str:
        """
        Render a conversation as plain text — no Markdown syntax — for
        the "Copy Conversation" action.

        Raises:
            ValidationError: If the session has no messages to export.
        """
        session, messages = self._load_conversation(session_id)

        lines: list[str] = [session.title, "=" * len(session.title), ""]

        for message in messages:
            speaker = self._speaker_label(message)
            lines.append(f"{speaker}:")
            lines.append(message.content)
            if message.citations:
                lines.append("Sources:")
                for citation in message.citations:
                    lines.append(f"  - {self._citation_label(citation)}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    # ------------------------------------------------------------------
    # PDF export
    # ------------------------------------------------------------------
    def export_to_pdf(self, session_id: str) -> bytes:
        """
        Render a conversation as a PDF document.

        Raises:
            ValidationError: If the session has no messages to export.
        """
        session, messages = self._load_conversation(session_id)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=LETTER,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            title=session.title,
        )

        styles = getSampleStyleSheet()
        speaker_style = ParagraphStyle(
            "SpeakerLabel", parent=styles["Heading4"], spaceBefore=10, spaceAfter=4,
        )
        citation_style = ParagraphStyle(
            "Citation", parent=styles["Normal"], fontSize=9, textColor="#555555", leftIndent=12,
        )

        story = [
            Paragraph(html.escape(session.title), styles["Title"]),
            Paragraph(
                f"Exported from DocIntel AI on {html.escape(self._formatted_timestamp())}",
                styles["Italic"],
            ),
            Spacer(1, 16),
        ]

        for message in messages:
            speaker = self._speaker_label(message)
            story.append(Paragraph(html.escape(speaker), speaker_style))
            story.append(Paragraph(self._to_pdf_safe_html(message.content), styles["Normal"]))

            if message.citations:
                story.append(Spacer(1, 4))
                for citation in message.citations:
                    story.append(
                        Paragraph(html.escape(self._citation_label(citation)), citation_style)
                    )

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        logger.info(
            "Exported conversation %s to PDF (%d messages, %d bytes)",
            session_id, len(messages), len(pdf_bytes),
        )
        return pdf_bytes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_conversation(self, session_id: str) -> tuple[ChatSession, list[ChatMessage]]:
        session = self.db.get_chat_session(session_id)
        messages = self.db.get_messages_for_session(session_id)
        if not messages:
            raise ValidationError("This conversation has no messages to export yet.")
        return session, messages

    @staticmethod
    def _speaker_label(message: ChatMessage) -> str:
        return "You" if message.role == MessageRole.USER else "DocIntel AI"

    @staticmethod
    def _citation_label(citation) -> str:
        page = f", page {citation.page_number}" if citation.page_number else ""
        return f"{citation.filename}{page} ({citation.similarity_score:.0%} match)"

    @staticmethod
    def _formatted_timestamp() -> str:
        return utc_now().strftime("%Y-%m-%d %H:%M UTC")

    @staticmethod
    def _to_pdf_safe_html(content: str) -> str:
        """Escape user content for reportlab's Paragraph XML dialect, preserving line breaks."""
        return html.escape(content).replace("\n", "<br/>")
