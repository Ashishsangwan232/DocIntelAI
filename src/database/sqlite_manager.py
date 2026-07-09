"""
sqlite_manager.py
=================
Repository pattern implementation over SQLite.

`SQLiteManager` is the ONLY module in the codebase allowed to write raw
SQL. Every other layer (services, UI) interacts with the database
exclusively through this class's typed methods, which accept and
return the dataclasses defined in `models.py`.

This isolation is what makes a future migration to PostgreSQL a
single-file change: a `PostgresManager` implementing the same method
signatures could be swapped in behind a shared `BaseRepository`
interface without touching `services/*.py`.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config import settings
from src.database.models import (
    ChatMessage,
    ChatSession,
    Chunk,
    Citation,
    Collection,
    Document,
    DocumentStatus,
    FileType,
    MessageRole,
    QueryLog,
)
from src.utils.exceptions import DatabaseError, RecordNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    collection_id   TEXT REFERENCES collections(id) ON DELETE SET NULL,
    filename        TEXT NOT NULL,
    file_type       TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_hash       TEXT UNIQUE NOT NULL,
    page_count      INTEGER,
    status          TEXT NOT NULL DEFAULT 'processing',
    uploaded_at     TEXT NOT NULL,
    processed_at    TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    page_number     INTEGER,
    content         TEXT NOT NULL,
    token_count     INTEGER,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id              TEXT PRIMARY KEY,
    collection_id   TEXT REFERENCES collections(id) ON DELETE SET NULL,
    title           TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id                TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role              TEXT NOT NULL,
    content           TEXT NOT NULL,
    citations_json    TEXT,
    response_time_ms  INTEGER,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_logs (
    id            TEXT PRIMARY KEY,
    document_id   TEXT REFERENCES documents(id) ON DELETE SET NULL,
    query_text    TEXT NOT NULL,
    top_score     REAL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_document ON query_logs(document_id);
"""


class SQLiteManager:
    """
    Repository providing typed CRUD access to the SQLite metadata store.

    A new connection is opened per operation via the `_connection`
    context manager. For an app of this scale (single-process
    Streamlit) this avoids cross-thread connection issues without
    needing a full connection pool, while keeping the door open to
    swap in pooling later without changing any call sites.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.database.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------
    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            logger.error("SQLite operation failed: %s", exc, exc_info=True)
            raise DatabaseError(f"Database operation failed: {exc}") from exc
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(_SCHEMA)
        logger.info("SQLite schema initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------
    def create_collection(self, collection: Collection) -> Collection:
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO collections (id, name, description, created_at)
                   VALUES (?, ?, ?, ?)""",
                (collection.id, collection.name, collection.description,
                 collection.created_at.isoformat()),
            )
        logger.info("Created collection '%s' (%s)", collection.name, collection.id)
        return collection

    def get_collection(self, collection_id: str) -> Collection:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM collections WHERE id = ?", (collection_id,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Collection {collection_id} not found")
        return _row_to_collection(row)

    def list_collections(self) -> list[Collection]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM collections ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_collection(r) for r in rows]

    def delete_collection(self, collection_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
        logger.info("Deleted collection %s", collection_id)

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------
    def create_document(self, document: Document) -> Document:
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO documents
                   (id, collection_id, filename, file_type, file_size_bytes,
                    file_hash, page_count, status, uploaded_at, processed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document.id,
                    document.collection_id,
                    document.filename,
                    document.file_type.value,
                    document.file_size_bytes,
                    document.file_hash,
                    document.page_count,
                    document.status.value,
                    document.uploaded_at.isoformat(),
                    document.processed_at.isoformat() if document.processed_at else None,
                ),
            )
        logger.info("Created document '%s' (%s)", document.filename, document.id)
        return document

    def get_document(self, document_id: str) -> Document:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Document {document_id} not found")
        return _row_to_document(row)

    def get_document_by_hash(self, file_hash: str) -> Document | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(
        self,
        collection_id: str | None = None,
        status: DocumentStatus | None = None,
    ) -> list[Document]:
        query = "SELECT * FROM documents WHERE 1=1"
        params: list[str] = []
        if collection_id is not None:
            query += " AND collection_id = ?"
            params.append(collection_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY uploaded_at DESC"

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_document(r) for r in rows]

    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        page_count: int | None = None,
        processed_at_iso: str | None = None,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """UPDATE documents
                   SET status = ?, page_count = COALESCE(?, page_count),
                       processed_at = COALESCE(?, processed_at)
                   WHERE id = ?""",
                (status.value, page_count, processed_at_iso, document_id),
            )
        logger.info("Updated document %s status -> %s", document_id, status.value)

    def delete_document(self, document_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        logger.info("Deleted document %s", document_id)

    def count_documents(self) -> int:
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
        return int(row["c"])

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------
    def bulk_create_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._connection() as conn:
            conn.executemany(
                """INSERT INTO chunks
                   (id, document_id, chunk_index, page_number, content,
                    token_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        c.id, c.document_id, c.chunk_index, c.page_number,
                        c.content, c.token_count, c.created_at.isoformat(),
                    )
                    for c in chunks
                ],
            )
        logger.info("Inserted %d chunks for document %s", len(chunks), chunks[0].document_id)

    def get_chunks_for_document(self, document_id: str) -> list[Chunk]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM chunks WHERE document_id = ?
                   ORDER BY chunk_index ASC""",
                (document_id,),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count_chunks_for_document(self, document_id: str) -> int:
        """Lightweight count query — avoids loading full chunk content just to count."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchone()
        return int(row["c"])

    def count_chunks(self) -> int:
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()
        return int(row["c"])

    def get_average_response_time_ms(self) -> float | None:
        """Average assistant response time across all chat messages, or None if no data yet."""
        with self._connection() as conn:
            row = conn.execute(
                """SELECT AVG(response_time_ms) AS avg_ms FROM chat_messages
                   WHERE role = 'assistant' AND response_time_ms IS NOT NULL"""
            ).fetchone()
        return float(row["avg_ms"]) if row["avg_ms"] is not None else None

    def get_total_storage_bytes(self) -> int:
        """Total size of all uploaded document files, in bytes."""
        with self._connection() as conn:
            row = conn.execute("SELECT COALESCE(SUM(file_size_bytes), 0) AS total FROM documents").fetchone()
        return int(row["total"])

    def count_chat_sessions(self) -> int:
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM chat_sessions").fetchone()
        return int(row["c"])

    def count_total_queries(self) -> int:
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM query_logs").fetchone()
        return int(row["c"])

    def get_documents_uploaded_per_day(self, days: int = 14) -> list[tuple[str, int]]:
        """Return (date, count) pairs for documents uploaded in the last N days, oldest first."""
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT DATE(uploaded_at) AS day, COUNT(*) AS c
                   FROM documents
                   WHERE uploaded_at >= DATE('now', ?)
                   GROUP BY day
                   ORDER BY day ASC""",
                (f"-{days} days",),
            ).fetchall()
        return [(r["day"], r["c"]) for r in rows]

    def delete_chunks_for_document(self, document_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    # ------------------------------------------------------------------
    # Chat sessions & messages
    # ------------------------------------------------------------------
    def create_chat_session(self, session: ChatSession) -> ChatSession:
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO chat_sessions (id, collection_id, title, created_at)
                   VALUES (?, ?, ?, ?)""",
                (session.id, session.collection_id, session.title,
                 session.created_at.isoformat()),
            )
        return session

    def list_chat_sessions(self) -> list[ChatSession]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_sessions ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_chat_session(r) for r in rows]

    def get_chat_session(self, session_id: str) -> ChatSession:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Chat session {session_id} not found")
        return _row_to_chat_session(row)

    def add_chat_message(self, message: ChatMessage) -> ChatMessage:
        citations_json = json.dumps(
            [
                {
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "similarity_score": c.similarity_score,
                    "excerpt": c.excerpt,
                    "page_number": c.page_number,
                }
                for c in message.citations
            ]
        )
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO chat_messages
                   (id, session_id, role, content, citations_json,
                    response_time_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    message.id, message.session_id, message.role.value,
                    message.content, citations_json, message.response_time_ms,
                    message.created_at.isoformat(),
                ),
            )
        return message

    def get_messages_for_session(self, session_id: str) -> list[ChatMessage]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM chat_messages WHERE session_id = ?
                   ORDER BY created_at ASC""",
                (session_id,),
            ).fetchall()
        return [_row_to_chat_message(r) for r in rows]

    def delete_chat_message(self, message_id: str) -> None:
        """Delete a single chat message (used by the 'regenerate' feature)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM chat_messages WHERE id = ?", (message_id,))

    def clear_session_messages(self, session_id: str) -> None:
        """Delete every message in a session but keep the session itself."""
        with self._connection() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        logger.info("Cleared all messages for chat session %s", session_id)

    def delete_chat_session(self, session_id: str) -> None:
        """Delete a chat session and cascade-delete all its messages."""
        with self._connection() as conn:
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        logger.info("Deleted chat session %s", session_id)

    # ------------------------------------------------------------------
    # Query logs (analytics)
    # ------------------------------------------------------------------
    def log_query(self, query_log: QueryLog) -> None:
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO query_logs (id, document_id, query_text, top_score, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    query_log.id, query_log.document_id, query_log.query_text,
                    query_log.top_score, query_log.created_at.isoformat(),
                ),
            )

    def get_most_queried_documents(self, limit: int = 5) -> list[tuple[str, int]]:
        """Return (document_id, query_count) tuples, most-queried first."""
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT document_id, COUNT(*) AS query_count
                   FROM query_logs
                   WHERE document_id IS NOT NULL
                   GROUP BY document_id
                   ORDER BY query_count DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [(r["document_id"], r["query_count"]) for r in rows]

    # ------------------------------------------------------------------
    # Application settings (key-value store for user preferences)
    # ------------------------------------------------------------------
    def set_setting(self, key: str, value: str) -> None:
        from src.utils.helpers import utc_now

        with self._connection() as conn:
            conn.execute(
                """INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                   updated_at = excluded.updated_at""",
                (key, value, utc_now().isoformat()),
            )

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default


# ---------------------------------------------------------------------------
# Row -> dataclass mapping helpers (module-private)
# ---------------------------------------------------------------------------
def _row_to_collection(row: sqlite3.Row) -> Collection:
    return Collection(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        created_at=_parse_dt(row["created_at"]),
    )


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        collection_id=row["collection_id"],
        filename=row["filename"],
        file_type=FileType(row["file_type"]),
        file_size_bytes=row["file_size_bytes"],
        file_hash=row["file_hash"],
        page_count=row["page_count"],
        status=DocumentStatus(row["status"]),
        uploaded_at=_parse_dt(row["uploaded_at"]),
        processed_at=_parse_dt(row["processed_at"]) if row["processed_at"] else None,
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        page_number=row["page_number"],
        content=row["content"],
        token_count=row["token_count"],
        created_at=_parse_dt(row["created_at"]),
    )


def _row_to_chat_session(row: sqlite3.Row) -> ChatSession:
    return ChatSession(
        id=row["id"],
        collection_id=row["collection_id"],
        title=row["title"],
        created_at=_parse_dt(row["created_at"]),
    )


def _row_to_chat_message(row: sqlite3.Row) -> ChatMessage:
    citations_raw = json.loads(row["citations_json"]) if row["citations_json"] else []
    citations = [
        Citation(
            document_id=c["document_id"],
            filename=c["filename"],
            chunk_id=c["chunk_id"],
            chunk_index=c["chunk_index"],
            similarity_score=c["similarity_score"],
            excerpt=c["excerpt"],
            page_number=c.get("page_number"),
        )
        for c in citations_raw
    ]
    return ChatMessage(
        id=row["id"],
        session_id=row["session_id"],
        role=MessageRole(row["role"]),
        content=row["content"],
        citations=citations,
        response_time_ms=row["response_time_ms"],
        created_at=_parse_dt(row["created_at"]),
    )


def _parse_dt(value: str):
    from datetime import datetime

    return datetime.fromisoformat(value)
