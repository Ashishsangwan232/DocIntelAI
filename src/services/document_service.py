"""
document_service.py
====================
Orchestrates the full document ingestion pipeline:

    validate -> deduplicate -> save to disk -> extract text ->
    clean -> chunk -> persist metadata + chunks

This is the only place in the application that coordinates loaders,
`TextCleaner`, and `TextSplitter` together. The Streamlit upload page
(Phase 7) will call `DocumentService.process_upload()` once per file
and never touch loaders or the splitter directly — keeping business
logic completely out of the UI layer.

Embedding generation and ChromaDB storage are performed as the final
step of this pipeline (via injected `EmbeddingService` /
`ChromaManager` dependencies). A document only transitions to
`DocumentStatus.READY` once its chunks are embedded and stored in the
vector store; if embedding fails, the document is marked `FAILED` and
its (embedding-less) chunks are removed from SQLite so a failed
document never has orphaned chunks that look queryable but aren't.
"""

from __future__ import annotations

from pathlib import Path

from config import settings
from src.database.models import Document, DocumentStatus, FileType
from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.loaders.loader_factory import get_loader, supported_extensions
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.splitter import TextSplitter
from src.vectorstore.chroma_manager import ChromaManager
from src.utils.exceptions import (
    CorruptedFileError,
    DocumentProcessingError,
    DuplicateDocumentError,
    EmbeddingError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    ValidationError,
    VectorStoreError,
)
from src.utils.helpers import compute_file_hash, format_bytes, generate_uuid, get_file_extension, sanitize_filename, truncate_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentService:
    """Coordinates document upload, validation, extraction, and chunking."""

    def __init__(
        self,
        db: SQLiteManager | None = None,
        cleaner: TextCleaner | None = None,
        splitter: TextSplitter | None = None,
        embedding_service: EmbeddingService | None = None,
        chroma_manager: ChromaManager | None = None,
        upload_dir: Path | None = None,
    ) -> None:
        # Dependency injection: every collaborator can be swapped or
        # mocked in tests without modifying this class.
        self.db = db or SQLiteManager()
        self.cleaner = cleaner or TextCleaner()
        self.splitter = splitter or TextSplitter()
        self.embedding_service = embedding_service or EmbeddingService()
        self.chroma_manager = chroma_manager or ChromaManager()
        self.upload_dir = upload_dir or settings.paths.upload_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_upload(
        self,
        filename: str,
        file_bytes: bytes,
        collection_id: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Document:
        """
        Run the full ingestion pipeline for a single uploaded file.

        Args:
            filename: Original filename as provided by the uploader.
            file_bytes: Raw file content.
            collection_id: Optional collection to attach the document to.
            chunk_size: Overrides the configured chunk size for this
                upload only (e.g. a user preference from the Settings
                panel). Does not affect already-processed documents.
            chunk_overlap: Overrides the configured chunk overlap for
                this upload only.

        Returns:
            The persisted `Document` record (status READY; chunks and
            embeddings already written to SQLite and ChromaDB).

        Raises:
            UnsupportedFileTypeError: Extension not in the allow-list.
            FileTooLargeError: File exceeds `settings.upload.max_file_size_mb`.
            DuplicateDocumentError: A document with identical content already exists.
            DocumentProcessingError: Extraction/cleaning/splitting failed.
        """
        safe_filename = sanitize_filename(filename)
        extension = self._validate(safe_filename, file_bytes)

        file_hash = compute_file_hash(file_bytes)
        existing = self.db.get_document_by_hash(file_hash)
        if existing is not None:
            raise DuplicateDocumentError(
                f"'{filename}' matches an already-uploaded document: "
                f"'{existing.filename}' (id={existing.id})."
            )

        document = Document(
            filename=safe_filename,
            file_type=FileType(extension),
            file_size_bytes=len(file_bytes),
            file_hash=file_hash,
            collection_id=collection_id,
        )
        self.db.create_document(document)

        stored_path = self._save_to_disk(document.id, safe_filename, file_bytes)

        try:
            loader = get_loader(extension)
            try:
                loaded_document = loader.load(stored_path)
            except CorruptedFileError:
                # Loaders report errors using the on-disk filename
                # (prefixed with the document's UUID for storage
                # uniqueness); re-raise using the name the user
                # actually recognizes.
                raise CorruptedFileError(
                    f"'{filename}' could not be read — it may be corrupted, "
                    f"password-protected, or not a valid file of its type."
                ) from None

            if loaded_document.is_empty():
                raise DocumentProcessingError(
                    f"'{filename}' contains no extractable text."
                )

            cleaned_pages = [
                page for page in loaded_document.pages if page.text.strip()
            ]
            for page in cleaned_pages:
                page.text = self.cleaner.clean(page.text)
            loaded_document.pages = [p for p in cleaned_pages if p.text.strip()]

            if loaded_document.is_empty():
                raise DocumentProcessingError(
                    f"'{filename}' contained no usable text after cleaning."
                )

            chunk_splitter = self.splitter
            if chunk_size is not None or chunk_overlap is not None:
                # A one-off splitter for this upload only — cheap to
                # construct (no heavy resources), and lets per-upload
                # chunking preferences from the Settings panel take
                # effect without reconstructing the cached DocumentService.
                chunk_splitter = TextSplitter(
                    chunk_size=chunk_size or self.splitter.chunk_size,
                    chunk_overlap=chunk_overlap or self.splitter.chunk_overlap,
                )

            chunks = chunk_splitter.split_document(loaded_document, document.id)
            if not chunks:
                raise DocumentProcessingError(
                    f"'{filename}' produced zero chunks during splitting."
                )

            self.db.bulk_create_chunks(chunks)

            try:
                embeddings = self.embedding_service.embed_texts(
                    [c.content for c in chunks]
                )
                self.chroma_manager.add_chunks(document, chunks, embeddings)
            except (EmbeddingError, VectorStoreError) as exc:
                # Embedding/storage failed after chunks were already
                # written to SQLite — remove them so a FAILED document
                # never has orphaned, unembedded chunks that look
                # queryable in the metadata store but aren't searchable.
                self.db.delete_chunks_for_document(document.id)
                raise DocumentProcessingError(
                    f"'{filename}' was chunked successfully but embedding failed: {exc}"
                ) from exc

            self.db.update_document_status(
                document.id,
                DocumentStatus.READY,
                page_count=loaded_document.page_count,
            )
            document.status = DocumentStatus.READY
            document.page_count = loaded_document.page_count

            logger.info(
                "Processed '%s' -> %d chunks embedded and stored (%s)",
                safe_filename, len(chunks), format_bytes(len(file_bytes)),
            )
            return document

        except DocumentProcessingError:
            self._fail_document(document.id, stored_path)
            raise
        except Exception as exc:
            logger.error("Unexpected failure processing '%s': %s", filename, exc, exc_info=True)
            self._fail_document(document.id, stored_path)
            raise DocumentProcessingError(
                f"Unexpected error while processing '{filename}': {exc}"
            ) from exc

    def delete_document(self, document_id: str) -> None:
        """Delete a document's vectors, chunks, metadata record, and file on disk."""
        document = self.db.get_document(document_id)
        stored_path = self._resolve_stored_path(document_id, document.filename)
        self.chroma_manager.delete_by_document(document_id)
        self.db.delete_document(document_id)  # cascades to chunks
        if stored_path.exists():
            stored_path.unlink()
        logger.info("Deleted document %s (vectors, chunks, metadata, file)", document_id)

    def get_document(self, document_id: str) -> Document:
        return self.db.get_document(document_id)

    def list_documents(
        self,
        collection_id: str | None = None,
        status: DocumentStatus | None = None,
    ) -> list[Document]:
        return self.db.list_documents(collection_id=collection_id, status=status)

    def get_chunk_count(self, document_id: str) -> int:
        """Number of chunks stored for a document (used by the library UI)."""
        return self.db.count_chunks_for_document(document_id)

    def get_preview_text(self, document_id: str, max_chars: int = 2000) -> str:
        """
        Return a preview of a document's extracted text, built from its
        stored chunks in order. Used by the document library's
        "Preview" action rather than re-reading the original file.
        """
        chunks = self.db.get_chunks_for_document(document_id)
        if not chunks:
            return "(No preview available — this document has no processed content.)"

        preview = ""
        for chunk in chunks:
            if len(preview) >= max_chars:
                break
            preview += (chunk.content + " ")
        return truncate_text(preview.strip(), max_length=max_chars)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate(self, filename: str, file_bytes: bytes) -> str:
        extension = get_file_extension(filename)
        if extension not in settings.upload.allowed_extensions:
            raise UnsupportedFileTypeError(
                f"'.{extension}' is not supported. Allowed types: "
                f"{', '.join(settings.upload.allowed_extensions)}."
            )
        if extension not in supported_extensions():
            # Defensive check: config allow-list and loader registry
            # must stay in sync; this catches drift early.
            raise UnsupportedFileTypeError(
                f"No loader registered for '.{extension}'."
            )
        if len(file_bytes) == 0:
            raise ValidationError(f"'{filename}' is empty.")
        if len(file_bytes) > settings.upload.max_file_size_bytes:
            raise FileTooLargeError(
                f"'{filename}' ({format_bytes(len(file_bytes))}) exceeds the "
                f"{settings.upload.max_file_size_mb}MB upload limit."
            )
        return extension

    def _save_to_disk(self, document_id: str, filename: str, file_bytes: bytes) -> Path:
        stored_path = self._resolve_stored_path(document_id, filename)
        stored_path.write_bytes(file_bytes)
        return stored_path

    def _resolve_stored_path(self, document_id: str, filename: str) -> Path:
        # Prefixing with the document ID guarantees uniqueness on disk
        # even if two documents share a sanitized filename.
        return self.upload_dir / f"{document_id}_{filename}"

    def _fail_document(self, document_id: str, stored_path: Path) -> None:
        try:
            self.db.update_document_status(document_id, DocumentStatus.FAILED)
        finally:
            # Keep the failed file on disk for now — useful for manual
            # debugging of extraction failures; cleanup happens on
            # explicit delete_document() calls, not silently here.
            pass
