"""
tests/test_security.py
=======================
Dedicated security regression tests, distinct from the functional
test suites elsewhere. These encode the guarantees from the Phase 13
security audit as executable tests, so future changes can't silently
regress them.

Covers:
- Path traversal via malicious filenames (unit + full pipeline)
- XSS-injection-style filenames neutralized before ever reaching
  citation rendering (which uses unsafe_allow_html=True)
- API keys/secrets never appear in log output or exception messages
- Upload size limit enforced before any parsing is attempted
- .env is never committed; .env.example contains no real secrets
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from src.database.sqlite_manager import SQLiteManager
from src.embeddings.embedding_service import EmbeddingService
from src.llm.ollama_cloud import OllamaCloudLLM
from src.services.document_service import DocumentService
from src.utils.exceptions import FileTooLargeError, LLMResponseError
from src.utils.helpers import sanitize_filename
from src.vectorstore.chroma_manager import ChromaManager

_PATH_TRAVERSAL_PAYLOADS = [
    "../../../etc/passwd",
    "..\\..\\windows\\system32\\config",
    "....//....//etc/passwd",
    "/etc/passwd",
    "C:\\Windows\\System32\\drivers\\etc\\hosts",
]

_XSS_PAYLOADS = [
    "<script>alert(1)</script>.pdf",
    '"><img src=x onerror=alert(1)>.txt',
    "<img src=x onerror=alert(document.cookie)>.md",
]


class FakeEmbeddingModel:
    def encode(self, sentences, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
        return np.array([[0.1] * 8 for _ in sentences])

    def get_sentence_embedding_dimension(self) -> int:
        return 8


@pytest.fixture()
def document_service(tmp_path: Path) -> DocumentService:
    db = SQLiteManager(db_path=tmp_path / "test.db")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return DocumentService(
        db=db,
        upload_dir=upload_dir,
        embedding_service=EmbeddingService(model=FakeEmbeddingModel()),
        chroma_manager=ChromaManager(persist_directory=tmp_path / "chroma", collection_name="test"),
    )


class TestPathTraversalDefense:
    @pytest.mark.parametrize("payload", _PATH_TRAVERSAL_PAYLOADS)
    def test_sanitize_filename_strips_path_separators(self, payload: str) -> None:
        result = sanitize_filename(payload)
        assert "/" not in result
        assert "\\" not in result

    def test_full_pipeline_cannot_escape_upload_directory(
        self, document_service: DocumentService, tmp_path: Path
    ) -> None:
        malicious_name = "../../../../tmp/evil_escaped_file.txt"
        document = document_service.process_upload(malicious_name, b"malicious content")

        stored_path = document_service.upload_dir / f"{document.id}_{document.filename}"
        upload_dir_resolved = document_service.upload_dir.resolve()

        assert str(stored_path.resolve()).startswith(str(upload_dir_resolved))
        # Confirm nothing was written to the traversal target itself.
        escape_target = Path("/tmp/evil_escaped_file.txt")
        assert not escape_target.exists()

    def test_deeply_nested_traversal_also_contained(
        self, document_service: DocumentService
    ) -> None:
        document = document_service.process_upload(
            "....//....//....//....//etc/shadow.txt", b"content"
        )
        stored_path = document_service.upload_dir / f"{document.id}_{document.filename}"
        assert stored_path.resolve().parent == document_service.upload_dir.resolve()


class TestXSSDefenseInFilenames:
    @pytest.mark.parametrize("payload", _XSS_PAYLOADS)
    def test_sanitize_filename_strips_html_special_characters(self, payload: str) -> None:
        result = sanitize_filename(payload)
        assert "<" not in result
        assert ">" not in result
        assert '"' not in result

    def test_xss_filename_survives_full_pipeline_neutralized(
        self, document_service: DocumentService
    ) -> None:
        """
        Citation rendering (ui/chat.py) displays `document.filename`
        inside an `unsafe_allow_html=True` markdown block (to embed the
        match-dial). This is only safe because filenames are sanitized
        at upload time — this test is the guarantee that holds that
        assumption true end-to-end, not just in the isolated helper.
        """
        document = document_service.process_upload(
            "<script>alert('xss')</script>.txt", b"content"
        )
        assert "<" not in document.filename
        assert ">" not in document.filename
        assert "script" in document.filename.lower()  # neutralized, not deleted


class TestUploadValidationOrdering:
    def test_oversized_file_rejected_before_parsing(
        self, document_service: DocumentService
    ) -> None:
        """
        A corrupted/adversarial file that's also oversized must be
        rejected on size alone, before any parser ever touches its
        bytes — validation order matters for resource-exhaustion
        defense, not just correctness.
        """
        oversized_content = b"x" * (26 * 1024 * 1024)
        with pytest.raises(FileTooLargeError):
            document_service.process_upload("huge.txt", oversized_content)
        assert document_service.db.count_documents() == 0


class TestSecretsNeverLeak:
    def test_api_key_not_in_exception_message_on_connection_failure(self) -> None:
        secret = "sk-supersecret-do-not-leak-1234567890"

        class FailingClient:
            def chat(self, *, model, messages, options, stream=False):
                raise ConnectionError("Failed to connect to Ollama Cloud")

        llm = OllamaCloudLLM(api_key=secret, client=FailingClient())
        with pytest.raises(LLMResponseError) as exc_info:
            llm.generate("test prompt")
        assert secret not in str(exc_info.value)

    def test_api_key_not_in_log_output_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        secret = "sk-supersecret-do-not-leak-1234567890"

        class FailingClient:
            def chat(self, *, model, messages, options, stream=False):
                raise ConnectionError("network unreachable")

        llm = OllamaCloudLLM(api_key=secret, client=FailingClient())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMResponseError):
                llm.generate("test prompt")

        assert secret not in caplog.text

    def test_api_key_not_in_exception_on_auth_rejection(self) -> None:
        secret = "sk-another-secret-value"

        class FakeResponseError(Exception):
            status_code = 401

        class FakeClient:
            def chat(self, *, model, messages, options, stream=False):
                raise FakeResponseError("Unauthorized")

        llm = OllamaCloudLLM(api_key=secret, client=FakeClient())
        with pytest.raises(Exception) as exc_info:
            llm.generate("test")
        assert secret not in str(exc_info.value)


class TestEnvFileHandling:
    def test_env_example_contains_no_real_looking_secrets(self) -> None:
        env_example_path = Path(__file__).resolve().parent.parent / ".env.example"
        content = env_example_path.read_text()
        # Placeholder should be present; nothing that looks like a
        # real, populated API key (long alphanumeric string) should be.
        assert "your_ollama_cloud_api_key_here" in content or "OLLAMA_CLOUD_API_KEY=" in content
        for line in content.splitlines():
            if line.strip().startswith("OLLAMA_CLOUD_API_KEY="):
                value = line.split("=", 1)[1].strip()
                assert value in ("", "your_ollama_cloud_api_key_here") or "your_" in value.lower()

    def test_gitignore_excludes_env_file(self) -> None:
        gitignore_path = Path(__file__).resolve().parent.parent / ".gitignore"
        content = gitignore_path.read_text()
        assert ".env" in content
