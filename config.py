"""
config.py
=========
Centralized application configuration for DocIntel AI.

Every module in the codebase reads settings through the single
`settings` instance exported from this file. Nothing outside this
file is allowed to call `os.getenv` directly — this keeps
configuration auditable and makes the Streamlit -> FastAPI migration
a one-file change instead of a search-and-replace across the repo.

Configuration values are loaded from environment variables, which in
local development are populated from a `.env` file via python-dotenv.
In production (Render), the platform injects real environment
variables directly, so `.env` is never required nor committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file (no-op in production if the file doesn't exist)
load_dotenv()

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent


def _env_str(key: str, default: str) -> str:
    """Fetch a string environment variable with a fallback default."""
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    """Fetch an integer environment variable with a fallback default."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Fetch a float environment variable with a fallback default."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Fetch a boolean environment variable with a fallback default."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(key: str, default: list[str]) -> list[str]:
    """Fetch a comma-separated environment variable as a list of strings."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class PathSettings:
    """Filesystem locations used throughout the application."""

    base_dir: Path = BASE_DIR
    upload_dir: Path = BASE_DIR / "uploads"
    database_dir: Path = BASE_DIR / "database"
    vectorstore_dir: Path = BASE_DIR / "vectorstore"
    logs_dir: Path = BASE_DIR / "logs"
    assets_dir: Path = BASE_DIR / "assets"

    def ensure_exist(self) -> None:
        """Create all required runtime directories if they don't exist."""
        for directory in (
            self.upload_dir,
            self.database_dir,
            self.vectorstore_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatabaseSettings:
    """SQLite configuration."""

    db_filename: str = field(default_factory=lambda: _env_str("SQLITE_DB_NAME", "docintel.db"))

    @property
    def db_path(self) -> Path:
        return PathSettings().database_dir / self.db_filename


@dataclass(frozen=True)
class VectorStoreSettings:
    """ChromaDB configuration."""

    persist_directory: Path = field(
        default_factory=lambda: PathSettings().vectorstore_dir
    )
    collection_name: str = field(
        default_factory=lambda: _env_str("CHROMA_COLLECTION_NAME", "docintel_documents")
    )
    distance_metric: str = field(
        default_factory=lambda: _env_str("CHROMA_DISTANCE_METRIC", "cosine")
    )


@dataclass(frozen=True)
class EmbeddingSettings:
    """Sentence-Transformers embedding model configuration."""

    model_name: str = field(
        default_factory=lambda: _env_str("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    )
    embedding_dimension: int = field(
        default_factory=lambda: _env_int("EMBEDDING_DIMENSION", 768)
    )
    batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 32))
    device: str = field(default_factory=lambda: _env_str("EMBEDDING_DEVICE", "cpu"))
    normalize_embeddings: bool = field(
        default_factory=lambda: _env_bool("EMBEDDING_NORMALIZE", True)
    )


@dataclass(frozen=True)
class LLMSettings:
    """Ollama Cloud LLM configuration."""

    provider: str = field(default_factory=lambda: _env_str("LLM_PROVIDER", "ollama_cloud"))
    api_base_url: str = field(
        default_factory=lambda: _env_str("OLLAMA_CLOUD_BASE_URL", "https://ollama.com")
    )
    api_key: str = field(default_factory=lambda: _env_str("OLLAMA_CLOUD_API_KEY", ""))
    model_name: str = field(default_factory=lambda: _env_str("LLM_MODEL", "gpt-oss:120b-cloud"))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.3))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))
    request_timeout_seconds: int = field(
        default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 60)
    )
    stream: bool = field(default_factory=lambda: _env_bool("LLM_STREAM", True))


@dataclass(frozen=True)
class RAGSettings:
    """Chunking and retrieval configuration."""

    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))
    top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 5))
    similarity_score_threshold: float = field(
        default_factory=lambda: _env_float("SIMILARITY_SCORE_THRESHOLD", 0.2)
    )


@dataclass(frozen=True)
class UploadSettings:
    """File upload validation rules."""

    max_file_size_mb: int = field(default_factory=lambda: _env_int("MAX_FILE_SIZE_MB", 25))
    allowed_extensions: list[str] = field(
        default_factory=lambda: _env_list("ALLOWED_EXTENSIONS", ["pdf", "docx", "txt", "md"])
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@dataclass(frozen=True)
class AppSettings:
    """Top-level application metadata and feature flags."""

    app_name: str = field(default_factory=lambda: _env_str("APP_NAME", "DocIntel AI"))
    app_env: str = field(default_factory=lambda: _env_str("APP_ENV", "development"))
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))


@dataclass(frozen=True)
class Settings:
    """Aggregate settings object — the single import surface for config."""

    app: AppSettings = field(default_factory=AppSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    vectorstore: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)
    upload: UploadSettings = field(default_factory=UploadSettings)

    def validate(self) -> list[str]:
        """
        Validate critical settings and return a list of human-readable
        warnings/errors. Does not raise — callers decide how strict to be
        (e.g. UI shows a banner, tests assert an empty list).
        """
        issues: list[str] = []
        if not self.llm.api_key:
            issues.append(
                "OLLAMA_CLOUD_API_KEY is not set. LLM calls will fail until it is configured."
            )
        if self.rag.chunk_overlap >= self.rag.chunk_size:
            issues.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.upload.max_file_size_mb <= 0:
            issues.append("MAX_FILE_SIZE_MB must be a positive integer.")
        return issues


# Single shared instance imported across the entire application.
settings = Settings()
settings.paths.ensure_exist()
