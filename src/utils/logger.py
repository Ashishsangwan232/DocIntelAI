"""
logger.py
=========
Structured logging factory for DocIntel AI.

Every module obtains its logger via `get_logger(__name__)` rather than
calling `logging.getLogger` directly. This guarantees a consistent
format, consistent handlers (console + rotating file), and a single
place to change log behavior (e.g. switching to JSON logs for a
production log aggregator) without touching call sites.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import settings

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Guards against re-adding handlers if get_logger is called many times
# for the same logger name (e.g. across Streamlit re-runs).
_CONFIGURED_LOGGERS: set[str] = set()


def _build_file_handler(log_dir: Path) -> RotatingFileHandler:
    """Create a rotating file handler capped at 5MB x 5 backups."""
    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / "docintel.log"
    handler = RotatingFileHandler(
        filename=str(file_path),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _build_console_handler() -> logging.StreamHandler:
    """Create a console handler that writes to stdout."""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Args:
        name: Typically `__name__` of the calling module.

    Returns:
        A `logging.Logger` instance with console + rotating-file handlers
        attached exactly once, at the level configured in `settings.app.log_level`.
    """
    logger = logging.getLogger(name)

    if name not in _CONFIGURED_LOGGERS:
        level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
        logger.setLevel(level)
        logger.addHandler(_build_console_handler())
        logger.addHandler(_build_file_handler(settings.paths.logs_dir))
        # Prevent double-logging via the root logger.
        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)

    return logger
