"""
loader_factory.py
=================
Maps a file extension to the correct `BaseLoader` implementation.

`document_service.py` calls `get_loader(extension)` instead of holding
an if/elif chain over file types — adding a new format later means
registering one new loader here, nothing else changes.
"""

from __future__ import annotations

from src.loaders.base_loader import BaseLoader
from src.loaders.docx_loader import DocxLoader
from src.loaders.pdf_loader import PDFLoader
from src.loaders.txt_loader import TextLoader
from src.utils.exceptions import UnsupportedFileTypeError

# Instantiated once — loaders are stateless, so reuse is safe and avoids
# re-allocating on every upload.
_LOADER_REGISTRY: dict[str, BaseLoader] = {}


def _register(loader: BaseLoader) -> None:
    for ext in loader.supported_extensions:
        _LOADER_REGISTRY[ext] = loader


_register(PDFLoader())
_register(DocxLoader())
_register(TextLoader())  # covers both "txt" and "md"


def get_loader(extension: str) -> BaseLoader:
    """
    Return the loader responsible for the given file extension.

    Args:
        extension: Lowercase extension without the dot (e.g. "pdf").

    Raises:
        UnsupportedFileTypeError: If no loader is registered for the extension.
    """
    loader = _LOADER_REGISTRY.get(extension.lower())
    if loader is None:
        supported = ", ".join(sorted(_LOADER_REGISTRY.keys()))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{extension}'. Supported types: {supported}."
        )
    return loader


def supported_extensions() -> list[str]:
    """Return all currently registered file extensions."""
    return sorted(_LOADER_REGISTRY.keys())
