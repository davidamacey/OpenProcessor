"""The ``BlobStore`` protocol: what the rest of the code may ask of blob storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO, NamedTuple, Protocol, runtime_checkable


if TYPE_CHECKING:
    from pathlib import Path


class BlobStat(NamedTuple):
    size: int
    mtime: float


class ReadOnlyRootError(PermissionError):
    """A write or delete reached a root that is read only (the ingest ``source`` root)."""


@runtime_checkable
class BlobStore(Protocol):
    """Blobs addressed by keys from :mod:`src.domain.storage_keys`.

    ``list_prefix`` and ``delete_prefix`` match whole path segments: ``a/b`` covers ``a/b/c``
    and not ``a/bc``. ``local_path`` is for code that needs a real file (Ultralytics, Triton
    loading, ``faiss.write_index``); a remote backend materialises the object first.
    """

    def open_read(self, key: str) -> BinaryIO: ...

    def write_atomic(self, key: str, data: bytes) -> None: ...

    def exists(self, key: str) -> bool: ...

    def stat(self, key: str) -> BlobStat: ...

    def list_prefix(self, prefix: str) -> list[str]: ...

    def delete_prefix(self, prefix: str) -> int: ...

    def local_path(self, key: str) -> Path: ...
