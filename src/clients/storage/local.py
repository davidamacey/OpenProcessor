"""``LocalFsStore``: a ``BlobStore`` over one directory, writes through ``atomic_write``."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO

from src.clients.storage.base import BlobStat
from src.utils.atomic_write import atomic_write_bytes


def _segments(key: str) -> list[str]:
    parts = key.split('/')
    for part in parts:
        if not part or part in {'.', '..'} or '\\' in part or '\x00' in part:
            raise ValueError(f'unsafe storage key: {key!r}')
    return parts


class LocalFsStore:
    """Keys are relative to ``root_path`` (the root name is not part of them here)."""

    def __init__(self, root_path: Path) -> None:
        self._root = Path(root_path)

    def local_path(self, key: str) -> Path:
        return self._root.joinpath(*_segments(key))

    def open_read(self, key: str) -> BinaryIO:
        return self.local_path(key).open('rb')

    def write_atomic(self, key: str, data: bytes) -> None:
        atomic_write_bytes(self.local_path(key), data)

    def exists(self, key: str) -> bool:
        return self.local_path(key).is_file()

    def stat(self, key: str) -> BlobStat:
        st = self.local_path(key).stat()
        return BlobStat(size=st.st_size, mtime=st.st_mtime)

    def list_prefix(self, prefix: str) -> list[str]:
        base = self.local_path(prefix)
        if base.is_file():
            return [prefix]
        if not base.is_dir():
            return []
        return sorted(
            f'{prefix}/{p.relative_to(base).as_posix()}'
            for p in base.rglob('*')
            if p.is_file() and not (p.name.startswith('.') and p.name.endswith('.tmp'))
        )

    def delete_prefix(self, prefix: str) -> int:
        """Remove every file under ``prefix`` (any area, listed anywhere or not); return the count."""
        base = self.local_path(prefix)
        if base.is_file():
            base.unlink()
            return 1
        if not base.is_dir():
            return 0
        count = sum(1 for p in base.rglob('*') if p.is_file())
        shutil.rmtree(base)
        return count
