"""``RootedStore``: one ``BlobStore`` over every named root, routed by the key's first segment."""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

from src.clients.storage.base import BlobStat, BlobStore, ReadOnlyRootError
from src.clients.storage.keys import make_key, parse_key
from src.clients.storage.local import LocalFsStore


if TYPE_CHECKING:
    from pathlib import Path

    from src.config.storage import StorageRootPaths

READ_ONLY_ROOTS = frozenset({'source'})


class RootedStore:
    """Keys carry their root (``projects/<slug>/...``); each root is its own ``LocalFsStore``."""

    def __init__(self, roots: dict[str, Path]) -> None:
        self._stores: dict[str, BlobStore] = {name: LocalFsStore(p) for name, p in roots.items()}

    def _route(self, key: str, *, write: bool = False) -> tuple[BlobStore, str]:
        parsed = parse_key(key)
        if write and parsed.root in READ_ONLY_ROOTS:
            raise ReadOnlyRootError(f'storage root {parsed.root!r} is read only')
        if not parsed.parts:
            raise ValueError(f'storage key names a root, not an object: {key!r}')
        return self._stores[parsed.root], '/'.join(parsed.parts)

    def open_read(self, key: str) -> BinaryIO:
        store, rel = self._route(key)
        return store.open_read(rel)

    def write_atomic(self, key: str, data: bytes) -> None:
        store, rel = self._route(key, write=True)
        store.write_atomic(rel, data)

    def exists(self, key: str) -> bool:
        store, rel = self._route(key)
        return store.exists(rel)

    def stat(self, key: str) -> BlobStat:
        store, rel = self._route(key)
        return store.stat(rel)

    def list_prefix(self, prefix: str) -> list[str]:
        prefix = prefix.removesuffix('/')
        store, rel = self._route(prefix)
        root = parse_key(prefix).root
        return [make_key(root, *k.split('/')) for k in store.list_prefix(rel)]

    def delete_prefix(self, prefix: str) -> int:
        """Remove everything under ``prefix`` (``projects/<slug>/`` takes every area of a project)."""
        prefix = prefix.removesuffix('/')
        store, rel = self._route(prefix, write=True)
        return store.delete_prefix(rel)

    def local_path(self, key: str) -> Path:
        store, rel = self._route(key)
        return store.local_path(rel)


def build_rooted_store(paths: StorageRootPaths) -> RootedStore:
    return RootedStore(
        {
            'source': paths.source,
            'projects': paths.projects,
            'models': paths.models,
            'cache': paths.cache,
            'upgrade': paths.upgrade,
        }
    )
