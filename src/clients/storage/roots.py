"""``RootedStore``: one ``BlobStore`` over every named root, routed by the key's first segment."""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

from src.clients.storage.base import BlobStat, BlobStore, ReadOnlyRootError
from src.clients.storage.local import LocalFsStore
from src.domain.storage_keys import make_key, parse_key


if TYPE_CHECKING:
    from pathlib import Path

    from src.config.storage import StorageRootPaths

READ_ONLY_ROOTS = frozenset({'source'})
# A project's bake-off results stay where 0.6.0 put them: ``<slug>/bakeoff_jobs/out/<job_id>``.
BAKEOFF_RESULTS_LAYOUT = ('bakeoff_jobs', 'out')
# Uploaded weights live on their own mount, not under the Triton model repository.
PYTORCH_MODELS_PART = 'pytorch_models'


class RootedStore:
    """Keys carry their root (``projects/<slug>/...``); each root is its own ``LocalFsStore``."""

    def __init__(self, roots: dict[str, Path]) -> None:
        self._stores: dict[str, BlobStore] = {
            name: LocalFsStore(p) for name, p in roots.items() if name != PYTORCH_MODELS_PART
        }
        self._pytorch_models: BlobStore | None = (
            LocalFsStore(roots[PYTORCH_MODELS_PART]) if PYTORCH_MODELS_PART in roots else None
        )

    def _route(self, key: str, *, write: bool = False) -> tuple[BlobStore, str]:
        parsed = parse_key(key)
        if write and parsed.root in READ_ONLY_ROOTS:
            raise ReadOnlyRootError(f'storage root {parsed.root!r} is read only')
        if not parsed.parts:
            raise ValueError(f'storage key names a root, not an object: {key!r}')
        parts = parsed.parts
        if parsed.root == 'bakeoff_results' and parts:
            parts = (parts[0], *BAKEOFF_RESULTS_LAYOUT, *parts[1:])
        if (
            parsed.root == 'models'
            and self._pytorch_models is not None
            and parts[:1] == (PYTORCH_MODELS_PART,)
            and len(parts) > 1
        ):
            return self._pytorch_models, '/'.join(parts[1:])
        return self._stores[parsed.root], '/'.join(parts)

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
        found = [k.split('/') for k in store.list_prefix(rel)]
        if root == 'bakeoff_results':
            n = len(BAKEOFF_RESULTS_LAYOUT)
            found = [[k[0], *k[n + 1 :]] for k in found]
        elif root == 'models' and store is self._pytorch_models:
            found = [[PYTORCH_MODELS_PART, *k] for k in found]
        return [make_key(root, *k) for k in found]

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
            'bakeoff_results': paths.bakeoff_results,
            PYTORCH_MODELS_PART: paths.pytorch_models,
        }
    )
