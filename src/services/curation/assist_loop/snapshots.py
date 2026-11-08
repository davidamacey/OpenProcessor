"""The loop's blobs behind the storage seam: pinned roots, child member lists, written ids.

A node row keeps only the key; the blob holds the sorted member ids (a root also each member's
lock-rule state and class-state token at the snapshot). Keys are
``projects/<slug>/assist_loop/<run_id>/<kind>/<path>.json``, built with ``make_key``. Blobs are
written whole with ``write_atomic``, so a reader sees a blob or none, and a missing or
unreadable one is an error, never an empty node.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from src.clients.storage.roots import build_rooted_store
from src.config.storage import storage_root_paths
from src.domain.storage_keys import make_key
from src.services.curation.assist_loop.partition import MemberState, RootRef


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from src.clients.storage.base import BlobStore


class SnapshotStore:
    def __init__(self, slug: str, run_id: uuid.UUID, *, store: BlobStore | None = None) -> None:
        self._slug = slug
        self._run = str(run_id)
        self._store = store or build_rooted_store(storage_root_paths())

    def key(self, kind: str, path: str) -> str:
        return make_key('projects', self._slug, 'assist_loop', self._run, kind, f'{path}.json')

    def _write(self, key: str, body: Any) -> str:
        self._store.write_atomic(key, json.dumps(body, default=str).encode('utf-8'))
        return key

    def _read(self, key: str) -> Any:
        try:
            with self._store.open_read(key) as handle:
                return json.loads(handle.read())
        except (OSError, ValueError) as exc:
            msg = f'blob {key} is missing or unreadable: {exc}'
            raise ValueError(msg) from exc

    async def write_root(self, root: RootRef) -> str:
        body = {
            'root': root.key,
            'cluster_id': root.cluster_id,
            'version': root.version,
            'members': [
                {
                    'id': m.crop_id,
                    'state': m.state,
                    'class_name': m.class_name,
                    'token': list(m.token),
                    'truth': m.truth,
                }
                for m in root.members
            ],
        }
        return await asyncio.to_thread(self._write, self.key('nodes', root.key), body)

    async def read_root(self, key: str) -> RootRef:
        raw = await asyncio.to_thread(self._read, key)
        try:
            return RootRef(
                key=raw['root'],
                cluster_id=int(raw['cluster_id']),
                version=raw['version'],
                members=[
                    MemberState(
                        m['id'], m['state'], m['class_name'], tuple(m['token']), bool(m['truth'])
                    )
                    for m in raw['members']
                ],
            )
        except (KeyError, TypeError, ValueError) as exc:
            msg = f'root blob {key} is malformed: {exc}'
            raise ValueError(msg) from exc

    async def write_ids(self, kind: str, path: str, ids: Sequence[str]) -> str:
        return await asyncio.to_thread(self._write, self.key(kind, path), sorted(ids))

    async def read_ids(self, key: str) -> list[str]:
        raw = await asyncio.to_thread(self._read, key)
        if not isinstance(raw, list):
            msg = f'blob {key} is not a list of ids'
            raise ValueError(msg)
        return sorted(str(i) for i in raw)


__all__ = ['SnapshotStore']
