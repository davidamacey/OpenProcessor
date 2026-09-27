"""Fixtures shared by tests/projects/*.

Deliberately does NOT reuse tests/conftest.py's autouse default-project
bind (that fixture does not exist yet -- it lands with commit 4's route
mounting); every test here binds explicitly.
"""

from __future__ import annotations

from typing import Any

import pytest


class FakeRegistryOpenSearch:
    """Just enough of AsyncOpenSearch for ``ProjectRegistry``/
    ``bootstrap_default_project``: ``get``/``index``/``search`` on one
    flat doc store, keyed by id."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    async def get(self, *, index: str, id: str) -> dict[str, Any]:  # noqa: A002, ARG002
        if id not in self.docs:
            return {'found': False, '_id': id}
        return {'found': True, '_id': id, '_source': self.docs[id]}

    async def index(self, *, index: str, id: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: A002, ARG002
        self.docs[id] = body
        return {'_id': id, 'result': 'created'}

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        prefix = (((body.get('query') or {}).get('prefix') or {}).get('_id')) or ''
        hits = [
            {'_id': doc_id, '_source': doc}
            for doc_id, doc in self.docs.items()
            if doc_id.startswith(prefix)
        ]
        return {'hits': {'hits': hits}}


@pytest.fixture
def fake_registry_client() -> FakeRegistryOpenSearch:
    return FakeRegistryOpenSearch()


class _FakeIndices:
    def __init__(self, outer: FakeLifecycleOpenSearch) -> None:
        self._outer = outer

    async def delete(self, *, index: str, ignore: Any = None) -> dict[str, Any]:  # noqa: ARG002
        self._outer.deleted_indexes.append(index)
        self._outer.indexes.pop(index, None)
        return {'acknowledged': True}

    async def create(self, *, index: str, body: Any = None) -> dict[str, Any]:  # noqa: ARG002
        self._outer.indexes[index] = self._outer.indexes.get(index, [])
        return {'acknowledged': True}


class _FakeTransport:
    """Faked ``/_cluster/health`` etc. so :func:`capacity_status` always
    reports ``ok`` (plenty of headroom) unless a test overrides it."""

    async def perform_request(
        self,
        method: str,  # noqa: ARG002
        url: str,
        params: Any = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Any:
        if url == '/_cluster/health':
            return {'active_shards': 5, 'number_of_data_nodes': 1}
        if url == '/_cluster/settings':
            return {'persistent': {'cluster.max_shards_per_node': 1000}, 'transient': {}}
        if url == '/_nodes/stats/jvm':
            return {'nodes': {'n1': {'jvm': {'mem': {'heap_max_in_bytes': 8 * 1024**3}}}}}
        if url.startswith('/_cat/indices'):
            return []
        raise NotImplementedError(url)


class VersionConflictError(Exception):
    """Shaped enough like opensearchpy's ``ConflictError`` for
    ``write_record``'s OCC callers -- they never inspect the type, only
    that *something* raised."""


class FakeLifecycleOpenSearch(FakeRegistryOpenSearch):
    """A richer fake covering everything ``lifecycle.py`` and
    ``stats.py`` touch: OCC-guarded ``index``/``get`` (with
    ``_seq_no``/``_primary_term``), ``count``, ``update`` (partial-doc
    merge, for the settings doc), ``indices.delete/create``, and a
    capacity-friendly ``transport``."""

    def __init__(self) -> None:
        super().__init__()
        self._seq: dict[str, int] = {}
        self.indexes: dict[str, list[dict[str, Any]]] = {}
        self.deleted_indexes: list[str] = []
        self.indices = _FakeIndices(self)
        self.transport = _FakeTransport()

    async def get(self, *, index: str, id: str) -> dict[str, Any]:  # noqa: A002, ARG002
        if id not in self.docs:
            return {'found': False, '_id': id}
        return {
            'found': True,
            '_id': id,
            '_source': self.docs[id],
            '_seq_no': self._seq.get(id, 0),
            '_primary_term': 1,
        }

    async def index(
        self,
        *,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        body: dict[str, Any],
        if_seq_no: int | None = None,
        if_primary_term: int | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        if if_seq_no is not None and self._seq.get(id, 0) != if_seq_no:
            raise VersionConflictError(f'seq_no mismatch for {id}')
        self.docs[id] = body
        self._seq[id] = self._seq.get(id, 0) + 1
        return {'_id': id, 'result': 'updated', '_seq_no': self._seq[id]}

    async def update(
        self,
        *,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        body: dict[str, Any],
    ) -> dict[str, Any]:
        doc = body.get('doc') or {}
        current = self.docs.get(id)
        if current is None:
            if not body.get('doc_as_upsert'):
                raise KeyError(id)
            current = {}
        merged = {**current, **doc}
        self.docs[id] = merged
        self._seq[id] = self._seq.get(id, 0) + 1
        return {'_id': id, 'result': 'updated'}

    async def count(self, *, index: str, body: Any = None) -> dict[str, Any]:  # noqa: ARG002
        return {'count': len(self.indexes.get(index, []))}


@pytest.fixture
def fake_lifecycle_client() -> FakeLifecycleOpenSearch:
    return FakeLifecycleOpenSearch()
