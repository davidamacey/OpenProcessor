"""Fixtures shared by tests/projects/*.

Tests here that exercise binding itself opt out of tests/conftest.py's
autouse default-project bind with ``@pytest.mark.unbound``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


class _FakeIndices:
    def __init__(self, owner: FakeRegistryOpenSearch | None = None) -> None:
        self.created: dict[str, dict[str, Any]] = {}
        self._owner = owner

    async def exists(self, *, index: str) -> bool:
        return index in self.created

    async def create(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.created[index] = body
        return {'acknowledged': True}

    async def refresh(self, *, index: str | None = None) -> dict[str, Any]:  # noqa: ARG002
        if self._owner is not None:
            self._owner._refresh_all()
        return {'_shards': {'total': 0, 'successful': 0, 'failed': 0}}


class FakeRegistryOpenSearch:
    """Just enough of AsyncOpenSearch for ``ProjectRegistry``/
    ``bootstrap_default_project``: ``get``/``index``/``search`` on one
    flat doc store, keyed by id, with seq_no OCC like the real thing.

    B2: models near-real-time search visibility -- a doc written without
    ``refresh='wait_for'``/``'true'`` is gettable by id immediately (like
    real OpenSearch) but invisible to ``search()`` until an explicit
    ``indices.refresh()`` or a subsequent ``wait_for``/``true`` write.
    Without this, a test using the fake could never have caught B2's live
    bug: ``ensure_fresh()``'s ``_search`` seeing the pre-write registry
    snapshot even though the write it's racing already returned."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.seq: dict[str, int] = {}
        self._visible: set[str] = set()
        self.indices = _FakeIndices(self)

    def _refresh_all(self) -> None:
        self._visible = set(self.docs.keys())

    async def get(self, *, index: str, id: str) -> dict[str, Any]:  # noqa: A002, ARG002
        await asyncio.sleep(0)  # a real read yields; lets concurrent writers interleave
        if id not in self.docs:
            return {'found': False, '_id': id}
        return {
            'found': True,
            '_id': id,
            '_source': self.docs[id],
            '_seq_no': self.seq[id],
            '_primary_term': 1,
        }

    async def index(
        self,
        *,
        index: str,  # noqa: ARG002 - mirrors the client signature
        id: str,  # noqa: A002
        body: dict[str, Any],
        op_type: str | None = None,
        if_seq_no: int | None = None,
        if_primary_term: int | None = None,  # noqa: ARG002 - one term in the fake
        refresh: str | bool | None = None,
    ) -> dict[str, Any]:
        from opensearchpy.exceptions import ConflictError

        if op_type == 'create' and id in self.docs:
            raise ConflictError(409, 'version_conflict_engine_exception', {})
        if if_seq_no is not None and self.seq.get(id) != if_seq_no:
            raise ConflictError(409, 'version_conflict_engine_exception', {})
        self.docs[id] = body
        self.seq[id] = self.seq.get(id, 0) + 1
        if refresh in ('wait_for', 'true', True):
            self._visible.add(id)
        else:
            self._visible.discard(id)
        return {'_id': id, 'result': 'created'}

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        from opensearchpy.exceptions import RequestError

        query = body.get('query') or {}
        if '_id' in (query.get('prefix') or {}):
            # Real OpenSearch refuses prefix queries on _id
            # (query_shard_exception); the fake must too, or the registry's
            # refresh query passes here and fails on a live cluster.
            raise RequestError(400, 'query_shard_exception', {})
        exists_field = (query.get('exists') or {}).get('field')
        hits: list[dict[str, Any]] = [
            {'_id': doc_id, '_source': doc}
            for doc_id, doc in sorted(self.docs.items())
            if doc_id in self._visible and (exists_field is None or exists_field in doc)
        ]
        after = body.get('search_after')
        if after:
            hits = [h for h in hits if h['_source']['slug'] > after[0]]
        hits = hits[: body.get('size', 10)]
        for hit in hits:
            hit['sort'] = [hit['_source']['slug']]
        return {'hits': {'hits': hits}}


@pytest.fixture
def fake_registry_client() -> FakeRegistryOpenSearch:
    return FakeRegistryOpenSearch()


class _FakeLifecycleIndices:
    def __init__(self, outer: FakeLifecycleOpenSearch) -> None:
        self._outer = outer

    async def delete(self, *, index: str, ignore: Any = None) -> dict[str, Any]:  # noqa: ARG002
        self._outer.deleted_indexes.append(index)
        self._outer.indexes.pop(index, None)
        return {'acknowledged': True}

    async def create(self, *, index: str, body: Any = None) -> dict[str, Any]:  # noqa: ARG002
        self._outer.indexes[index] = self._outer.indexes.get(index, [])
        return {'acknowledged': True}

    async def exists(self, *, index: str) -> bool:
        return index in self._outer.indexes

    async def refresh(self, *, index: str) -> dict[str, Any]:  # noqa: ARG002
        self._outer._refresh_all()
        return {'_shards': {'total': 0, 'successful': 0, 'failed': 0}}


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
        self.indices: Any = _FakeLifecycleIndices(self)
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
        op_type: str | None = None,
        if_seq_no: int | None = None,
        if_primary_term: int | None = None,  # noqa: ARG002
        refresh: str | bool | None = None,
    ) -> dict[str, Any]:
        if op_type == 'create' and id in self.docs:
            raise VersionConflictError(f'doc already exists for {id}')
        if if_seq_no is not None and self._seq.get(id, 0) != if_seq_no:
            raise VersionConflictError(f'seq_no mismatch for {id}')
        self.docs[id] = body
        self._seq[id] = self._seq.get(id, 0) + 1
        if refresh in ('wait_for', 'true', True):
            self._visible.add(id)
        else:
            self._visible.discard(id)
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

    async def bulk(self, *, body: list[Any], refresh: bool = False) -> dict[str, Any]:  # noqa: ARG002
        """Just enough of ``_bulk`` for ``ClassRegistry.sync_to_opensearch``:
        every other-odd item is an action header, every even item its doc."""
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            index = next(iter(action.values()))['_index']
            doc_id = next(iter(action.values())).get('_id')
            self.indexes.setdefault(index, [])
            if doc_id is not None:
                self.docs[f'{index}:{doc_id}'] = doc
            self.indexes[index].append(doc)
        return {'errors': False, 'items': []}


@pytest.fixture
def fake_lifecycle_client() -> FakeLifecycleOpenSearch:
    return FakeLifecycleOpenSearch()


async def seed_default_project(client: Any) -> Any:
    """``default`` is now an ordinary project record (no env-synthesis
    fallback in ``lifecycle._get_mutable_record``); tests that need to
    archive/delete/protect it must first bootstrap it exactly like
    ``src.main``'s startup lifespan does."""
    from src.services.projects.bootstrap import bootstrap_default_project

    return await bootstrap_default_project(client)


@pytest.fixture
def not_stale_registry(monkeypatch: pytest.MonkeyPatch) -> Any:
    """M2 made a stale registry refuse every non-GET route at bind time
    (``project_read_only``, defence for file-backed writes the guard
    never saw). The process-wide default test registry
    (``tests/conftest.py::_requests_start_unbound``) never successfully
    refreshes on purpose, so it is always ``stale`` -- fine for tests
    that only read, but any write-route test now needs one that
    actually succeeds. Opt in with this fixture."""
    from src.services.projects import registry as registry_mod
    from src.services.projects.registry import ProjectRegistry

    registry = ProjectRegistry(lambda: None)
    registry._by_slug = dict(registry_mod.get_project_registry().snapshot())

    async def _fresh(self: ProjectRegistry) -> None:
        return None

    monkeypatch.setattr(ProjectRegistry, 'ensure_fresh', _fresh)
    registry_mod.set_project_registry(registry)
    return registry
