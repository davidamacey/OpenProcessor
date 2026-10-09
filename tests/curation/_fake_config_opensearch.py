"""In-memory fake covering the small OpenSearch surface the config store
(``src.services.config_store``) uses: ``get`` / ``index`` / ``update`` /
``delete`` / ``search``, with real OCC (``if_seq_no``/``if_primary_term``)
and the one painless script ``bump_config_revision`` sends.

Not a test module itself -- imported by ``tests/curation/test_config_*.py``.
"""

from __future__ import annotations

from typing import Any

from opensearchpy.exceptions import ConflictError, NotFoundError


class _FakeIndices:
    """``client.indices.refresh(...)`` -- a no-op here since
    :class:`FakeConfigOpenSearch`'s ``_search`` is always fully
    consistent with ``index``/``update`` (no simulated NRT lag). See
    :class:`NearRealTimeConfigOpenSearch` below for a fake that models
    the lag ``refresh`` is meant to force through.

    ``exists``/``create`` back a bare in-memory set of created index
    names -- just enough for ``ensure_global_configs_index``'s
    create-if-missing check (op_global_configs, W2 M3)."""

    def __init__(self) -> None:
        self._created: set[str] = set()

    async def refresh(self, index: str) -> dict[str, Any]:  # noqa: ARG002
        return {'_shards': {'total': 1, 'successful': 1, 'failed': 0}}

    async def exists(self, index: str) -> bool:
        return index in self._created

    async def create(self, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        self._created.add(index)
        return {'acknowledged': True}


class FakeConfigOpenSearch:
    def __init__(self) -> None:
        # index -> doc_id -> {"_source": {...}, "_seq_no": int}
        self._docs: dict[str, dict[str, dict[str, Any]]] = {}
        self._seq = 0
        self.indices: Any = _FakeIndices()

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def get(self, index: str, id: str, ignore: Any = None) -> dict[str, Any]:  # noqa: A002
        del ignore
        entry = self._docs.get(index, {}).get(id)
        if entry is None:
            raise NotFoundError(404, f'[404] not found: {index}/{id}', {})
        return {
            '_source': dict(entry['_source']),
            '_seq_no': entry['_seq_no'],
            '_primary_term': 1,
        }

    async def index(
        self,
        index: str,
        id: str,  # noqa: A002
        body: dict[str, Any],
        if_seq_no: int | None = None,
        if_primary_term: int | None = None,
        refresh: str | bool | None = None,
    ) -> dict[str, Any]:
        del if_primary_term, refresh
        docs = self._docs.setdefault(index, {})
        entry = docs.get(id)
        if if_seq_no is not None and (entry is None or entry['_seq_no'] != if_seq_no):
            raise ConflictError(409, 'version_conflict_engine_exception', {})
        docs[id] = {'_source': dict(body), '_seq_no': self._next_seq()}
        return {'result': 'created' if entry is None else 'updated'}

    async def update(
        self,
        index: str,
        id: str,  # noqa: A002
        body: dict[str, Any],
        retry_on_conflict: int = 0,
        refresh: str | bool | None = None,
    ) -> dict[str, Any]:
        del retry_on_conflict, refresh
        docs = self._docs.setdefault(index, {})
        entry = docs.get(id)
        if entry is None:
            if body.get('doc_as_upsert'):
                upsert = dict(body.get('doc') or {})
            else:
                upsert = dict(body.get('upsert') or {})
            docs[id] = {'_source': upsert, '_seq_no': self._next_seq()}
            return {'result': 'created'}
        src = dict(entry['_source'])
        script = (body.get('script') or {}).get('source', '')
        if 'config_revision += 1' in script:
            src['config_revision'] = int(src.get('config_revision', 0)) + 1
        elif 'doc' in body:
            # Mirrors OpenSearch's real partial-update recursive-merge
            # semantics for object fields (the settings doc's ``defaults``
            # map in particular) -- a plain top-level overwrite would
            # clobber axes not mentioned in this call.
            for field, value in body['doc'].items():
                if isinstance(value, dict) and isinstance(src.get(field), dict):
                    merged = dict(src[field])
                    merged.update(value)
                    src[field] = merged
                else:
                    src[field] = value
        docs[id] = {'_source': src, '_seq_no': self._next_seq()}
        return {'result': 'updated'}

    async def delete(
        self,
        index: str,
        id: str,  # noqa: A002
        if_seq_no: int | None = None,
        if_primary_term: int | None = None,
        refresh: str | bool | None = None,
    ) -> dict[str, Any]:
        del if_primary_term, refresh
        docs = self._docs.get(index, {})
        entry = docs.get(id)
        if entry is None:
            raise NotFoundError(404, f'[404] not found: {index}/{id}', {})
        if if_seq_no is not None and entry['_seq_no'] != if_seq_no:
            raise ConflictError(409, 'version_conflict_engine_exception', {})
        del docs[id]
        return {'result': 'deleted'}

    async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        docs = self._docs.get(index, {})
        filters = (body.get('query', {}).get('bool', {}) or {}).get('filter', [])

        def matches(src: dict[str, Any]) -> bool:
            for f in filters:
                for key, value in f.get('term', {}).items():
                    if src.get(key) != value:
                        return False
            return True

        hits = [
            {'_id': doc_id, '_source': entry['_source']}
            for doc_id, entry in docs.items()
            if matches(entry['_source'])
        ]
        sort = body.get('sort')
        if sort:
            key, spec = next(iter(sort[0].items()))
            order = spec.get('order', 'asc') if isinstance(spec, dict) else 'asc'
            hits.sort(key=lambda h: h['_source'].get(key, 0), reverse=(order == 'desc'))
        size = body.get('size', 10)
        return {'hits': {'hits': hits[:size]}}

    async def count(self, index: str, body: dict[str, Any] | None = None) -> dict[str, Any]:  # noqa: ARG002
        """This fake stores config docs only: no items, so nothing to count
        (the profile-impact stale-item count runs on activation)."""
        return {'count': 0}


class NearRealTimeConfigOpenSearch(FakeConfigOpenSearch):
    """Models the real gap ``indices.refresh()`` exists to close:
    ``get``/``index``/``update`` (realtime) write straight through, but
    ``search`` serves a snapshot taken at construction time (or at the
    last explicit ``indices.refresh()``) until one is called again --
    B5's reviewer probe #8 ("save+activate, refresh, index refresh,
    refresh" still stale) reproduced as a fixture other tests can drive.
    """

    def __init__(self) -> None:
        super().__init__()
        self._search_snapshot: dict[str, dict[str, dict[str, Any]]] = {}
        self.indices = _LaggingFakeIndices(self)

    async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        docs = self._search_snapshot.get(index, {})
        filters = (body.get('query', {}).get('bool', {}) or {}).get('filter', [])

        def matches(src: dict[str, Any]) -> bool:
            for f in filters:
                for key, value in f.get('term', {}).items():
                    if src.get(key) != value:
                        return False
            return True

        hits = [
            {'_id': doc_id, '_source': entry['_source']}
            for doc_id, entry in docs.items()
            if matches(entry['_source'])
        ]
        size = body.get('size', 10)
        return {'hits': {'hits': hits[:size]}}

    def _sync_search_snapshot(self) -> None:
        import copy

        self._search_snapshot = copy.deepcopy(self._docs)


class _LaggingFakeIndices:
    def __init__(self, client: NearRealTimeConfigOpenSearch) -> None:
        self._client = client

    async def refresh(self, index: str) -> dict[str, Any]:  # noqa: ARG002
        self._client._sync_search_snapshot()
        return {'_shards': {'total': 1, 'successful': 1, 'failed': 0}}


class TwoProjectOpenSearch(FakeConfigOpenSearch):
    """``FakeConfigOpenSearch`` plus enough ``search`` support for
    ``ProjectRegistry._refresh`` (an ``exists: field=slug`` query, no
    ``bool.filter``) so one fake client can serve the registry, every
    project's own config store, and the (always-empty) items index."""

    async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        query = body.get('query', {})
        exists_field = query.get('exists', {}).get('field')
        if exists_field is None:
            return await super().search(index=index, body=body)
        docs = self._docs.get(index, {})
        hits = [
            {'_id': doc_id, '_source': entry['_source']}
            for doc_id, entry in docs.items()
            if exists_field in entry['_source']
        ]
        return {'hits': {'hits': hits}}
