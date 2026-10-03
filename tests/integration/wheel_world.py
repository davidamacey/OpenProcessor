"""One in-memory OpenSearch for the wheel-example end-to-end test.

The production code talks to a single cluster; the existing fakes each model
one slice of it (project lifecycle + registry docs, the config store, the
query-evaluating item store). This routes by index name so one client serves
a project that is *created through the lifecycle route*, configured through
the config routes, imported into, processed by the worker and exported from.

It also keeps an audit log of every index each call touched, so a test can
prove a sibling project's indexes were never read or written.
"""

from __future__ import annotations

import json
import re
from typing import Any

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.query_fakes import QueryFakeOpenSearch
from opensearchpy.serializer import JSONSerializer
from projects.conftest import FakeLifecycleOpenSearch


_DATA_INDEX = re.compile(r'^op_prj_.+__(items|images|labels_confirmed|classes|umap_state)$')
_SERIALIZER = JSONSerializer()  # the client's own: handles numpy scalars, datetimes, UUIDs
#: Every audit op that changes the cluster: what "never written" assertions use.
WRITE_OPS = frozenset(
    {
        'delete_index',
        'index',
        'update',
        'delete',
        'create',
        'bulk',
        'put_settings',
        'put_mapping',
        'update_by_query',
        'delete_by_query',
    }
)


def _wire(obj: Any) -> Any:
    """What the cluster would store: a JSON round trip. The in-memory fakes
    keep Python objects, so an enum written by the code under test would
    stay an enum and ``str()`` it differently from the string a real
    cluster hands back."""
    return json.loads(_SERIALIZER.dumps(obj))


class _Indices:
    def __init__(self, outer: RoutedOpenSearch) -> None:
        self._o = outer

    async def create(self, *, index: str, body: Any = None) -> dict[str, Any]:
        self._o._log('create', index)
        if _DATA_INDEX.match(index):
            self._o.data.docs(index)
            return {'acknowledged': True}
        return await self._o.lifecycle.indices.create(index=index, body=body)

    async def exists(self, *, index: str) -> bool:
        if _DATA_INDEX.match(index):
            return index in self._o.data.store
        if self._o._is_config(index) and index in self._o.cfg._docs:
            return True
        return await self._o.lifecycle.indices.exists(index=index)

    async def refresh(self, *, index: str | None = None, **kw: Any) -> dict[str, Any]:
        self._o.lifecycle._refresh_all()
        if index is not None and _DATA_INDEX.match(index):
            return await self._o.data.indices.refresh(index=index, **kw)
        return {'_shards': {'total': 1, 'successful': 1, 'failed': 0}}

    async def delete(self, *, index: str, **kw: Any) -> dict[str, Any]:
        self._o._log('delete_index', index)
        if _DATA_INDEX.match(index):
            self._o.data.store.pop(index, None)
            return {'acknowledged': True}
        return await self._o.lifecycle.indices.delete(index=index, **kw)

    async def get_settings(self, *, index: str, **kw: Any) -> dict[str, Any]:
        return await self._o.data.indices.get_settings(index=index, **kw)

    async def put_settings(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        self._o._log('put_settings', index)
        return await self._o.data.indices.put_settings(index=index, body=body, **kw)

    async def put_mapping(self, *, index: str, **_kw: Any) -> dict[str, Any]:
        """Logged as a write; the mapping itself is not modeled."""
        self._o._log('put_mapping', index)
        return {'acknowledged': True}


class RoutedOpenSearch:
    def __init__(self) -> None:
        self.lifecycle = FakeLifecycleOpenSearch()
        self.cfg = FakeConfigOpenSearch()
        self.data = QueryFakeOpenSearch()
        self.indices = _Indices(self)
        self.transport = self.lifecycle.transport
        #: ``(op, index)`` per call, in order.
        self.audit: list[tuple[str, str]] = []
        self.bulk_results: list[dict[str, Any]] = []

    # --------------------------------------------------------------- routing

    @staticmethod
    def _is_config(index: str) -> bool:
        return index.endswith('__configs') or index == 'op_global_configs'

    def _log(self, op: str, index: str) -> None:
        self.audit.append((op, index))

    def _pick(self, op: str, index: str) -> Any:
        self._log(op, index)
        if self._is_config(index):
            return self.cfg
        if _DATA_INDEX.match(index):
            return self.data
        return self.lifecycle

    def touched(self, *, writes_only: bool = False) -> set[str]:
        return {index for op, index in self.audit if not writes_only or op in WRITE_OPS}

    # ------------------------------------------------------------ operations

    async def get(self, *, index: str, id: str, **kw: Any) -> dict[str, Any]:  # noqa: A002
        return await self._pick('get', index).get(index=index, id=id, **kw)

    async def index(self, *, index: str, **kw: Any) -> dict[str, Any]:
        target = self._pick('index', index)
        if target is self.data:
            kw.pop('refresh', None)
            kw.pop('if_primary_term', None)
            kw.pop('op_type', None)
            kw.pop('if_seq_no', None)
            doc_id = kw['id']
            self.data.docs(index)[doc_id] = _wire(kw['body'])
            self.data.write_calls += 1
            return {'_id': doc_id, 'result': 'created'}
        return await target.index(index=index, **kw)

    async def update(self, *, index: str, **kw: Any) -> dict[str, Any]:
        target = self._pick('update', index)
        if target is self.data and 'body' in kw:
            kw['body'] = _wire(kw['body'])
        return await target.update(index=index, **kw)

    async def delete(self, *, index: str, **kw: Any) -> dict[str, Any]:
        return await self._pick('delete', index).delete(index=index, **kw)

    async def search(self, *, index: str | None = None, **kw: Any) -> dict[str, Any]:
        return await self._pick('search', index or '').search(index=index, **kw)

    async def count(self, *, index: str, **kw: Any) -> dict[str, Any]:
        target = self._pick('count', index)
        if target is self.lifecycle:
            return await target.count(index=index, **kw)
        return await target.count(index=index, **kw)

    async def mget(self, *, body: dict[str, Any], index: str | None = None, **kw: Any) -> Any:
        for spec in body.get('docs') or []:
            self._log('mget', str(spec.get('_index') or index))
        if index is not None:
            self._log('mget', index)
        return await self.data.mget(body=body, index=index, **kw)

    async def bulk(self, *, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        for action in body:
            if len(action) == 1 and next(iter(action)) in {'index', 'update', 'create', 'delete'}:
                meta = next(iter(action.values()))
                if isinstance(meta, dict) and '_index' in meta:
                    self._log('bulk', meta['_index'])
        indexes = {
            next(iter(a.values()))['_index']
            for a in body
            if len(a) == 1
            and next(iter(a)) in {'index', 'update', 'create', 'delete'}
            and isinstance(next(iter(a.values())), dict)
            and '_index' in next(iter(a.values()))
        }
        if indexes and all(_DATA_INDEX.match(i) for i in indexes):
            res = await self.data.bulk(body=_wire(body), **kw)
            self.bulk_results.append(res)
            return res
        return await self.lifecycle.bulk(body=body, **kw)

    async def update_by_query(self, *, index: str, **_kw: Any) -> dict[str, Any]:
        self._log('update_by_query', index)
        raise NotImplementedError('update_by_query is not modeled; the attempt is in the audit')

    async def delete_by_query(self, *, index: str, **_kw: Any) -> dict[str, Any]:
        self._log('delete_by_query', index)
        raise NotImplementedError('delete_by_query is not modeled; the attempt is in the audit')

    async def scroll(self, **kw: Any) -> dict[str, Any]:
        return await self.data.scroll(**kw)

    async def clear_scroll(self, **kw: Any) -> dict[str, Any]:
        return await self.data.clear_scroll(**kw)

    async def close(self) -> None:
        return None
