"""An OpenSearch stand-in with the calls a near-duplicate run makes.

Documents are ``{index: {id: source}}``. It answers ``mget`` (with ``_source_includes``),
``msearch`` k-NN queries (exact cosine over the stored vectors, with a ``terms`` pre-filter),
the scroll calls the page reader falls back to when no point in time can be opened, ``bulk``
partial updates and ``update_by_query`` field removal.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


FIELDS = (
    'near_dup_group_id',
    'near_dup_leak_group_id',
    'near_dup_is_representative',
    'near_dup_group_size',
    'near_dup_run_id',
)


class FakeNearDupOpenSearch:
    def __init__(self, docs: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.docs = docs
        self.bulk_ops = 0

    # -- reads -----------------------------------------------------------------------------
    async def mget(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        includes = kw.get('_source_includes')
        out = []
        for doc_id in body['ids']:
            source = self.docs.get(index, {}).get(doc_id)
            if source is None:
                out.append({'_id': doc_id, 'found': False})
                continue
            kept = {k: v for k, v in source.items() if includes is None or k in includes}
            out.append({'_id': doc_id, 'found': True, '_source': copy.deepcopy(kept)})
        return {'docs': out}

    async def msearch(self, *, body: list[dict[str, Any]]) -> dict[str, Any]:
        responses = []
        for header, query in zip(body[::2], body[1::2], strict=True):
            responses.append(self._knn(header['index'], query))
        return {'responses': responses}

    def _knn(self, index: str, query: dict[str, Any]) -> dict[str, Any]:
        ((field, knn),) = query['query']['knn'].items()
        vector = np.asarray(knn['vector'], dtype=np.float32)
        allowed = None
        if 'filter' in knn:
            (terms,) = knn['filter']['terms'].items()
            allowed = set(terms[1])
        scored = []
        for doc_id, source in self.docs.get(index, {}).items():
            other = source.get(field)
            if other is None:
                continue
            if allowed is not None and source.get('image_id') not in allowed:
                continue
            scored.append((float(np.asarray(other, dtype=np.float32) @ vector), doc_id))
        scored.sort(key=lambda t: (-t[0], t[1]))
        hits = []
        for score, doc_id in scored[: knn['k']]:
            source = {k: v for k, v in self.docs[index][doc_id].items() if k in query['_source']}
            hits.append({'_id': doc_id, '_score': score, '_source': copy.deepcopy(source)})
        return {'hits': {'hits': hits}}

    async def create_pit(self, **_kw: Any) -> dict[str, Any]:
        raise RuntimeError('no point in time in the stand-in')

    async def search(self, *, index: str, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        query = body['query']
        hits = []
        for doc_id, source in sorted(self.docs.get(index, {}).items()):
            if self._matches(query, source):
                kept = {k: v for k, v in source.items() if k in body['_source']}
                hits.append({'_id': doc_id, '_source': copy.deepcopy(kept)})
        self._pending = [hits, []]
        return {'_scroll_id': 's', 'hits': {'hits': self._pending.pop(0)}}

    async def scroll(self, **_kw: Any) -> dict[str, Any]:
        return {'_scroll_id': 's', 'hits': {'hits': self._pending.pop(0) if self._pending else []}}

    async def clear_scroll(self, **_kw: Any) -> None:
        return None

    @staticmethod
    def _matches(query: dict[str, Any], source: dict[str, Any]) -> bool:
        if 'exists' in query:
            return source.get(query['exists']['field']) is not None
        if 'bool' in query:
            return all(FakeNearDupOpenSearch._matches(q, source) for q in query['bool']['filter'])
        if 'terms' in query:
            ((name, values),) = query['terms'].items()
            return source.get(name) in values
        raise NotImplementedError(query)

    # -- writes ----------------------------------------------------------------------------
    async def bulk(self, *, body: list[dict[str, Any]], **_kw: Any) -> dict[str, Any]:
        items = []
        for action, payload in zip(body[::2], body[1::2], strict=True):
            meta = action['update']
            source = self.docs.get(meta['_index'], {}).get(meta['_id'])
            if source is None:
                items.append({'update': {'status': 404}})
                continue
            self.bulk_ops += 1
            if 'doc' in payload:
                source.update(payload['doc'])
            else:
                for field in FIELDS:
                    source.pop(field, None)
            items.append({'update': {'status': 200}})
        return {'items': items}

    async def update_by_query(
        self, *, index: str, body: dict[str, Any], **_kw: Any
    ) -> dict[str, Any]:
        keep = body['query']['bool']['must_not'][0]['term']['near_dup_run_id']
        for source in self.docs.get(index, {}).values():
            run = source.get('near_dup_run_id')
            if run is not None and run != keep:
                for field in FIELDS:
                    source.pop(field, None)
        return {}
