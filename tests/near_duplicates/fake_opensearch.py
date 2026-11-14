"""An OpenSearch stand-in with the calls a near-duplicate run makes.

Documents are ``{index: {id: source}}``. It answers ``mget`` (with ``_source_includes``),
``msearch`` k-NN queries (exact cosine over the stored vectors, with a ``terms`` pre-filter),
point-in-time searches (a snapshot at create time, ``search_after``, ``slice``, ``docvalue_fields``), ``bulk``
partial updates and ``update_by_query`` field removal.
"""

from __future__ import annotations

import base64
import copy
import zlib
from typing import Any

import numpy as np
from opensearchpy.exceptions import NotFoundError


FIELDS = (
    'near_dup_group_id',
    'near_dup_leak_group_id',
    'near_dup_is_representative',
    'near_dup_group_size',
    'near_dup_run_id',
)


def _docvalues(source: dict[str, Any], field: str) -> list[Any]:
    """A ``knn_vector`` through ``docvalue_fields``: one base64 little-endian float32 string (3.7)."""
    value = source[field]
    if isinstance(value, list) and value and all(isinstance(v, float) for v in value):
        return [base64.b64encode(np.asarray(value, dtype='<f4').tobytes()).decode('ascii')]
    return [value]


class FakeNearDupOpenSearch:
    def __init__(self, docs: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.docs = docs
        self.bulk_ops = 0
        self.pits: dict[str, dict[str, dict[str, Any]]] = {}
        self._pit_seq = 0

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

    # -- point in time ---------------------------------------------------------------------
    # OpenSearch 3.x: POST /{index}/_search/point_in_time creates a view of the index as it is
    # now; a search body with ``pit`` (and no index) reads that view, ``sort: _shard_doc`` pages
    # it with ``search_after`` and ``slice`` splits it into disjoint parts; DELETE
    # /_search/point_in_time frees it, and a search on a freed one is refused.
    async def create_pit(self, *, index: str, **_kw: Any) -> dict[str, Any]:
        self._pit_seq += 1
        pit_id = f'pit-{self._pit_seq}'
        self.pits[pit_id] = copy.deepcopy(self.docs.get(index, {}))
        return {'pit_id': pit_id}

    async def delete_pit(self, *, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        for pit_id in body['pit_id']:
            self.pits.pop(pit_id)
        return {}

    async def search(self, *, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        pit_id = body['pit']['id']
        if pit_id not in self.pits:
            raise NotFoundError(404, 'search_context_missing_exception', pit_id)
        view = self.pits[pit_id]
        sliced = body.get('slice')
        after = (body.get('search_after') or [-1])[0]
        wanted = body['_source']
        fields = body.get('docvalue_fields', [])
        hits = []
        for position, doc_id in enumerate(sorted(view)):
            source = view[doc_id]
            if position <= after or not self._matches(body['query'], source):
                continue
            if sliced and zlib.crc32(doc_id.encode()) % sliced['max'] != sliced['id']:
                continue
            hit: dict[str, Any] = {'_id': doc_id, 'sort': [position]}
            if wanted is not False:
                hit['_source'] = {k: copy.deepcopy(v) for k, v in source.items() if k in wanted}
            if fields:
                hit['fields'] = {f: _docvalues(source, f) for f in fields if f in source}
            hits.append(hit)
            if len(hits) == body['size']:
                break
        return {'hits': {'hits': hits}}

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
