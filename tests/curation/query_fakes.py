"""In-memory AsyncOpenSearch double that actually evaluates queries.

Most curation test fakes return canned hits and assert on the query body.
The operator tools tested with this one (registry reclassification, region
requeue, the label -> export round trip) are only correct if their query
*selects the right documents*, so this double stores docs per index and
evaluates the subset of the query DSL those paths use:

- ``term`` / ``terms`` / ``exists`` / ``match_all`` and ``bool`` with
  ``must`` / ``filter`` / ``must_not`` / ``should`` (``should`` = any-of);
- ``search`` with ``size``, a single-field ``sort``, ``search_after``,
  ``scroll`` (everything in the first page) and ``terms`` aggregations
  (with ``missing`` and nested sub-aggregations, incl. ``top_hits``);
- ``count``, ``get``, ``update`` (``if_seq_no`` honoured), ``mget``,
  ``bulk`` (``update`` with ``if_seq_no`` and ``index``), ``indices.refresh``.

A ``None`` field value is treated as absent, matching how OpenSearch never
indexes nulls (``exists`` is false for them).
"""

from __future__ import annotations

import copy
import re
from typing import Any


class _ConflictError(Exception):
    """Stands in for opensearchpy's ConflictError (name contains 'Conflict')."""


def _values(doc: dict[str, Any], field: str) -> list[Any]:
    value = doc.get(field)
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v is not None]
    return [value]


def _wildcard_regex(value: str) -> str:
    out, chars = [], iter(value)
    for c in chars:
        if c == '\\':
            out.append(re.escape(next(chars, '')))
        elif c == '*':
            out.append('.*')
        elif c == '?':
            out.append('.')
        else:
            out.append(re.escape(c))
    return ''.join(out)


def _wildcard_matches(doc: dict[str, Any], clause: dict[str, Any]) -> bool:
    ((field, spec),) = clause.items()
    if not isinstance(spec, dict):
        spec = {'value': spec}
    flags = re.IGNORECASE if spec.get('case_insensitive') else 0
    pattern = _wildcard_regex(spec['value'])
    return any(re.fullmatch(pattern, str(v), flags) is not None for v in _values(doc, field))


_LEAF_MATCHERS = {
    'exists': lambda doc, clause: bool(_values(doc, clause['field'])),
    'wildcard': _wildcard_matches,
}


def matches(doc: dict[str, Any], query: dict[str, Any] | None) -> bool:
    if not query or 'match_all' in query:
        return True
    leaf = next((k for k in _LEAF_MATCHERS if k in query), None)
    if leaf is not None:
        return _LEAF_MATCHERS[leaf](doc, query[leaf])
    if 'term' in query:
        ((field, value),) = query['term'].items()
        if isinstance(value, dict):
            value = value['value']
        return value in _values(doc, field)
    if 'terms' in query:
        ((field, values),) = query['terms'].items()
        return any(v in values for v in _values(doc, field))
    if 'range' in query:
        ((field, bounds),) = query['range'].items()
        ops = {
            'gt': lambda a, b: a > b,
            'gte': lambda a, b: a >= b,
            'lt': lambda a, b: a < b,
            'lte': lambda a, b: a <= b,
        }
        return any(
            all(ops[op](v, bound) for op, bound in bounds.items() if op in ops)
            for v in _values(doc, field)
            if v is not None
        )
    if 'bool' in query:
        b = query['bool']
        for key in ('must', 'filter'):
            clauses = b.get(key) or []
            if isinstance(clauses, dict):
                clauses = [clauses]
            if not all(matches(doc, c) for c in clauses):
                return False
        must_not = b.get('must_not') or []
        if isinstance(must_not, dict):
            must_not = [must_not]
        if any(matches(doc, c) for c in must_not):
            return False
        should = b.get('should') or []
        return not should or any(matches(doc, c) for c in should)
    raise NotImplementedError(f'query clause not supported by fake: {query}')


def _aggregate(docs: list[dict[str, Any]], aggs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in aggs.items():
        if 'top_hits' in spec:
            size = spec['top_hits'].get('size', 3)
            out[name] = {'hits': {'hits': [{'_source': copy.deepcopy(d)} for d in docs[:size]]}}
            continue
        if 'terms' not in spec:
            raise NotImplementedError(f'agg not supported by fake: {spec}')
        field = spec['terms']['field']
        missing = spec['terms'].get('missing')
        groups: dict[Any, list[dict[str, Any]]] = {}
        for doc in docs:
            vals = _values(doc, field)
            if not vals and missing is not None:
                vals = [missing]
            for v in vals:
                groups.setdefault(v, []).append(doc)
        buckets = []
        for key, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
            bucket: dict[str, Any] = {'key': key, 'doc_count': len(members)}
            if spec.get('aggs'):
                bucket.update(_aggregate(members, spec['aggs']))
            buckets.append(bucket)
        out[name] = {'buckets': buckets[: spec['terms'].get('size', 10)]}
    return out


class QueryFakeOpenSearch:
    def __init__(self, indexes: dict[str, dict[str, dict[str, Any]]] | None = None) -> None:
        self.store: dict[str, dict[str, dict[str, Any]]] = {
            name: {doc_id: copy.deepcopy(doc) for doc_id, doc in docs.items()}
            for name, docs in (indexes or {}).items()
        }
        self.seq: dict[tuple[str, str], int] = {}
        self.searched_indexes: list[str] = []
        self.bulk_calls = 0
        self.indices = _Indices(self)

    # ------------------------------------------------------------------ helpers

    def docs(self, index: str) -> dict[str, dict[str, Any]]:
        return self.store.setdefault(index, {})

    def _bump(self, index: str, doc_id: str) -> int:
        self.seq[(index, doc_id)] = self.seq.get((index, doc_id), 1) + 1
        return self.seq[(index, doc_id)]

    def _hit(self, index: str, doc_id: str, doc: dict[str, Any]) -> dict[str, Any]:
        return {'_index': index, '_id': doc_id, '_source': copy.deepcopy(doc)}

    # ------------------------------------------------------------------ search

    async def search(
        self,
        *,
        index: str,
        body: dict[str, Any],
        scroll: str | None = None,
        **_kw: Any,
    ) -> dict[str, Any]:
        self.searched_indexes.append(index)
        pool = [
            (doc_id, doc)
            for doc_id, doc in self.docs(index).items()
            if matches(doc, body.get('query'))
        ]
        sort = body.get('sort')
        sort_field = None
        if sort:
            first = sort[0]
            sort_field = first if isinstance(first, str) else next(iter(first))
            pool.sort(key=lambda kv: str(kv[1].get(sort_field, kv[0])))
            if body.get('search_after') is not None:
                after = str(body['search_after'][0])
                pool = [kv for kv in pool if str(kv[1].get(sort_field, kv[0])) > after]
        size = body.get('size', 10) if scroll is None else len(pool)
        hits = []
        for doc_id, doc in pool[:size]:
            hit = self._hit(index, doc_id, doc)
            if sort_field is not None:
                hit['sort'] = [doc.get(sort_field, doc_id)]
            hits.append(hit)
        resp: dict[str, Any] = {'hits': {'total': {'value': len(pool)}, 'hits': hits}}
        if body.get('aggs'):
            resp['aggregations'] = _aggregate([doc for _id, doc in pool], body['aggs'])
        if scroll is not None:
            resp['_scroll_id'] = 'fake-scroll'
        return resp

    async def scroll(self, *, scroll_id: str, **_kw: Any) -> dict[str, Any]:
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, **_kw: Any) -> dict[str, Any]:
        return {}

    async def count(self, *, index: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        query = (body or {}).get('query')
        return {'count': sum(1 for d in self.docs(index).values() if matches(d, query))}

    # ------------------------------------------------------------------ doc ops

    async def get(self, *, index: str, id: str, **_kw: Any) -> dict[str, Any]:  # noqa: A002
        doc = self.docs(index)[id]
        return {
            '_id': id,
            '_source': copy.deepcopy(doc),
            '_seq_no': self.seq.get((index, id), 1),
            '_primary_term': 1,
            'found': True,
        }

    async def update(
        self,
        *,
        index: str,
        id: str,  # noqa: A002
        body: dict[str, Any],
        if_seq_no: int | None = None,
        **_kw: Any,
    ) -> dict[str, Any]:
        if if_seq_no is not None and if_seq_no != self.seq.get((index, id), 1):
            raise _ConflictError('409 version conflict')
        self.docs(index)[id].update(copy.deepcopy(body['doc']))
        self._bump(index, id)
        return {'result': 'updated'}

    async def mget(
        self, *, body: dict[str, Any], index: str | None = None, **_kw: Any
    ) -> dict[str, Any]:
        out = []
        specs = body.get('docs') or [{'_id': i, '_index': index} for i in body.get('ids', [])]
        for spec in specs:
            idx = str(spec.get('_index') or index)
            doc = self.docs(idx).get(spec['_id'])
            if doc is None:
                out.append({'_id': spec['_id'], 'found': False})
                continue
            out.append(
                {
                    '_index': idx,
                    '_id': spec['_id'],
                    'found': True,
                    '_source': copy.deepcopy(doc),
                    '_seq_no': self.seq.get((idx, spec['_id']), 1),
                    '_primary_term': 1,
                }
            )
        return {'docs': out}

    async def bulk(self, *, body: list[dict[str, Any]], **_kw: Any) -> dict[str, Any]:
        self.bulk_calls += 1
        items = []
        errors = False
        for action, payload in zip(body[0::2], body[1::2], strict=True):
            ((op, meta),) = action.items()
            idx, doc_id = meta['_index'], meta['_id']
            if op == 'index':
                self.docs(idx)[doc_id] = copy.deepcopy(payload)
                self._bump(idx, doc_id)
                items.append({op: {'_id': doc_id, 'status': 201}})
                continue
            if op != 'update':
                raise NotImplementedError(op)
            if doc_id not in self.docs(idx):
                errors = True
                items.append({op: {'_id': doc_id, 'status': 404, 'error': {'type': 'nf'}}})
                continue
            if 'if_seq_no' in meta and meta['if_seq_no'] != self.seq.get((idx, doc_id), 1):
                errors = True
                items.append(
                    {
                        op: {
                            '_id': doc_id,
                            'status': 409,
                            'error': {'type': 'version_conflict_engine_exception'},
                        }
                    }
                )
                continue
            self.docs(idx)[doc_id].update(copy.deepcopy(payload['doc']))
            self._bump(idx, doc_id)
            items.append({op: {'_id': doc_id, 'status': 200, 'result': 'updated'}})
        return {'errors': errors, 'items': items}


class _Indices:
    def __init__(self, parent: QueryFakeOpenSearch) -> None:
        self.parent = parent
        self.refreshed: list[str] = []

    async def refresh(self, *, index: str, **_kw: Any) -> dict[str, Any]:
        self.refreshed.append(index)
        return {}

    async def exists(self, *, index: str) -> bool:
        return index in self.parent.store
