"""In-memory AsyncOpenSearch double that actually evaluates queries.

Most curation test fakes return canned hits and assert on the query body.
The operator tools tested with this one (registry reclassification, region
requeue, the label -> export round trip) are only correct if their query
*selects the right documents*, so this double stores docs per index and
evaluates the subset of the query DSL those paths use:

- ``term`` / ``terms`` / ``exists`` / ``match_all`` and ``bool`` with
  ``must`` / ``filter`` / ``must_not`` / ``should`` (``should`` = any-of);
- ``nested`` (W8c): matches the parent doc when any element of the nested
  list field satisfies the inner query. Elements are re-keyed with the
  ``<path>.`` prefix (e.g. ``region_boxes.detector``) before evaluation, so
  a nested clause's field names — always the full dotted path, matching
  real OpenSearch and :func:`~src.services.curation.region_boxes.box_query`
  — resolve the same way a top-level field does;
- ``search`` with ``size``, a single-field ``sort``, ``search_after``,
  ``scroll`` (everything in the first page) and ``terms`` aggregations
  (with ``missing`` and nested sub-aggregations, incl. ``top_hits``), plus
  a ``nested`` aggregation (flattens each doc's nested elements, prefixed
  the same way, before running its sub-aggs — so a nested terms agg
  counts BOXES, not items; fine for dry-run reporting, not exact for an
  item with 2+ boxes in the same bucket);
- ``inner_hits`` on a top-level ``nested`` clause (the matching elements'
  ``_nested.offset``; a ``size`` above ``max_inner_result_window`` -- 100
  unless a test sets it through ``indices.put_settings`` -- raises like
  OpenSearch's 400), and ``top_hits`` inside a nested aggregation with
  ``docvalue_fields`` (parent ``_id`` + the requested ``fields``);
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


def _term_matches(doc: dict[str, Any], clause: dict[str, Any]) -> bool:
    ((field, value),) = clause.items()
    if isinstance(value, dict) and value.get('case_insensitive'):
        wanted = str(value['value']).casefold()
        return any(str(v).casefold() == wanted for v in _values(doc, field))
    if isinstance(value, dict):
        value = value['value']
    return value in _values(doc, field)


_REVERSE_NESTED_PARENT_KEY = '__parent_id__'
_DOC_ID_KEY = '__doc_id__'
_PARENT_DOC_KEY = '__parent_doc__'


def _nested_elements(doc: dict[str, Any], path: str) -> list[dict[str, Any]]:
    """Every element of ``doc[path]``, re-keyed with the ``<path>.`` prefix
    so a nested clause's dotted field names resolve like a top-level field.

    Each flattened element also carries a private ``__parent_id__`` (the
    parent doc's identity) so a ``reverse_nested`` sub-agg can count
    distinct parent docs instead of nested elements."""
    elements = doc.get(path) or []
    if not isinstance(elements, list):
        return []
    prefix = f'{path}.'
    return [
        {f'{prefix}{k}': v for k, v in el.items()}
        | {
            _REVERSE_NESTED_PARENT_KEY: id(doc),
            _DOC_ID_KEY: doc.get(_DOC_ID_KEY),
            _PARENT_DOC_KEY: doc,
        }
        for el in elements
        if isinstance(el, dict)
    ]


def _nested_matches(doc: dict[str, Any], clause: dict[str, Any]) -> bool:
    return any(matches(el, clause['query']) for el in _nested_elements(doc, clause['path']))


_LEAF_MATCHERS = {
    'exists': lambda doc, clause: bool(_values(doc, clause['field'])),
    'wildcard': _wildcard_matches,
    'nested': _nested_matches,
}


def matches(doc: dict[str, Any], query: dict[str, Any] | None) -> bool:
    if not query or 'match_all' in query:
        return True
    leaf = next((k for k in _LEAF_MATCHERS if k in query), None)
    if leaf is not None:
        return _LEAF_MATCHERS[leaf](doc, query[leaf])
    if 'term' in query:
        return _term_matches(doc, query['term'])
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


def _top_hit(doc: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    doc_id = doc.get(_DOC_ID_KEY)
    visible = {
        k: v
        for k, v in doc.items()
        if k not in (_DOC_ID_KEY, _REVERSE_NESTED_PARENT_KEY, _PARENT_DOC_KEY)
    }
    hit: dict[str, Any] = {'_id': doc_id}
    if spec.get('docvalue_fields'):
        hit['fields'] = {f: _values(visible, f) for f in spec['docvalue_fields']}
    if spec.get('_source', True) is not False:
        hit['_source'] = copy.deepcopy(visible)
    return hit


def _nested_clauses_with_inner_hits(query: Any) -> list[dict[str, Any]]:
    """Every ``nested`` clause under ``query`` that asks for ``inner_hits``."""
    found: list[dict[str, Any]] = []
    if isinstance(query, dict):
        nested = query.get('nested')
        if isinstance(nested, dict) and 'inner_hits' in nested:
            found.append(nested)
        for value in query.values():
            found.extend(_nested_clauses_with_inner_hits(value))
    elif isinstance(query, list):
        for item in query:
            found.extend(_nested_clauses_with_inner_hits(item))
    return found


def _sorted_by(
    docs: list[dict[str, Any]], sort: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """``docs`` ordered by an OpenSearch ``sort`` list (the first key only;
    ``missing: _last`` / ``_first`` honoured, ``_last`` by default). Without a
    sort the order is the fake's insertion order, which is what an unsorted
    ``top_hits`` would not guarantee."""
    if not sort:
        return docs
    ((field, opts),) = sort[0].items()
    descending = (opts.get('order', 'asc') if isinstance(opts, dict) else opts) == 'desc'
    missing_first = isinstance(opts, dict) and opts.get('missing') == '_first'
    present = [d for d in docs if _values(d, field)]
    absent = [d for d in docs if not _values(d, field)]
    present.sort(key=lambda d: _values(d, field)[0], reverse=descending)
    return [*absent, *present] if missing_first else [*present, *absent]


def _aggregate(docs: list[dict[str, Any]], aggs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in aggs.items():
        if 'cardinality' in spec:
            field = spec['cardinality']['field']
            distinct = {v for d in docs for v in _values(d, field)}
            out[name] = {'value': len(distinct)}
            continue
        if 'reverse_nested' in spec:
            # The distinct PARENT docs the current group's nested elements
            # came from, not the elements themselves -- an item with 2 boxes
            # matching the same terms bucket counts once. Sub-aggs run over
            # those parents.
            parents: dict[int, dict[str, Any]] = {}
            for d in docs:
                if _PARENT_DOC_KEY in d:
                    parents.setdefault(d[_REVERSE_NESTED_PARENT_KEY], d[_PARENT_DOC_KEY])
            parent_docs = list(parents.values()) if parents else docs
            rn_bucket: dict[str, Any] = {'doc_count': len(parent_docs)}
            if spec.get('aggs'):
                rn_bucket.update(_aggregate(parent_docs, spec['aggs']))
            out[name] = rn_bucket
            continue
        if 'top_hits' in spec:
            size = spec['top_hits'].get('size', 3)
            ordered = _sorted_by(docs, spec['top_hits'].get('sort'))
            out[name] = {'hits': {'hits': [_top_hit(d, spec['top_hits']) for d in ordered[:size]]}}
            continue
        if 'composite' in spec:
            csize = spec['composite'].get('size', 10)
            ((src_name, src_spec),) = spec['composite']['sources'][0].items()
            field = src_spec['terms']['field']
            composite_groups: dict[Any, list[dict[str, Any]]] = {}
            for doc in docs:
                for v in _values(doc, field):
                    composite_groups.setdefault(v, []).append(doc)
            sorted_keys = sorted(composite_groups.keys())
            after = spec['composite'].get('after')
            if after is not None:
                after_val = after[src_name]
                sorted_keys = [k for k in sorted_keys if k > after_val]
            page_keys = sorted_keys[:csize]
            composite_buckets = []
            for key in page_keys:
                members = composite_groups[key]
                composite_bucket = {'key': {src_name: key}, 'doc_count': len(members)}
                if spec.get('aggs'):
                    composite_bucket.update(_aggregate(members, spec['aggs']))
                composite_buckets.append(composite_bucket)
            result: dict[str, Any] = {'buckets': composite_buckets}
            if composite_buckets:
                result['after_key'] = {src_name: page_keys[-1]}
            out[name] = result
            continue
        if 'filter' in spec:
            kept = [d for d in docs if matches(d, spec['filter'])]
            filter_bucket: dict[str, Any] = {'doc_count': len(kept)}
            if spec.get('aggs'):
                filter_bucket.update(_aggregate(kept, spec['aggs']))
            out[name] = filter_bucket
            continue
        if 'nested' in spec:
            path = spec['nested']['path']
            flattened = [el for doc in docs for el in _nested_elements(doc, path)]
            nested_bucket: dict[str, Any] = {'doc_count': len(flattened)}
            if spec.get('aggs'):
                nested_bucket.update(_aggregate(flattened, spec['aggs']))
            out[name] = nested_bucket
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
        self.mget_calls = 0
        self.max_inner_result_window = 100
        self.settings_puts: list[dict[str, Any]] = []
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
        for clause in _nested_clauses_with_inner_hits(body.get('query')):
            self._attach_inner_hits(resp['hits']['hits'], pool, clause)
        if body.get('aggs'):
            resp['aggregations'] = _aggregate(
                [{**doc, _DOC_ID_KEY: doc_id} for doc_id, doc in pool], body['aggs']
            )
        if scroll is not None:
            resp['_scroll_id'] = 'fake-scroll'
        return resp

    def _attach_inner_hits(
        self,
        hits: list[dict[str, Any]],
        pool: list[tuple[str, dict[str, Any]]],
        clause: dict[str, Any],
    ) -> None:
        spec = clause['inner_hits']
        size = spec.get('size', 3)
        if size > self.max_inner_result_window:
            raise ValueError(
                f'400: inner_hits size {size} exceeds index.max_inner_result_window '
                f'{self.max_inner_result_window}'
            )
        by_id = dict(pool)
        for hit in hits:
            doc = by_id[hit['_id']]
            matched = [
                {'_nested': {'field': clause['path'], 'offset': i}}
                for i, el in enumerate(_nested_elements(doc, clause['path']))
                if matches(el, clause['query'])
            ]
            hit.setdefault('inner_hits', {})[spec.get('name', clause['path'])] = {
                'hits': {'total': {'value': len(matched)}, 'hits': matched[:size]}
            }

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
        self.mget_calls += 1
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
            if op == 'create':
                if doc_id in self.docs(idx):
                    errors = True
                    items.append(
                        {op: {'_id': doc_id, 'status': 409, 'error': {'type': 'version_conflict'}}}
                    )
                    continue
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

    async def get_settings(self, *, index: str, **_kw: Any) -> dict[str, Any]:
        window = str(self.parent.max_inner_result_window)
        return {index: {'settings': {'index': {'max_inner_result_window': window}}}}

    async def put_settings(self, *, index: str, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        self.parent.settings_puts.append({'index': index, 'body': body})
        flat = body.get('index', body)
        value = flat.get('max_inner_result_window', body.get('index.max_inner_result_window'))
        if value is not None:
            self.parent.max_inner_result_window = int(value)
        return {'acknowledged': True}
