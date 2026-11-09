"""The OpenSearch side of the re-index job (embeddings plan 5.5): the ``ReindexIndex`` adapter.

Reads are a point-in-time (PIT) with ``search_after`` over ``_id``; writes are scripted partial
updates that skip a document whose geometry moved since the page was read (the bbox guard); the
exists count, the random sample and the clear are plain search, count and update-by-query calls.
Docs: https://docs.opensearch.org/latest/search-plugins/searching-data/paginate/ (PIT,
``search_after``), the opensearch-py PIT guide and
https://docs.opensearch.org/latest/api-reference/document-apis/update-by-query/.

A box document id is ``<item id>#<box id>``: a page never splits one item's boxes, so the cursor
(``after``) is always an item id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.config.region_fields import get_region_fields
from src.domain.embedding_spaces import Target
from src.domain.embedding_state import EMBEDDED
from src.domain.region_boxes import read_boxes
from src.services.curation.region_box_embeddings import EMBEDDED_BOX_STATES, embeddable
from src.services.embedding_spaces.reindex import (
    DocRef,
    Outcome,
    ReindexIndex,
    ReindexMode,
    VectorWrite,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.db.repos import embedding_spaces as repo

BOX_SEP = '#'
PIT_KEEP_ALIVE = '2m'
CLEAR_TIMEOUT_S = 3600

_MOVED = """
if (params.bbox != null) {
  def have = ctx._source.bbox_norm;
  if (have == null || have.size() != params.bbox.size()) { ctx.op = 'none'; return; }
  for (int i = 0; i < have.size(); i++) {
    if (Math.abs(((Number) have.get(i)).doubleValue() - ((Number) params.bbox.get(i)).doubleValue()) > 1e-6) {
      ctx.op = 'none'; return;
    }
  }
}
"""

_WRITE_FIELD = (
    _MOVED
    + """
ctx._source[params.field] = params.vector;
if (params.state_field != null) { ctx._source[params.state_field] = params.state; }
"""
)

_WRITE_BOX = """
def live = null;
if (ctx._source[params.boxes] != null) {
  for (b in ctx._source[params.boxes]) { if (b.box_id == params.box_id) { live = b; } }
}
if (live == null) { ctx.op = 'none'; return; }
def have = live.bbox_norm;
if (have == null || have.size() != params.bbox.size()) { ctx.op = 'none'; return; }
for (int i = 0; i < have.size(); i++) {
  if (Math.abs(((Number) have.get(i)).doubleValue() - ((Number) params.bbox.get(i)).doubleValue()) > 1e-6) {
    ctx.op = 'none'; return;
  }
}
if (ctx._source[params.path] == null) { ctx._source[params.path] = new ArrayList(); }
ctx._source[params.path].removeIf(e -> e.box_id == params.box_id);
ctx._source[params.path].add(['box_id': params.box_id, 'bbox_norm': params.bbox, params.leaf: params.vector]);
"""

_CLEAR = """
ctx._source.remove(params.field);
if (params.state_field != null) { ctx._source.remove(params.state_field); }
"""


def split_doc_id(doc_id: str) -> tuple[str, str | None]:
    """``(document id, box id)``: the box id is ``None`` for an item or a frame."""
    head, sep, tail = doc_id.rpartition(BOX_SEP)
    return (head, tail) if sep else (doc_id, None)


def _leaf(space: repo.SpaceRow) -> str:
    assert space.nested_path is not None
    return space.vector_field.removeprefix(f'{space.nested_path}.')


def _exists(space: repo.SpaceRow) -> dict[str, Any]:
    clause = {'exists': {'field': space.vector_field}}
    if space.nested_path is None:
        return clause
    return {'nested': {'path': space.nested_path, 'query': clause}}


class OpenSearchReindexIndex(ReindexIndex):
    """``client`` is an ``AsyncOpenSearch`` from ``make_script_opensearch`` (project guarded)."""

    def __init__(self, client: Any) -> None:
        self._os = client

    async def aclose(self) -> None:
        await self._os.close()

    @staticmethod
    def _index(space: repo.SpaceRow) -> str:
        cfg = get_curation_config()
        return cfg.images_index if space.index_role == 'images' else cfg.items_index

    @staticmethod
    def _scope(scope: Mapping[str, Any]) -> None:
        if scope:
            msg = f'the re-index scope {sorted(scope)} is not supported'
            raise ValueError(msg)

    def _source_query(self, space: repo.SpaceRow, mode: ReindexMode) -> dict[str, Any]:
        """The documents a run visits; a backfill keeps only those with no vector yet."""
        F = get_region_fields()
        if space.target is Target.BOX:
            must: list[dict[str, Any]] = [
                {
                    'nested': {
                        'path': F.boxes,
                        'query': {'terms': {f'{F.boxes}.state': list(EMBEDDED_BOX_STATES)}},
                    }
                }
            ]
            return {'bool': {'filter': must}}
        filters: list[dict[str, Any]] = (
            [{'exists': {'field': 'bbox_norm'}}] if space.target is Target.ITEM else []
        )
        query: dict[str, Any] = {'bool': {'filter': filters}}
        if mode is ReindexMode.BACKFILL:
            query['bool']['must_not'] = [_exists(space)]
        return query

    async def put_mapping(self, index_role: str, _properties: dict[str, Any]) -> None:
        cfg = get_curation_config()
        index = cfg.images_index if index_role == 'images' else cfg.items_index
        await self._os.indices.put_mapping(index=index, body={'properties': _properties})

    async def count(self, space: repo.SpaceRow, mode: ReindexMode, scope: Mapping[str, Any]) -> int:
        self._scope(scope)
        index = self._index(space)
        query = self._source_query(space, mode)
        if space.target is not Target.BOX:
            return int((await self._os.count(index=index, body={'query': query}))['count'])
        F = get_region_fields()
        resp = await self._os.search(
            index=index,
            body={
                'size': 0,
                'query': query,
                'aggs': {
                    'boxes': {
                        'nested': {'path': F.boxes},
                        'aggs': {
                            'live': {
                                'filter': {
                                    'terms': {f'{F.boxes}.state': list(EMBEDDED_BOX_STATES)}
                                },
                                'aggs': {'n': {'value_count': {'field': f'{F.boxes}.box_id'}}},
                            }
                        },
                    }
                },
            },
        )
        return int(resp['aggregations']['boxes']['live']['n']['value'])

    async def page(
        self,
        space: repo.SpaceRow,
        mode: ReindexMode,
        scope: Mapping[str, Any],
        after: str | None,
        size: int,
    ) -> list[DocRef]:
        self._scope(scope)
        F = get_region_fields()
        includes = [F.boxes] if space.target is Target.BOX else ['bbox_norm']
        if space.target is Target.BOX and mode is ReindexMode.BACKFILL:
            includes.append(f'{space.nested_path}.box_id')
        pit = (await self._os.create_pit(index=self._index(space), keep_alive=PIT_KEEP_ALIVE))[
            'pit_id'
        ]
        try:
            body: dict[str, Any] = {
                'size': size,
                'pit': {'id': pit, 'keep_alive': PIT_KEEP_ALIVE},
                'sort': [{'_id': 'asc'}],
                '_source': includes,
                'query': self._source_query(space, mode),
            }
            if after is not None:
                body['search_after'] = [after]
            hits = (await self._os.search(body=body))['hits']['hits']
        finally:
            await self._os.delete_pit(body={'pit_id': [pit]})
        return [ref for hit in hits for ref in self._refs(space, mode, hit)]

    @staticmethod
    def _refs(space: repo.SpaceRow, mode: ReindexMode, hit: dict[str, Any]) -> list[DocRef]:
        src = hit.get('_source') or {}
        if space.target is Target.FRAME:
            return [DocRef(hit['_id'])]
        if space.target is Target.ITEM:
            box = src.get('bbox_norm')
            return [DocRef(hit['_id'], tuple(float(v) for v in box) if box else None)]
        held = {
            e.get('box_id') for e in src.get(space.nested_path or '') or [] if isinstance(e, dict)
        }
        return [
            DocRef(f'{hit["_id"]}{BOX_SEP}{b.box_id}', tuple(b.bbox_norm))
            for b in embeddable(read_boxes(src))
            if mode is not ReindexMode.BACKFILL or b.box_id not in held
        ]

    async def write(
        self, space: repo.SpaceRow, writes: Sequence[VectorWrite]
    ) -> dict[str, Outcome]:
        index = self._index(space)
        body: list[dict[str, Any]] = []
        for w in writes:
            item_id, box_id = split_doc_id(w.doc_id)
            body.append({'update': {'_index': index, '_id': item_id, 'retry_on_conflict': 3}})
            body.append({'script': {'lang': 'painless', **self._script(space, w, box_id)}})
        resp = await self._os.bulk(body=body, refresh=False)
        out: dict[str, Outcome] = {}
        for w, item in zip(writes, resp['items'], strict=True):
            result = item['update']
            if result.get('status') == 404:
                out[w.doc_id] = 'missing'
            elif result.get('status', 200) >= 300:
                msg = f're-index write failed for {w.doc_id}: {str(result)[:300]}'
                raise RuntimeError(msg)
            else:
                out[w.doc_id] = 'moved' if result.get('result') == 'noop' else 'written'
        return out

    @staticmethod
    def _script(space: repo.SpaceRow, w: VectorWrite, box_id: str | None) -> dict[str, Any]:
        vector = [float(v) for v in w.vector]
        bbox = list(w.expected_bbox) if w.expected_bbox is not None else None
        if box_id is not None:
            return {
                'source': _WRITE_BOX,
                'params': {
                    'boxes': get_region_fields().boxes,
                    'path': space.nested_path,
                    'leaf': _leaf(space),
                    'box_id': box_id,
                    'bbox': bbox,
                    'vector': vector,
                },
            }
        return {
            'source': _WRITE_FIELD,
            'params': {
                'field': space.vector_field,
                'state_field': space.state_field,
                'state': EMBEDDED,
                'bbox': bbox,
                'vector': vector,
            },
        }

    async def mark(self, space: repo.SpaceRow, doc_ids: Sequence[str], state: str) -> None:
        if space.state_field is None:
            return
        index = self._index(space)
        body: list[dict[str, Any]] = []
        for doc_id in dict.fromkeys(split_doc_id(d)[0] for d in doc_ids):
            body.append({'update': {'_index': index, '_id': doc_id, 'retry_on_conflict': 3}})
            body.append({'doc': {space.state_field: state}})
        if body:
            await self._os.bulk(body=body, refresh=False)

    async def count_with_vector(self, space: repo.SpaceRow) -> int:
        resp = await self._os.count(index=self._index(space), body={'query': _exists(space)})
        return int(resp['count'])

    async def sample(self, space: repo.SpaceRow, size: int) -> list[DocRef]:
        body: dict[str, Any] = {
            'size': size,
            '_source': ['bbox_norm'],
            'query': {
                'function_score': {
                    'query': {'bool': {'filter': [_exists(space)]}},
                    'random_score': {},
                }
            },
        }
        if space.nested_path is not None:
            body['_source'] = False
            body['query']['function_score']['query']['bool']['filter'] = [
                {
                    'nested': {
                        'path': space.nested_path,
                        'query': {'exists': {'field': space.vector_field}},
                        'inner_hits': {
                            'size': 1,
                            '_source': [
                                f'{space.nested_path}.box_id',
                                f'{space.nested_path}.bbox_norm',
                            ],
                        },
                    }
                }
            ]
        hits = (await self._os.search(index=self._index(space), body=body))['hits']['hits']
        refs: list[DocRef] = []
        for hit in hits:
            if space.nested_path is None:
                box = (hit.get('_source') or {}).get('bbox_norm')
                refs.append(DocRef(hit['_id'], tuple(float(v) for v in box) if box else None))
                continue
            inner = hit['inner_hits'][space.nested_path]['hits']['hits']
            if inner:
                entry = inner[0]['_source']
                refs.append(
                    DocRef(
                        f'{hit["_id"]}{BOX_SEP}{entry["box_id"]}',
                        tuple(float(v) for v in entry['bbox_norm']),
                    )
                )
        return refs

    async def read_vectors(
        self, space: repo.SpaceRow, doc_ids: Sequence[str]
    ) -> dict[str, Sequence[float]]:
        ids = list(dict.fromkeys(split_doc_id(d)[0] for d in doc_ids))
        path = space.nested_path
        includes = [f'{path}.box_id', space.vector_field] if path else [space.vector_field]
        resp = await self._os.mget(
            index=self._index(space), body={'ids': ids}, _source_includes=includes
        )
        found: dict[str, Sequence[float]] = {}
        for doc in resp['docs']:
            if not doc.get('found'):
                continue
            src = doc.get('_source') or {}
            if path is None:
                if src.get(space.vector_field) is not None:
                    found[doc['_id']] = src[space.vector_field]
                continue
            for entry in src.get(path) or []:
                vector = entry.get(_leaf(space))
                if vector is not None:
                    found[f'{doc["_id"]}{BOX_SEP}{entry["box_id"]}'] = vector
        return found

    async def clear(self, space: repo.SpaceRow) -> None:
        field = space.nested_path or space.vector_field
        await self._os.update_by_query(
            index=self._index(space),
            body={
                'query': _exists(space),
                'script': {
                    'lang': 'painless',
                    'source': _CLEAR,
                    'params': {'field': field, 'state_field': space.state_field},
                },
            },
            conflicts='proceed',
            refresh=True,
            wait_for_completion=True,
            request_timeout=CLEAR_TIMEOUT_S,
        )


__all__ = ['BOX_SEP', 'OpenSearchReindexIndex', 'split_doc_id']
