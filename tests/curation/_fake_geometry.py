"""An in-memory items index for the cluster geometry pass.

Evaluates queries with the shared matcher, serves scroll (and, when ``pit=True``, the
point-in-time sliced search the fast fetch uses; ``sort`` is the document position, the
``_shard_doc`` of a one-shard index) and applies the pass's guarded script updates.
"""

# ruff: noqa: ARG002 - accepts the real client's keyword arguments and ignores most of them
from __future__ import annotations

import asyncio
import copy
from typing import Any
from unittest.mock import AsyncMock

from curation.query_fakes import matches


class GeometryFake:
    def __init__(self, docs: dict[str, dict[str, Any]], *, pit: bool = False) -> None:
        self.docs = copy.deepcopy(docs)
        self.pit = pit
        self.indices = AsyncMock()
        self.bulk_calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.bulk_delay_s = 0.0
        self.docs_written = 0
        self.search_calls = 0
        self._order = list(self.docs)

    async def create_pit(self, *, index: str, keep_alive: str) -> dict[str, Any]:
        if not self.pit:
            raise RuntimeError('no point in time')
        return {'pit_id': 'p'}

    async def delete_pit(self, *, body: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _source(self, doc: dict[str, Any], source: Any) -> dict[str, Any]:
        out = copy.deepcopy(doc)
        if isinstance(source, list):
            out = {k: v for k, v in out.items() if k in source}
        return out

    async def search(
        self, *, body: dict[str, Any], index: str | None = None, scroll: str | None = None
    ) -> Any:
        self.search_calls += 1
        pool = [
            (p, i) for p, i in enumerate(self._order) if matches(self.docs[i], body.get('query'))
        ]
        if body.get('aggs'):
            keys = sorted({self.docs[i]['cluster_id'] for _p, i in pool})
            return {'aggregations': {'ids': {'buckets': [{'key': k} for k in keys]}}}
        if 'pit' in body:
            sl = body.get('slice')
            after = body.get('search_after', [-1])[0]
            pool = [
                (p, i) for p, i in pool if p > after and (sl is None or p % sl['max'] == sl['id'])
            ][: body['size']]
            hits = [
                {'_id': i, '_source': self._source(self.docs[i], body.get('_source')), 'sort': [p]}
                for p, i in pool
            ]
            return {'hits': {'hits': hits}}
        hits = [
            {'_id': i, '_source': self._source(self.docs[i], body.get('_source'))} for _p, i in pool
        ]
        return {'_scroll_id': 's', 'hits': {'hits': hits}}

    async def scroll(self, **_kw: Any) -> Any:
        return {'_scroll_id': 's', 'hits': {'hits': []}}

    async def clear_scroll(self, **_kw: Any) -> None:
        return None

    async def bulk(self, *, body: list[dict[str, Any]], **_kw: Any) -> Any:
        self.bulk_calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.bulk_delay_s)
            for action, payload in zip(body[::2], body[1::2], strict=True):
                doc = self.docs[action['update']['_id']]
                params = payload['script']['params']
                if doc.get('cluster_id') == params['cid']:
                    doc.update(params['fields'])
                    self.docs_written += 1
            return {'errors': False, 'items': []}
        finally:
            self.in_flight -= 1
