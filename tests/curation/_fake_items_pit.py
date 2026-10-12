"""An in-memory items index with the calls the cluster pass makes.

Serves the point-in-time sliced search the parallel fetch uses, the scroll
``assign_only_residuals`` uses, ``count``, ``bulk`` (the guarded painless update is
applied by reading its ``params``, which are the whole effect of the script on a
document no human owns) and ``indices.refresh``. Documents keep their insertion order,
which is the ``_shard_doc`` order of a one-shard index.
"""

# ruff: noqa: ARG002 - accepts the real client's keyword arguments and ignores most of them
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np


class FakeItemsClient:
    def __init__(self, ids: list[str], vectors: np.ndarray, field: str = 'pe_embedding') -> None:
        self.field = field
        self.ids = list(ids)
        self.docs = [{field: row.tolist()} for row in vectors]
        # id -> (cluster_id, distance): what the guarded update stored.
        self.written: dict[str, tuple[int, float | None]] = {}
        self.bulk_calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.bulk_delay_s = 0.0
        self._scrolls: dict[str, int] = {}
        self.indices = _Indices()

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        return {'count': len(self.ids)}

    async def create_pit(self, *, index: str, keep_alive: str) -> dict[str, Any]:
        return {'pit_id': 'pit-1'}

    async def delete_pit(self, *, body: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _hit(self, pos: int) -> dict[str, Any]:
        return {'_id': self.ids[pos], '_source': self.docs[pos], 'sort': [pos]}

    async def search(self, *, body: dict[str, Any], index: str | None = None, scroll: Any = None):
        size = body['size']
        if 'pit' in body:
            sl = body.get('slice')
            after = body.get('search_after', [-1])[0]
            members = [
                p
                for p in range(len(self.ids))
                if p > after and (sl is None or p % sl['max'] == sl['id'])
            ]
            return {'hits': {'hits': [self._hit(p) for p in members[:size]]}}
        self._scrolls['s1'] = size
        return {
            '_scroll_id': 's1',
            'hits': {'hits': [self._hit(p) for p in range(size) if p < len(self.ids)]},
        }

    async def scroll(self, *, scroll_id: str, scroll: str) -> dict[str, Any]:
        size = self._scrolls[scroll_id]
        start = self._scrolls.get('pos', 0) + size
        self._scrolls['pos'] = start
        hits = [self._hit(p) for p in range(start, min(start + size, len(self.ids)))]
        return {'_scroll_id': scroll_id, 'hits': {'hits': hits}}

    async def clear_scroll(self, *, scroll_id: str) -> dict[str, Any]:
        return {}

    async def bulk(self, *, body: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
        self.bulk_calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.bulk_delay_s)
            items = []
            for i in range(0, len(body), 2):
                doc_id = body[i]['update']['_id']
                params = body[i + 1]['script']['params']
                self.written[doc_id] = (params['cid'], params['dist'])
                items.append({'update': {'_id': doc_id, 'status': 200}})
            return {'errors': False, 'items': items}
        finally:
            self.in_flight -= 1


class _Indices:
    async def refresh(self, *, index: str) -> dict[str, Any]:
        return {}
