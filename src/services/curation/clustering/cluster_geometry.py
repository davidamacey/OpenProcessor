"""The lazy write-back of ``order=core_first`` centroid distances.

``crop_orders`` computes each member's distance to its cluster's member-mean centroid on
first use and stores it, with the cluster it was measured against
(``cluster_distance_cluster_id``), so the next page reads it from the index. Every write is
guarded on the item still being in the cluster it was measured for, so an item moved while the
backfill ran is left alone; the wire treats a distance whose reference differs from the item's
cluster as stale. The index-wide geometry pass that used to write these for every cluster is
gone with the legacy cluster pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any


_DISTANCE_REF_FIELD = 'cluster_distance_cluster_id'
_BULK_CHUNK = 2000
_BACKFILL_PAUSE_S = 0.05

_WRITE_SCRIPT = (
    "if (ctx._source['cluster_id'] == null || ctx._source['cluster_id'] != params.cid)"
    " { ctx.op = 'noop'; return; }"
    ' for (entry in params.fields.entrySet())'
    ' { ctx._source[entry.getKey()] = entry.getValue(); }'
)


def _update(index: str, crop_id: str, cluster_id: int, fields: dict[str, Any]) -> list[Any]:
    return [
        {'update': {'_index': index, '_id': crop_id}},
        {
            'script': {
                'lang': 'painless',
                'source': _WRITE_SCRIPT,
                'params': {'cid': cluster_id, 'fields': fields},
            }
        },
    ]


def _bulk_item_failed(item: dict[str, Any]) -> bool:
    """True when a bulk response item carries an ``error`` for its action."""
    action: dict[str, Any] = next(iter(item.values()), {}) or {}
    return bool(action.get('error'))


async def _write(client: Any, actions: list[Any]) -> int:
    """Bulk-write ``actions`` in chunks; returns the number of failed items."""
    failed = 0
    for start in range(0, len(actions), 2 * _BULK_CHUNK):
        resp = await client.bulk(body=actions[start : start + 2 * _BULK_CHUNK], refresh=False)
        if resp.get('errors'):
            failed += sum(1 for it in resp.get('items') or [] if _bulk_item_failed(it))
    return failed


async def write_member_distances(
    client: Any, *, index: str, cluster_id: int, distances: dict[str, float]
) -> int:
    """Persist ``cluster_distance`` (+ its reference cluster) for ``distances``
    (crop id -> distance to the member-mean centroid), guarded on each item
    still being in ``cluster_id``. Throttled between bulk chunks so a lazy
    backfill never floods a small node. Returns the failed-item count."""
    items = list(distances.items())
    failed = 0
    for start in range(0, len(items), _BULK_CHUNK):
        actions: list[Any] = []
        for crop_id, dist in items[start : start + _BULK_CHUNK]:
            fields = {_DISTANCE_REF_FIELD: cluster_id, 'cluster_distance': float(dist)}
            actions.extend(_update(index, crop_id, cluster_id, fields))
        failed += await _write(client, actions)
        await asyncio.sleep(_BACKFILL_PAUSE_S)
    return failed


__all__ = ['write_member_distances']
