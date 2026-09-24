"""Cluster geometry for every clustered item, written after each clustering run.

DQ-M3. The residual run writes ``cluster_distance`` for the candidate
clusters it assigns, but class clusters (``cluster_id == class_id``) are
filled by labelling, so most of their members had no distance — and an
item the VLM moved into a class cluster kept a distance to the candidate
centroid it came from. With no distance the wire's ``cluster_is_core``
is null, so the cluster view's core/outlier cut line had nothing to cut.

This pass runs after the residual stage of the auto-label pipeline:

1. For every cluster id ``>= 0`` it computes the member-mean centroid of
   the members' item embeddings (unit-normalized, the same centroid
   :mod:`centroid_distance` and ``order=outliers`` use).
2. It writes, per member, ``cluster_distance_cluster_id`` (the cluster
   the geometry was measured against) and — for class clusters —
   ``cluster_distance`` (``1 - cos`` to that centroid). A candidate
   member keeps the distance its clustering method wrote.
3. It writes ``cluster_nearest_id``: the cluster whose centroid is
   nearest the item among all clusters. ``GET /clusters`` serves a
   cluster's ``purity`` as the share of its measured members whose
   nearest centroid is their own (DQ-M2) — a signal independent of the
   labels that placed them there.

Every write is guarded on the item still being in the cluster it was
measured for, so an item moved while the pass ran is left alone; the
wire then treats a distance whose ``cluster_distance_cluster_id`` differs
from the item's ``cluster_id`` as stale (null).

Memory stays bounded by the largest single cluster: centroids are built
one cluster at a time, and members are fetched again per cluster for the
write.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import get_curation_config
from src.config.curation import ITEM_EMBEDDING_FIELD
from src.core.logging import get_logger
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.export_support import scroll_hits


logger = get_logger(__name__)

DISTANCE_REF_FIELD = 'cluster_distance_cluster_id'
# The cluster whose centroid is nearest the item (DQ-M2 cluster purity).
NEAREST_FIELD = 'cluster_nearest_id'
_MAX_CLUSTER_BUCKETS = 20000
_BULK_CHUNK = 2000

_WRITE_SCRIPT = (
    "if (ctx._source['cluster_id'] == null || ctx._source['cluster_id'] != params.cid)"
    " { ctx.op = 'noop'; return; }"
    ' for (entry in params.fields.entrySet())'
    ' { ctx._source[entry.getKey()] = entry.getValue(); }'
)


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def unit_centroid(embeddings: np.ndarray) -> np.ndarray | None:
    """Unit-normalized mean of the unit-normalized rows, or ``None`` when
    the mean vanishes (no rows, or rows that cancel out)."""
    if len(embeddings) == 0:
        return None
    centroid = _normalize(np.asarray(embeddings, dtype=np.float32)).mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    return None if norm == 0.0 else centroid / norm


def centroid_distances(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """``1 - cos`` of every row to ``centroid``, clipped to ``[0, 2]``."""
    x = _normalize(np.asarray(embeddings, dtype=np.float32))
    return np.clip(1.0 - x @ centroid, 0.0, 2.0)


def _members_query(cluster_id: int) -> dict[str, Any]:
    return {
        'bool': {
            'filter': [
                {'term': {'cluster_id': cluster_id}},
                {'exists': {'field': ITEM_EMBEDDING_FIELD}},
            ],
            'must_not': [{'term': {'class_excluded': True}}],
        }
    }


async def _cluster_ids(client: Any, index: str) -> list[int]:
    body = {
        'size': 0,
        'query': {
            'bool': {
                'filter': [
                    {'range': {'cluster_id': {'gte': 0}}},
                    {'exists': {'field': ITEM_EMBEDDING_FIELD}},
                ],
                'must_not': [{'term': {'class_excluded': True}}],
            }
        },
        'aggs': {'ids': {'terms': {'field': 'cluster_id', 'size': _MAX_CLUSTER_BUCKETS}}},
    }
    resp = await client.search(index=index, body=body)
    buckets = ((resp.get('aggregations') or {}).get('ids') or {}).get('buckets') or []
    return sorted(int(b['key']) for b in buckets)


async def _members(client: Any, index: str, cluster_id: int) -> tuple[list[str], np.ndarray]:
    hits = await scroll_hits(
        client,
        index=index,
        query=_members_query(cluster_id),
        source=['crop_id', ITEM_EMBEDDING_FIELD],
    )
    ids: list[str] = []
    vecs: list[list[float]] = []
    for h in hits:
        emb = (h.get('_source') or {}).get(ITEM_EMBEDDING_FIELD)
        if emb:
            ids.append(str(h.get('_id')))
            vecs.append(emb)
    return ids, np.asarray(vecs, dtype=np.float32)


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


async def _write(client: Any, actions: list[Any]) -> int:
    """Bulk-write ``actions`` in chunks; returns the number of failed items."""
    failed = 0
    for start in range(0, len(actions), 2 * _BULK_CHUNK):
        resp = await client.bulk(body=actions[start : start + 2 * _BULK_CHUNK], refresh=False)
        if resp.get('errors'):
            failed += sum(
                1
                for it in resp.get('items') or []
                if (next(iter(it.values()), {}) or {}).get('error')
            )
    return failed


async def write_cluster_geometry(client: Any, *, index: str | None = None) -> dict[str, Any]:
    """Measure every clustered item against the cluster centroids and write
    the geometry fields. Returns a stage summary for the pipeline.

    Two passes, one cluster at a time: the first builds every cluster's
    centroid, the second writes each member's own distance (class
    clusters), measured cluster and nearest centroid.
    """
    items_index = index or get_curation_config().items_index
    cluster_ids = await _cluster_ids(client, items_index)
    centroid_ids: list[int] = []
    centroids: list[np.ndarray] = []
    for cid in cluster_ids:
        _ids, x = await _members(client, items_index, cid)
        centroid = unit_centroid(x)
        if centroid is not None:
            centroid_ids.append(cid)
            centroids.append(centroid)
    if not centroids:
        return {'status': 'success', 'n_clusters': 0, 'n_items': 0, 'n_class_distances': 0}
    matrix = np.stack(centroids)
    column = {cid: k for k, cid in enumerate(centroid_ids)}

    n_items = 0
    n_class_distances = 0
    n_nearest_elsewhere = 0
    n_write_errors = 0
    for cid in centroid_ids:
        ids, x = await _members(client, items_index, cid)
        if not ids:
            continue
        sims = _normalize(x) @ matrix.T
        own = np.clip(1.0 - sims[:, column[cid]], 0.0, 2.0)
        nearest = [centroid_ids[k] for k in np.argmax(sims, axis=1).tolist()]
        is_class_cluster = cid < RESIDUAL_CLUSTER_ID_OFFSET
        actions: list[Any] = []
        for crop_id, dist, near in zip(ids, own.tolist(), nearest, strict=True):
            fields: dict[str, Any] = {DISTANCE_REF_FIELD: cid, NEAREST_FIELD: near}
            if is_class_cluster:
                fields['cluster_distance'] = float(dist)
            actions.extend(_update(items_index, crop_id, cid, fields))
        n_items += len(ids)
        n_class_distances += len(ids) if is_class_cluster else 0
        n_nearest_elsewhere += sum(1 for near in nearest if near != cid)
        n_write_errors += await _write(client, actions)
    if n_items:
        await client.indices.refresh(index=items_index)
    if n_write_errors:
        logger.warning('cluster_geometry_write_errors', n_errors=n_write_errors)
    return {
        'status': 'success',
        'n_clusters': len(centroid_ids),
        'n_items': n_items,
        'n_class_distances': n_class_distances,
        'n_nearest_elsewhere': n_nearest_elsewhere,
        'n_write_errors': n_write_errors,
    }


async def cluster_geometry_stage(client: Any) -> dict[str, Any]:
    """:func:`write_cluster_geometry` as a pipeline sub-stage: a failure is
    reported in the stage summary (and logged), never raised — the
    clustering it annotates already succeeded."""
    try:
        return await write_cluster_geometry(client)
    except Exception as exc:
        logger.error('cluster_geometry_failed', error=str(exc))
        return {'status': 'error', 'error': str(exc)}


__all__ = [
    'DISTANCE_REF_FIELD',
    'NEAREST_FIELD',
    'centroid_distances',
    'cluster_geometry_stage',
    'unit_centroid',
    'write_cluster_geometry',
]
