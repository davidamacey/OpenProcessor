"""Cluster geometry for every clustered item, written after each clustering run.

The residual run writes ``cluster_distance`` for the candidate
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
   nearest centroid is their own — a signal independent of the
   labels that placed them there.

Every write is guarded on the item still being in the cluster it was
measured for, so an item moved while the pass ran is left alone; the
wire then treats a distance whose ``cluster_distance_cluster_id`` differs
from the item's ``cluster_id`` as stale (null).

Every clustered vector is read once, by point-in-time slices at once, and the
centroids, distances and nearest clusters are computed per cluster from that
matrix (rows in index order, so each sum is the one a per-cluster read would
make). Memory is one float32 matrix of the clustered items. An item that already
stores exactly the fields the pass would write is not written again; the rest go
out several bulk requests at a time.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from src.config import get_curation_config
from src.config.curation import ITEM_EMBEDDING_FIELD
from src.core.logging import get_logger
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.clustering.assignment_writer import BULK_CHUNK, AssignmentWriter
from src.services.curation.clustering.page_rows import hits_to_rows
from src.services.curation.clustering.pit_pages import iter_pages
from src.services.curation.embedding_state import embedded_clause


logger = get_logger(__name__)

DISTANCE_REF_FIELD = 'cluster_distance_cluster_id'
# The cluster whose centroid is nearest the item (cluster purity).
NEAREST_FIELD = 'cluster_nearest_id'
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


def _clustered_query() -> dict[str, Any]:
    return {
        'bool': {
            'filter': [
                {'range': {'cluster_id': {'gte': 0}}},
                embedded_clause(ITEM_EMBEDDING_FIELD),
            ],
            'must_not': [{'term': {'class_excluded': True}}],
        }
    }


class _Clustered:
    """Every clustered item with a vector: ids, cluster ids, stored geometry and the
    float32 matrix, rows in index order (the order a per-cluster scroll returns them)."""

    def __init__(self, ids: list[str], cluster: np.ndarray, vectors: np.ndarray, stored: list[Any]):
        self.ids = ids
        self.cluster = cluster
        self.vectors = vectors
        self.stored = stored


async def _fetch_clustered(client: Any, index: str) -> _Clustered:
    ids: list[str] = []
    order: list[int] = []
    cluster: list[int] = []
    stored: list[Any] = []
    pages: list[np.ndarray] = []
    seen = 0
    source = [
        ITEM_EMBEDDING_FIELD,
        'cluster_id',
        'cluster_distance',
        DISTANCE_REF_FIELD,
        NEAREST_FIELD,
    ]
    async for hits in iter_pages(client, index=index, query=_clustered_query(), source=source):
        kept = [h for h in hits if (h.get('_source') or {}).get(ITEM_EMBEDDING_FIELD)]
        page_ids, page = hits_to_rows(kept, ITEM_EMBEDDING_FIELD)
        if page is None:
            continue
        pages.append(page)
        for h in kept:
            sort = h.get('sort')
            order.append(int(sort[0]) if sort else seen)
            seen += 1
            src = h['_source']
            cluster.append(int(src['cluster_id']))
            stored.append(
                (src.get('cluster_distance'), src.get(DISTANCE_REF_FIELD), src.get(NEAREST_FIELD))
            )
        ids.extend(page_ids)
    if not pages:
        return _Clustered([], np.zeros(0, dtype=np.int64), np.zeros((0, 0), np.float32), [])
    rank = np.argsort(np.asarray(order), kind='stable')
    vectors = np.concatenate(pages) if len(pages) > 1 else pages[0]
    return _Clustered(
        [ids[i] for i in rank.tolist()],
        np.asarray(cluster, dtype=np.int64)[rank],
        vectors[rank],
        [stored[i] for i in rank.tolist()],
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


def _bodies(index: str) -> Any:
    def build(batch: Any) -> list[Any]:
        actions: list[Any] = []
        for crop_id, cid, fields in batch:
            actions.extend(_update(index, crop_id, cid, fields))
        return actions

    return build


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


_BACKFILL_PAUSE_S = 0.05


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
            fields = {DISTANCE_REF_FIELD: cluster_id, 'cluster_distance': float(dist)}
            actions.extend(_update(index, crop_id, cluster_id, fields))
        failed += await _write(client, actions)
        await asyncio.sleep(_BACKFILL_PAUSE_S)
    return failed


async def write_cluster_geometry(client: Any, *, index: str | None = None) -> dict[str, Any]:
    """Measure every clustered item against the cluster centroids and write
    the geometry fields. Returns a stage summary for the pipeline.

    Two passes, one cluster at a time: the first builds every cluster's
    centroid, the second writes each member's own distance (class
    clusters), measured cluster and nearest centroid.
    """
    items_index = index or get_curation_config().items_index
    data = await _fetch_clustered(client, items_index)
    members = {int(c): np.nonzero(data.cluster == c)[0] for c in np.unique(data.cluster)}
    centroid_ids: list[int] = []
    centroids: list[np.ndarray] = []
    for cid, rows in members.items():
        centroid = unit_centroid(data.vectors[rows])
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
    n_unchanged = 0
    async with AssignmentWriter(client, 'cluster_geometry', body_fn=_bodies(items_index)) as writer:
        pending: list[tuple[str, int, dict[str, Any]]] = []
        for cid in centroid_ids:
            rows = members[cid]
            sims = _normalize(data.vectors[rows]) @ matrix.T
            own = np.clip(1.0 - sims[:, column[cid]], 0.0, 2.0)
            nearest = [centroid_ids[k] for k in np.argmax(sims, axis=1).tolist()]
            is_class_cluster = cid < RESIDUAL_CLUSTER_ID_OFFSET
            for row, dist, near in zip(rows.tolist(), own.tolist(), nearest, strict=True):
                fields: dict[str, Any] = {DISTANCE_REF_FIELD: cid, NEAREST_FIELD: near}
                if is_class_cluster:
                    fields['cluster_distance'] = float(dist)
                have_dist, have_ref, have_near = data.stored[row]
                if (
                    have_ref == cid
                    and have_near == near
                    and (not is_class_cluster or have_dist == float(dist))
                ):
                    n_unchanged += 1
                else:
                    pending.append((data.ids[row], cid, fields))
                if len(pending) >= BULK_CHUNK:
                    await writer.submit(pending)
                    pending = []
            n_items += len(rows)
            n_class_distances += len(rows) if is_class_cluster else 0
            n_nearest_elsewhere += sum(1 for near in nearest if near != cid)
        if pending:
            await writer.submit(pending)
    n_write_errors = writer.n_failed
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
        'n_unchanged': n_unchanged,
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
    'write_member_distances',
]
