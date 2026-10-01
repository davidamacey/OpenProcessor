"""Region-box clustering: coarse partition, per-bucket AHC refine and the
false-positive centroid matcher, all over *boxes*.

Regions of one item are independent clustering units: each accepted box
with a vector is a row in the KMeans partition, each ``false_positive`` box
is a row in the FP sub-typing, and the cluster result is stored on the box
(``region_boxes[].cluster_id`` / ``cluster_subid`` / ``cluster_distance``).
An item with a good box and a false-positive box therefore sits in a good
bucket *and* in the FP bucket at once -- the item itself has no cluster.

Regions are all one class, so this is outlier discovery: similar regions
group together and false positives / bad boxes fall out as sub-cluster
outliers under refine. The job orchestration (single-flight background
runs, status files, TTL markers) lives in
:mod:`src.services.curation.clustering.region_cluster_jobs`; every write
goes through :func:`~src.services.curation.clustering.region_box_rows.write_box_edits`.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config import get_region_fields
from src.config.curation import items_index
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.curation.cluster_ids import FALSE_POSITIVE_REGION_CLUSTER_ID
from src.services.curation.clustering.orchestrator import (
    AHC_DISTANCE_THRESHOLD,
    MAX_REFINE_MEMBERS,
    refine_members,
    subcluster_label,
)
from src.services.curation.clustering.region_box_rows import (
    BoxEdit,
    RegionBoxRow,
    count_boxes,
    scroll_box_rows,
    with_cluster,
    write_box_edits,
)
from src.services.curation.region_box_edits import with_state
from src.services.curation.region_boxes import RegionBox, derive_status


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)

_FALSE_POSITIVE = RegionStatus.FALSE_POSITIVE.value
_GOOD_STATES: tuple[str, ...] = ('accepted',)

REGION_TARGET_BUCKET_SIZE = 800
# Target members per coarse bucket. K is chosen so buckets land well under
# MAX_REFINE_MEMBERS, keeping per-bucket AHC refine cheap.
MIN_REGIONS_FOR_CLUSTERING = 32

# Permanent region false-positive bucket (``FALSE_POSITIVE_REGION_CLUSTER_ID``,
# defined in ``cluster_ids``). FPs vary widely (background clutter,
# similar-looking non-target objects, empty boxes) so
# build_region_fp_centroids sub-types this bucket.
FP_TARGET_SUBTYPE_SIZE = 150  # target members per FP sub-type
FP_MIN_FOR_SUBTYPES = 32  # below this, one whole-bucket centroid


def _unit_rows(rows: Sequence[RegionBoxRow]) -> np.ndarray:
    """The rows' vectors re-normalised to unit length. The k-means / cosine
    distance math assumes unit-norm rows, and a vector read straight off the
    index carries no guarantee its writer's normalisation survived."""
    x = np.asarray([r.vector for r in rows], dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _edits_by_crop(rows: Sequence[RegionBoxRow], edit_for: Any) -> dict[str, dict[str, BoxEdit]]:
    """``{crop_id: {box_id: edit}}`` for ``rows``; ``edit_for(i, row)`` is
    that row's edit."""
    edits: dict[str, dict[str, BoxEdit]] = {}
    for i, row in enumerate(rows):
        edits.setdefault(row.crop_id, {})[row.box_id] = edit_for(i, row)
    return edits


async def cluster_region_residuals(
    client: AsyncOpenSearch, *, max_rank: int | None = None
) -> dict[str, Any]:
    """Coarse-partition accepted region boxes into ``cluster_id`` buckets.

    Reads every accepted box that carries a vector (optionally gated to the
    top-N largest crops via ``max_rank`` over ``crop_rank_in_image``), runs
    MiniBatchKMeans over the unit-norm vectors and writes ``cluster_id`` +
    ``cluster_distance`` onto each box. Each bucket can then be AHC-refined
    via :func:`refine_region_cluster` to surface outliers.

    False-positive boxes live in the permanent
    ``FALSE_POSITIVE_REGION_CLUSTER_ID`` bucket and are never rows here, so
    k-means can't reshuffle them into good buckets and the good buckets'
    centroids recompute clean. No confident-class gate and no
    RESIDUAL_CLUSTER_ID_OFFSET -- regions are a single flat namespace
    (0..K-1).
    """
    filters: list[dict[str, Any]] = []
    if max_rank is not None:
        filters.append({'range': {'crop_rank_in_image': {'lte': int(max_rank)}}})
    rows = await scroll_box_rows(client, index=items_index(), states=_GOOD_STATES, filters=filters)

    n = len(rows)
    if n < MIN_REGIONS_FOR_CLUSTERING:
        return {'status': 'skipped', 'reason': 'too_few_regions', 'n_regions': n, 'n_clusters': 0}

    from sklearn.cluster import MiniBatchKMeans

    x = _unit_rows(rows)
    k = min(max(8, round(n / REGION_TARGET_BUCKET_SIZE)), n)  # never more clusters than points

    def _fit() -> tuple[Any, Any]:
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=4096)
        labels = km.fit_predict(x)
        # Cosine-ish distance to assigned centroid (vectors are unit-norm).
        return labels, np.linalg.norm(x - km.cluster_centers_[labels], axis=1)

    labels, dists = await asyncio.to_thread(_fit)

    # A fresh coarse partition invalidates any prior refine, so the edit
    # clears cluster_subid; it applies only to a box still accepted when
    # the write lands, and never to a human-final item or a locked box.
    edits = _edits_by_crop(
        rows,
        lambda i, _row: with_cluster(int(labels[i]), float(dists[i]), only_states=_GOOD_STATES),
    )
    result = await write_box_edits(client, index=items_index(), edits=edits, respect_human=True)
    await _refresh(client)

    logger.info('curation_cluster_region_residuals_done', n_regions=n, n_clusters=int(k), **result)
    return {
        'status': 'success',
        'method': 'minibatch_kmeans',
        'n_regions': n,
        'n_clusters': int(k),
        'assigned': result['written'],
        'max_rank': max_rank,
    }


async def _refresh(client: AsyncOpenSearch) -> None:
    try:
        await client.indices.refresh(index=items_index())
    except Exception as exc:
        logger.debug('curation_region_cluster_refresh_failed', error=str(exc))


async def refine_region_cluster(
    client: AsyncOpenSearch,
    region_cluster_id: int,
    *,
    distance_threshold: float = AHC_DISTANCE_THRESHOLD,
    max_members: int = MAX_REFINE_MEMBERS,
) -> dict[str, Any]:
    """AHC-refine one region bucket; writes ``cluster_subid`` onto its boxes.

    The bucket's members are the boxes whose ``cluster_id`` is
    ``region_cluster_id`` (accepted boxes in a good bucket, false-positive
    boxes in the FP bucket), so outlier regions split into their own
    sub-clusters exactly like item-class refine. A box that left the bucket
    between the fit and the write is skipped, not stamped with stale
    numbering.
    """
    index = items_index()
    states = (*_GOOD_STATES, _FALSE_POSITIVE)

    async def fetch_members() -> list[dict[str, Any]]:
        rows = await scroll_box_rows(
            client, index=index, states=states, cluster_id=region_cluster_id
        )
        return [
            {'_id': (r.crop_id, r.box_id), 'embedding': r.vector, 'class_name': r.class_name}
            for r in rows
        ]

    async def write_subids(updates: list[tuple[Any, str]]) -> int:
        edits: dict[str, dict[str, BoxEdit]] = {}
        for (crop_id, box_id), subid in updates:
            edits.setdefault(crop_id, {})[box_id] = _subid_edit(region_cluster_id, subid)
        result = await write_box_edits(client, index=index, edits=edits, respect_human=False)
        await _refresh(client)
        return result['written']

    return await refine_members(
        region_cluster_id,
        count_members=lambda: count_boxes(
            client, index=index, states=states, cluster_id=region_cluster_id
        ),
        fetch_members=fetch_members,
        write_subids=write_subids,
        distance_threshold=distance_threshold,
        max_members=max_members,
    )


def _subid_edit(cluster_id: int, subid: str) -> BoxEdit:
    def edit(box: RegionBox) -> RegionBox:
        if box.cluster_id != cluster_id:
            return box
        return dataclasses.replace(box, cluster_subid=subid)

    return edit


async def count_false_positive_boxes(client: AsyncOpenSearch) -> int:
    """Current count of false-positive boxes (the FP bucket population)."""
    try:
        return await count_boxes(client, index=items_index(), states=(_FALSE_POSITIVE,))
    except Exception as exc:
        logger.warning('curation_count_fp_failed', error=str(exc))
        return 0


async def build_region_fp_centroids(client: AsyncOpenSearch) -> dict[str, Any]:
    """Sub-type the FP bucket via MiniBatchKMeans + persist one centroid/sub-type.

    Reads every ``false_positive`` box that carries a vector, partitions them
    into ``k`` sub-types, writes ``cluster_subid`` (``'-100a'`` ...) +
    ``cluster_distance`` onto each box (the FP bucket id is already its
    ``cluster_id``) and saves the ``k`` centroids to
    :class:`FalsePositiveCentroidStore`. Sub-types are fed by *boxes*: an
    item's good boxes never contribute their vectors. FP boxes without a
    vector still carry the FP cluster id (set on mark) and still export --
    they are simply not sub-typed here. MiniBatchKMeans (not AHC): no
    50/2000-member bounds, since the FP bucket starts tiny and grows large.
    """
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    rows = await scroll_box_rows(client, index=items_index(), states=(_FALSE_POSITIVE,))
    n = len(rows)
    if n == 0:
        return {'status': 'skipped', 'reason': 'no_fp_embeddings', 'n_members': 0}

    from sklearn.cluster import MiniBatchKMeans

    x = _unit_rows(rows)
    k = 1 if n < FP_MIN_FOR_SUBTYPES else min(max(1, round(n / FP_TARGET_SUBTYPE_SIZE)), n)

    def _fit() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if k == 1:
            c = x.mean(axis=0, keepdims=True)
            c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
            labels = np.zeros(n, dtype=int)
            return c.astype(np.float32), labels, np.linalg.norm(x - c[labels], axis=1)
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=4096)
        labels = km.fit_predict(x)
        # k-means centroids (an arithmetic mean of unit-norm members) are not
        # themselves unit-norm. FalsePositiveCentroidStore persists these
        # into an IndexFlatL2 that fp_store.search() maps to cosine
        # similarity assuming every stored vector is unit-norm -- an
        # un-normalized centroid silently shifts that mapping.
        centers = km.cluster_centers_
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
        return centers.astype(np.float32), labels, np.linalg.norm(x - centers[labels], axis=1)

    centroids, labels, dists = await asyncio.to_thread(_fit)

    def _edit_for(i: int, _row: RegionBoxRow) -> BoxEdit:
        subid = f'{FALSE_POSITIVE_REGION_CLUSTER_ID}{subcluster_label(int(labels[i]))}'
        distance = float(dists[i])

        def edit(box: RegionBox) -> RegionBox:
            if box.state != _FALSE_POSITIVE:
                return box
            return dataclasses.replace(
                box,
                cluster_id=FALSE_POSITIVE_REGION_CLUSTER_ID,
                cluster_subid=subid,
                cluster_distance=distance,
            )

        return edit

    # A human-marked FP is exactly what this sub-types, so no human guard:
    # the edit still applies only to a box that is false_positive right now.
    await write_box_edits(
        client,
        index=items_index(),
        edits=_edits_by_crop(rows, _edit_for),
        respect_human=False,
    )
    await _refresh(client)

    now = datetime.now(UTC).isoformat()
    subids = [f'{FALSE_POSITIVE_REGION_CLUSTER_ID}{subcluster_label(i)}' for i in range(int(k))]
    FalsePositiveCentroidStore().save(
        centroids,
        {
            'trained_at': now,
            'k': int(k),
            'n_members': n,
            'subids': subids,
            'dim': int(centroids.shape[1]),
        },
    )
    logger.info('curation_build_region_fp_centroids_done', n_members=n, k=int(k))
    return {'status': 'success', 'n_members': n, 'k': int(k), 'subids': subids}


def fp_candidate_must_not() -> list[dict[str, Any]]:
    """Item-level must-not clauses for the FP-centroid candidate pool:
    test-holdout items and items carrying a HUMAN verdict
    (``RegionFields.label_source`` / ``RegionFields.verifier`` == ``human``).

    VLM-validated items are deliberately NOT excluded:
    ``RegionFields.validated=true`` is set for the large majority of
    regions, and a VLM verdict isn't trusted as ground truth -- so a
    "validated/detected" verdict must not shield a real false positive.
    Only a human's decision is final. The per-box side is
    :func:`fp_candidate_rows`.
    """
    F = get_region_fields()
    return [
        {'term': {'test_holdout': True}},
        {'term': {F.label_source: 'human'}},
        {'term': {F.verifier: 'human'}},
    ]


async def fp_candidate_rows(client: AsyncOpenSearch) -> list[RegionBoxRow]:
    """The boxes the FP matcher scores: accepted, with a vector, in an item
    that isn't test-holdout or human-decided -- and never a locked box (one
    a human or an import owns). The one candidate pool behind the auto-pull
    and ``GET /regions/suspected_false_positives``."""
    from src.clients.occ_locks import is_locked_box

    rows = await scroll_box_rows(
        client, index=items_index(), states=_GOOD_STATES, must_not=fp_candidate_must_not()
    )
    return [r for r in rows if not is_locked_box(r.box)]


async def auto_assign_fp_from_centroids(
    client: AsyncOpenSearch, *, threshold: float = 0.20
) -> dict[str, Any]:
    """Auto-move tight FP-centroid matches into the permanent FP bucket.

    Scores every candidate box (:func:`fp_candidate_rows`); any whose vector
    is within ``threshold`` (L2 on unit-norm vectors) of a persisted FP
    sub-type centroid becomes a ``false_positive`` *box* parked in
    ``FALSE_POSITIVE_REGION_CLUSTER_ID`` with the matched sub-id, and its
    item's status is re-derived from the whole box list -- a sibling
    accepted box keeps the item ``detected``. Looser matches are left for
    the human ``suspected_false_positives`` review. No-op when no centroids
    exist. ``RegionFields.label_source='auto_fp_centroid'`` marks the moves
    as auditable + reversible (un-marking the box releases it).

    The write is guarded like every automated writer (a human-final item or
    a locked box is never touched), re-checked against the live item.
    """
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    store = FalsePositiveCentroidStore()
    if not store.load():
        return {'status': 'skipped', 'reason': 'no_centroids', 'n_moved': 0, 'threshold': threshold}

    subids = store.metadata.get('subids', [])
    rows = await fp_candidate_rows(client)
    moved: list[tuple[RegionBoxRow, str | None, float]] = []
    if rows:
        dist, idx = store.search(_unit_rows(rows))
        for row, d, ci in zip(rows, dist, idx, strict=True):
            if float(d) <= threshold:
                sub = subids[int(ci)] if 0 <= int(ci) < len(subids) else None
                moved.append((row, sub, float(d)))

    def _edit_for(subid: str | None, distance: float) -> BoxEdit:
        def edit(box: RegionBox) -> RegionBox:
            if box.state != 'accepted':
                return box
            return dataclasses.replace(
                with_state(box, _FALSE_POSITIVE), cluster_subid=subid, cluster_distance=distance
            )

        return edit

    edits: dict[str, dict[str, BoxEdit]] = {}
    for row, sub, d in moved:
        edits.setdefault(row.crop_id, {})[row.box_id] = _edit_for(sub, d)

    F = get_region_fields()

    def item_fields(boxes: Sequence[RegionBox]) -> dict[str, Any]:
        status = derive_status(boxes, empty_status=RegionStatus.NO_REGION_BOX)
        return {
            F.status: status.value,
            F.verified: any(b.state == 'accepted' for b in boxes),
            F.label_source: 'auto_fp_centroid',
        }

    result = await write_box_edits(
        client,
        index=items_index(),
        edits=edits,
        respect_human=True,
        item_fields=item_fields,
    )
    if moved:
        await _refresh(client)

    logger.info(
        'curation_auto_assign_fp_done',
        n_scanned=len(rows),
        n_moved=len(moved),
        threshold=threshold,
        written_items=result['written'],
    )
    return {
        'status': 'success',
        'n_scanned': len(rows),
        'n_moved': len(moved),
        'threshold': threshold,
    }


__all__ = [
    'FP_MIN_FOR_SUBTYPES',
    'FP_TARGET_SUBTYPE_SIZE',
    'MIN_REGIONS_FOR_CLUSTERING',
    'REGION_TARGET_BUCKET_SIZE',
    'auto_assign_fp_from_centroids',
    'build_region_fp_centroids',
    'cluster_region_residuals',
    'count_false_positive_boxes',
    'fp_candidate_must_not',
    'fp_candidate_rows',
    'refine_region_cluster',
]
