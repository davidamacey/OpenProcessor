"""The region false-positive matcher: sub-type the false-positive boxes, keep one centroid per
sub-type and pull tight matches into the false-positive state.

Region false positives are a box *state* (``false_positive``), not a cluster: a box does not carry
an integer bucket or sub-cluster id, and the served cluster fields of a box belong to the
partition projection alone. The sub-type labels (``a``, ``b``, ...) live in the centroid store's
metadata (``FalsePositiveCentroidStore``), which ``GET /regions/suspected_false_positives`` reads.
Region partitioning itself is ``cluster_run(target='region_box')``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from opensearchpy.exceptions import OpenSearchException

from src.config import get_region_fields
from src.config.curation_accessors import items_index
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.domain.region_boxes import RegionBox, derive_status
from src.services.curation.clustering.region_box_rows import (
    BoxEdit,
    RegionBoxRow,
    scroll_box_rows,
    write_box_edits,
)
from src.services.curation.region_box_edits import with_state


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)

_FALSE_POSITIVE = RegionStatus.FALSE_POSITIVE.value
_GOOD_STATES: tuple[str, ...] = ('accepted',)

FP_TARGET_SUBTYPE_SIZE = 150  # target members per FP sub-type
FP_MIN_FOR_SUBTYPES = 32  # below this, one whole-bucket centroid


def subtype_label(idx: int) -> str:
    """``0, 1, 2, ...`` as ``a, b, ..., z, aa, ab, ...``."""
    if idx < 0:
        raise ValueError('sub-type index must be >= 0')
    out = ''
    n = idx
    while True:
        out = chr(ord('a') + (n % 26)) + out
        n = n // 26 - 1
        if n < 0:
            return out


def _unit_rows(rows: Sequence[RegionBoxRow]) -> np.ndarray:
    """The rows' vectors re-normalised to unit length. The k-means / cosine
    distance math assumes unit-norm rows, and a vector read straight off the
    index carries no guarantee its writer's normalisation survived."""
    x = np.asarray([r.vector for r in rows], dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


async def _refresh(client: AsyncOpenSearch) -> None:
    try:
        await client.indices.refresh(index=items_index())
    except OpenSearchException as exc:
        logger.debug('curation_region_fp_refresh_failed', error=str(exc))


async def build_region_fp_centroids(client: AsyncOpenSearch) -> dict[str, Any]:
    """Sub-type the false-positive boxes via MiniBatchKMeans and persist one centroid per sub-type.

    Reads every ``false_positive`` box that carries a vector, partitions them into ``k``
    sub-types and saves the ``k`` centroids to :class:`FalsePositiveCentroidStore` with the
    sub-type labels. Sub-types are fed by *boxes*: an item's good boxes never contribute their
    vectors. MiniBatchKMeans (not AHC): no 50/2000-member bounds, since the FP set starts tiny
    and grows large. Nothing is written onto the boxes.
    """
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    rows = await scroll_box_rows(client, index=items_index(), states=(_FALSE_POSITIVE,))
    n = len(rows)
    if n == 0:
        return {'status': 'skipped', 'reason': 'no_fp_embeddings', 'n_boxes': 0}

    x = _unit_rows(rows)
    k = 1 if n < FP_MIN_FOR_SUBTYPES else min(max(1, round(n / FP_TARGET_SUBTYPE_SIZE)), n)

    def _fit() -> np.ndarray:
        if k == 1:
            c = x.mean(axis=0, keepdims=True)
            c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
            return c.astype(np.float32)
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=4096)
        km.fit(x)
        # k-means centroids (an arithmetic mean of unit-norm members) are not themselves
        # unit-norm. FalsePositiveCentroidStore persists these into an IndexFlatL2 that
        # fp_store.search() maps to cosine similarity assuming every stored vector is
        # unit-norm; an un-normalized centroid silently shifts that mapping.
        centers = np.asarray(km.cluster_centers_, dtype=np.float32)
        unit: np.ndarray = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
        return unit

    centroids = await asyncio.to_thread(_fit)
    subids = [subtype_label(i) for i in range(int(k))]
    FalsePositiveCentroidStore().save(
        centroids,
        {
            'trained_at': datetime.now(UTC).isoformat(),
            'k': int(k),
            'n_boxes': n,
            'subids': subids,
            'dim': int(centroids.shape[1]),
        },
    )
    logger.info('curation_build_region_fp_centroids_done', n_boxes=n, k=int(k))
    return {'status': 'success', 'n_boxes': n, 'k': int(k), 'subids': subids}


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
    """Auto-move tight FP-centroid matches into the ``false_positive`` state.

    Scores every candidate box (:func:`fp_candidate_rows`); any whose vector is within
    ``threshold`` (L2 on unit-norm vectors) of a persisted FP sub-type centroid becomes a
    ``false_positive`` *box*, and its item's status is re-derived from the whole box list -- a
    sibling accepted box keeps the item ``detected``. Looser matches are left for the human
    ``suspected_false_positives`` review. No-op when no centroids exist.
    ``RegionFields.label_source='auto_fp_centroid'`` marks the moves as auditable + reversible
    (un-marking the box releases it).

    The write is guarded like every automated writer (a human-final item or a locked box is
    never touched), re-checked against the live item.
    """
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    store = FalsePositiveCentroidStore()
    if not store.load():
        return {
            'status': 'skipped',
            'reason': 'no_centroids',
            'n_boxes_moved': 0,
            'threshold': threshold,
        }

    rows = await fp_candidate_rows(client)
    moved: list[RegionBoxRow] = []
    if rows:
        dist, _idx = store.search(_unit_rows(rows))
        moved = [row for row, d in zip(rows, dist, strict=True) if float(d) <= threshold]

    def edit(box: RegionBox) -> RegionBox:
        return with_state(box, _FALSE_POSITIVE) if box.state == 'accepted' else box

    edits: dict[str, dict[str, BoxEdit]] = {}
    for row in moved:
        edits.setdefault(row.crop_id, {})[row.box_id] = edit

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
        n_boxes_scanned=len(rows),
        n_boxes_moved=len(moved),
        threshold=threshold,
        items_written=result['items_written'],
    )
    return {
        'status': 'success',
        'n_boxes_scanned': len(rows),
        'n_boxes_moved': len(moved),
        'threshold': threshold,
    }


__all__ = [
    'FP_MIN_FOR_SUBTYPES',
    'FP_TARGET_SUBTYPE_SIZE',
    'auto_assign_fp_from_centroids',
    'build_region_fp_centroids',
    'fp_candidate_must_not',
    'fp_candidate_rows',
    'subtype_label',
]
