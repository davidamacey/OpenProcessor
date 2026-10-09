"""IVF centroid auto-retrain policy (growth + cooldown)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.curation import items_index
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.clustering.pool_size import MIN_RESIDUALS_FOR_CLUSTERING
from src.services.curation.embedding_state import embedded_clause


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


# Auto-retrain policy. Industry practice for IVF / k-means partition
# indexes (FAISS, Milvus, Pinecone): train centroids ONCE on a
# representative sample, add vectors continuously WITHOUT retraining,
# and reindex only on a coarse cadence or material distribution drift —
# because retraining shifts centroids, which reshuffles every crop's
# bucket and disrupts a human mid-sort. So auto-retrain fires only when
# BOTH hold:
#
#   1. growth — the residual pool exceeds RETRAIN_GROWTH_FACTOR x the
#      count the centroids were last trained on (1.5 = +50%); AND
#   2. cooldown — at least RETRAIN_MIN_INTERVAL_S has elapsed since the
#      last train, so a fast ingest burst can't trigger back-to-back
#      retrains.
#
# Per-crop ingest assignment (always on) keeps new crops query-able in
# the meantime against the existing centroids — retraining is only about
# refreshing the partition, not about routing new data.
RETRAIN_GROWTH_FACTOR = float(os.getenv('OP_IVF_RETRAIN_GROWTH', '1.5'))
# Default 24h cooldown — reindexing daily-or-slower is the conventional
# cadence; keeps centroids stable enough for multi-hour sort sessions.
RETRAIN_MIN_INTERVAL_S = float(os.getenv('OP_IVF_RETRAIN_MIN_INTERVAL_S', '86400'))


async def _count_residual_pool(client: AsyncOpenSearch, *, strict: bool = False) -> int:
    """Count crops in the residual cohort (same gate as the fetchers).

    ``strict=False`` (default) counts the FULL pool — every crop eligible
    for clustering, including ones already sitting in a candidate cluster
    from a prior run. This matches ``recluster_unvalidated=True`` /
    ``cluster_residuals``'s "broad" mode.

    ``strict=True`` counts only the NARROW slice ``cluster_residuals``'s
    default (``recluster_unvalidated=False``) mode actually trains on:
    crops with no ``cluster_id`` yet or one below
    ``RESIDUAL_CLUSTER_ID_OFFSET`` (i.e. not already in a candidate
    cluster). Needed so :func:`should_retrain_centroids` compares
    like-for-like against whichever mode produced the stored
    ``n_trained_on`` — see ``trained_mode`` in the centroid store
    metadata.
    """
    from src.services.curation.clustering import embedding_reduce as _ker

    filt: list[dict[str, Any]] = [embedded_clause(_ker.RESIDUAL_EMBEDDING_FIELD)]
    if strict:
        filt.append(
            {
                'bool': {
                    'should': [
                        {'bool': {'must_not': [{'exists': {'field': 'cluster_id'}}]}},
                        {'range': {'cluster_id': {'lt': RESIDUAL_CLUSTER_ID_OFFSET}}},
                    ],
                    'minimum_should_match': 1,
                }
            }
        )
    query = {
        'bool': {
            'filter': filt,
            'must_not': [
                {'term': {'class_validated': True}},
                {'terms': {'class_source': list(_ker.CONFIDENT_CLASS_SOURCES)}},
                {'term': {'class_excluded': True}},
            ],
        }
    }
    resp = await client.count(index=items_index(), body={'query': query})
    return int(resp.get('count', 0))


def _seconds_since_trained(trained_at: str | None) -> float | None:
    """Seconds since an ISO8601 ``trained_at``; None if unparseable."""
    if not trained_at:
        return None
    try:
        ts = datetime.fromisoformat(trained_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return (datetime.now(UTC) - ts).total_seconds()
    except (ValueError, TypeError):
        return None


async def should_retrain_centroids(client: AsyncOpenSearch) -> dict[str, Any]:
    """Decide whether the IVF centroids are stale enough to retrain.

    Fires only when BOTH the growth threshold and the cooldown are
    satisfied (see RETRAIN_GROWTH_FACTOR / RETRAIN_MIN_INTERVAL_S). If
    no centroids exist yet, trains as soon as the pool clears
    ``MIN_RESIDUALS_FOR_CLUSTERING`` (the first train; no cooldown).

    The growth comparison must use the SAME pool definition the stored
    ``n_trained_on`` was measured against. ``cluster_residuals``'s default
    mode (``recluster_unvalidated=False``, persisted as
    ``trained_mode='strict_residuals'``) trains on a narrow slice — crops
    not yet in any candidate cluster — while ``recluster_unvalidated=True``
    (``trained_mode='recluster_unvalidated'``) trains on the full pool.
    Comparing a strict-mode ``n_trained_on`` against the full pool count
    means the growth gate is satisfied permanently after any strict run
    (huge full-pool count vs a tiny narrow-slice training count), with
    only the 24h cooldown preventing constant retriggering. The automatic
    idle-worker trigger (the auto-label worker) always requests
    ``recluster_unvalidated=True``, so ``trained_mode`` defaults to
    ``'recluster_unvalidated'`` for centroids persisted before this field
    existed — that matches the common case and preserves prior behavior
    for stores the automatic path trained.

    Returns a diagnostic dict with ``should`` + the inputs behind it.
    """
    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

    store = IVFCentroidStore()

    if not store.is_trained():
        residual_count = await _count_residual_pool(client)
        should = residual_count >= MIN_RESIDUALS_FOR_CLUSTERING
        return {
            'should': should,
            'reason': 'no_centroids_yet' if should else 'too_few_residuals',
            'residual_count': residual_count,
            'n_trained_on': 0,
            'threshold': MIN_RESIDUALS_FOR_CLUSTERING,
            'age_seconds': None,
        }

    trained_mode = store.metadata.get('trained_mode', 'recluster_unvalidated')
    residual_count = await _count_residual_pool(client, strict=trained_mode == 'strict_residuals')
    n_trained_on = int(store.metadata.get('n_trained_on', 0))
    threshold = int(n_trained_on * RETRAIN_GROWTH_FACTOR)
    age = _seconds_since_trained(store.metadata.get('trained_at'))
    grown = residual_count > threshold
    cooled = age is None or age >= RETRAIN_MIN_INTERVAL_S

    if grown and cooled:
        reason = 'pool_grew'
    elif grown and not cooled:
        reason = 'grown_but_cooling_down'
    else:
        reason = 'within_threshold'
    return {
        'should': grown and cooled,
        'reason': reason,
        'residual_count': residual_count,
        'n_trained_on': n_trained_on,
        'trained_mode': trained_mode,
        'threshold': threshold,
        'age_seconds': age,
        'min_interval_s': RETRAIN_MIN_INTERVAL_S,
    }
