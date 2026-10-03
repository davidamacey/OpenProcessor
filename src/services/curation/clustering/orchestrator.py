"""
Curation clustering orchestrator — registers the item-level "vehicles"
cluster index, runs per-cluster AHC refinement, and dispatches the
residual-pool clusterer.

This module is the thin orchestrator. The actual residual-pool algorithm
implementations live in :py:mod:`src.services.curation.clustering.methods`
behind a small registry (HDBSCAN by default on cuML GPU, AHC as a
fallback). The refine endpoint still uses sklearn AHC directly because
the per-cluster cap (``MAX_REFINE_MEMBERS``, default 8000) keeps it fast and the
``distance_threshold`` knob is the right tool for splitting an existing
cluster.

Public surface:

1. :py:data:`ITEMS_CLUSTER_INDEX` — FAISS cluster-index handle.
2. :py:func:`refine_cluster` — per-cluster AHC, called from
   ``POST /curation/clusters/refine/{cluster_id}``.
3. :py:func:`cluster_residuals` — residual-pool clusterer; dispatches
   through :py:func:`cluster_methods.get_method`.
4. :py:func:`auto_promote_clusters` — re-exported from
   :py:mod:`src.services.curation.clustering.auto_promote` for back-compat.

Why complete + cosine + threshold (for the refine path):

- **Complete linkage** uses the *maximum* pairwise distance when merging,
  so a sub-cluster only grows if every member stays within
  ``distance_threshold`` of every other member.
- **Cosine distance** matches the PE / classifier training objective.
- **distance_threshold=0.25** (~75 % cosine similarity) — shared between
  refine and the AHC residual fallback so both surfaces are calibrated
  identically.

Skip-rules:

- Refine: > MAX_REFINE_MEMBERS (default 8000) members (skip+warn);
  < MIN_REFINE_MEMBERS (4) members (skip+info).
- Residual pool: < 32 residuals → no-op (return empty summary).
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.config.curation import items_index
from src.core.logging import get_logger
from src.services.clustering import ClusterIndex
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.clustering.id_normalize import run_update_by_query_polled
from src.services.curation.clustering.pool_size import MIN_RESIDUALS_FOR_CLUSTERING

# Class clusters occupy cluster_id 0..OFFSET-1; candidate clusters produced
# by ``cluster_residuals`` get the offset added so the namespaces never collide.
from src.services.curation.embedding_state import embedded_clause


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from opensearchpy import AsyncOpenSearch

    from src.services.clustering import ClusteringService


logger = get_logger(__name__)


ITEMS_CLUSTER_INDEX: ClusterIndex = ClusterIndex.VEHICLES
"""FAISS cluster-index role for the curation item tenant.

Deliberately reuses the pre-existing, unrelated visual-search
``ClusterIndex.VEHICLES`` role rather than adding a curation-specific
member to ``src/services/clustering.py`` — a naming leftover from the
reference deployment, tracked as a documented gap in
``docs/design/curation_design_rationale.md`` §6.
"""


# Refinement thresholds.
# > this — refinement is skipped. The refine path runs sklearn AHC with NO
# connectivity graph, so it builds a FULL pairwise distance matrix: ~8*n^2
# bytes (float64). Rough transient RAM in the yolo-api process:
#   2000 -> ~32 MB    5000 -> ~200 MB    8000 -> ~512 MB    12000 -> ~1.15 GB
# Plus ~4 KB/member to fetch embeddings. It's fast (AHC fit is seconds even
# at 8k) and off-loaded to a worker thread, so the binding constraint is RAM,
# not latency. Raise via OP_MAX_REFINE_MEMBERS as far as the container allows.
MAX_REFINE_MEMBERS = int(os.getenv('OP_MAX_REFINE_MEMBERS', '8000'))
# < this — skip (AHC needs at least a few points). The floor used to be 50 on
# the theory that small clusters don't benefit, but operators also use refine
# purely to *organize* a small bucket into like-kind sub-groups for faster
# select-and-label, so the floor is low — just enough to keep AHC well-defined.
MIN_REFINE_MEMBERS = 4

# AHC constants live in cluster_methods.ahc now; re-exported here so
# the refine endpoint and existing callers keep working.
from src.services.curation.clustering.methods.ahc import (  # noqa: E402
    AHC_DISTANCE_THRESHOLD,
    AHC_LINKAGE,
    AHC_METRIC,
)


def subcluster_label(idx: int) -> str:
    """Convert a sub-cluster index ``0,1,2,...`` into a label suffix ``a,b,c,...,aa,ab,...``."""
    if idx < 0:
        raise ValueError('subcluster index must be >= 0')
    out = ''
    n = idx
    while True:
        out = chr(ord('a') + (n % 26)) + out
        n = n // 26 - 1
        if n < 0:
            break
    return out


# ---------------------------------------------------------------------------
# Guarded bulk writers.
#
# Every clustering writer below fetches candidates, spends seconds-to-minutes
# fitting a model, then bulk-writes cluster_id/cluster_subid back. A blind
# ``{'doc': {...}}`` update clobbers any human label/verification/exclusion
# made to a doc *during* that fit. Each writer instead sends a guarded
# painless ``script`` update that noops when the doc's current state shows
# human ownership -- the write simply doesn't happen; the doc keeps whatever
# the human set.
#
# A ``GuardClause`` list is the single source of truth for a guard: the same
# list renders the painless condition (``_guard_condition_painless``) and
# decides the noop in plain Python (``_guard_condition_matches``, used by
# tests), so the two can't independently drift the way hand-written painless
# text mirroring a separate Python predicate could.
GuardClause = tuple[str, str, Any]


def _guard_condition_painless(clauses: list[GuardClause]) -> str:
    """Render an OR-of-clauses guard condition as painless source.

    ``op='eq'`` -> ``ctx._source['field'] == <literal>``.
    ``op='contains'`` -> ``ctx._source['field'] != null &&
    ctx._source['field'].contains('<value>')``.

    Bracket notation throughout (not ``ctx._source.field``) so this stays
    correct even when a deployment renames a ``RegionFields`` attribute to
    something that isn't a valid painless identifier.
    """
    parts: list[str] = []
    for field, op, value in clauses:
        if op == 'eq':
            lit = 'true' if value is True else 'false' if value is False else f"'{value}'"
            parts.append(f"ctx._source['{field}'] == {lit}")
        elif op == 'contains':
            parts.append(
                f"(ctx._source['{field}'] != null && ctx._source['{field}'].contains('{value}'))"
            )
        else:
            raise ValueError(f'unsupported guard op {op!r}')
    return ' || '.join(parts)


def _guard_condition_matches(clauses: list[GuardClause], source: dict[str, Any]) -> bool:
    """Python-side mirror of :func:`_guard_condition_painless`.

    The same ``clauses`` list drives both, so a future change to one
    guard's fields/values automatically shows up on both sides -- there is
    nothing to keep "in sync" because there's only one definition.
    """
    for field, op, value in clauses:
        v = source.get(field)
        if op == 'eq' and v == value:
            return True
        if op == 'contains' and isinstance(v, str) and value in v:
            return True
    return False


# NOT a full mirror of src.clients.occ_locks.is_locked_class (W10 fix
# pass, Opus review 2026-09-28, lock-rule call-site m3): this covers the
# human-marker class_source check (a string containing 'human') plus the
# class_validated / class_excluded guards vlm.py's _class_locked already
# applies on its own write path, but it deliberately does NOT cover
# is_locked_class's `test_holdout` clause. This write is cluster
# PLACEMENT (cluster_id/cluster_distance), not a class write, so an
# unvalidated holdout item may still have its cluster assignment updated
# by residual clustering -- freezing a holdout item's CLASS is a
# separate guard (vlm.py, the region worker, class_write_guard.py), not
# this one. Kept as its own clause list (rather than calling
# is_locked_class from painless, which isn't possible);
# test_orchestrator_guarded_writes.py cross-checks the two stay
# equivalent on the fields this clause list DOES cover (human-marker,
# class_validated, class_excluded), with explicit holdout/validated-
# import samples pinning the intended divergence.
CLASS_CLUSTER_WRITE_GUARD_CLAUSES: list[GuardClause] = [
    ('class_validated', 'eq', True),
    ('class_excluded', 'eq', True),
    ('class_source', 'contains', 'human'),
]


def _guarded_class_cluster_write(cid: int, dist: float | None) -> dict[str, Any]:
    """Guarded bulk-update body for the residual/assign class-cluster
    writers (:func:`cluster_residuals`, :func:`assign_only_residuals`):
    noop instead of overwriting a human-owned (or validated/excluded)
    class row that changed while the fit was running."""
    return {
        'script': {
            'lang': 'painless',
            'params': {'cid': cid, 'dist': dist},
            'source': (
                f'if ({_guard_condition_painless(CLASS_CLUSTER_WRITE_GUARD_CLAUSES)})'
                " { ctx.op = 'noop'; return; }"
                " ctx._source['cluster_id'] = params.cid;"
                " ctx._source.remove('cluster_subid');"
                " ctx._source['cluster_distance'] = params.dist;"
                # The cluster the distance was measured against;
                # a later move leaves it pointing at the old cluster.
                " ctx._source['cluster_distance_cluster_id'] = params.cid;"
            ),
        }
    }


def _log_bulk_write_errors(op: str, resp: dict[str, Any]) -> None:
    """Log each failed bulk item (id + status + reason) at
    warning level instead of only a chunk-level 'errors: true' flag.
    Doesn't raise -- matches this module's existing partial-bulk-failure
    behavior of proceeding rather than aborting the whole run."""
    for item in resp.get('items') or []:
        action: dict[str, Any] = next(iter(item.values()), {})
        status = action.get('status')
        if status is not None and status >= 300:
            logger.warning(
                'clustering_bulk_write_item_failed',
                op=op,
                doc_id=action.get('_id'),
                status=status,
                error=action.get('error'),
            )


async def _fetch_cluster_members(
    client: AsyncOpenSearch,
    cluster_id: int,
    *,
    page_size: int = 1000,
    index: str | None = None,
) -> list[dict[str, Any]]:
    """Pull every item with ``cluster_id == cluster_id`` from ``index``.

    Reads the item residual embedding (``pe_embedding``), 1024x4B ~ 4KB
    each, so MAX_REFINE_MEMBERS (default 8000) members ~ 32MB -- safe to
    load into RAM. The embedding is normalized to the ``'embedding'`` key
    so :func:`refine_members` stays field-name-agnostic.
    """
    from src.services.curation.clustering.embedding_reduce import RESIDUAL_EMBEDDING_FIELD

    if index is None:
        index = items_index()
    embedding_field = RESIDUAL_EMBEDDING_FIELD

    members: list[dict[str, Any]] = []
    body = {
        'size': page_size,
        'query': {'term': {'cluster_id': cluster_id}},
        '_source': ['crop_id', 'class_name', 'class_id', 'class_validated', embedding_field],
    }
    resp = await client.search(index=index, body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    while hits:
        for h in hits:
            src = h.get('_source') or {}
            emb = src.get(embedding_field)
            if emb is None:
                continue
            members.append({**src, 'embedding': emb, '_id': h['_id']})
        resp = await client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']

    if scroll_id:
        try:
            await client.clear_scroll(scroll_id=scroll_id)
        except Exception as e:
            logger.warning('curation_clear_scroll_failed', error=str(e))
    return members


_SUBID_UPDATE_CHUNK = 1000


async def _bulk_update_subids(
    client: AsyncOpenSearch,
    updates: list[tuple[str, str]],
    *,
    index: str | None = None,
    expected_cluster_id: int | None = None,
    chunk_size: int = _SUBID_UPDATE_CHUNK,
) -> int:
    """Bulk-update ``cluster_subid`` on the supplied (doc_id, subid) pairs.

    When ``expected_cluster_id`` is given (refine's caller always passes it
    -- the cluster being refined), the write is a guarded painless script
    that noops if the doc's ``cluster_id`` no longer equals it. Refine
    snapshots members, fits AHC (can take seconds on a large cluster), then
    writes; a doc that moved to a different cluster in that window (a human
    relabel, a move endpoint call, another clustering job) must not have
    refine's now-stale sub-cluster numbering stamped onto it.

    Chunks into batches of ``chunk_size`` (<=1000) bulk actions with
    ``refresh=False`` per chunk, then issues one explicit index refresh at
    the end -- avoids refreshing the index once per chunk on a large refine.
    """
    if index is None:
        index = items_index()
    if not updates:
        return 0
    now = datetime.now(UTC).isoformat()
    for start in range(0, len(updates), chunk_size):
        chunk = updates[start : start + chunk_size]
        body: list[dict[str, Any]] = []
        for doc_id, subid in chunk:
            body.append({'update': {'_index': index, '_id': doc_id}})
            if expected_cluster_id is None:
                body.append({'doc': {'cluster_subid': subid, 'updated_at': now}})
            else:
                body.append(
                    {
                        'script': {
                            'lang': 'painless',
                            'params': {'cid': expected_cluster_id, 'subid': subid, 'now': now},
                            'source': (
                                "if (ctx._source['cluster_id'] != params.cid)"
                                " { ctx.op = 'noop'; return; }"
                                " ctx._source['cluster_subid'] = params.subid;"
                                ' ctx._source.updated_at = params.now;'
                            ),
                        }
                    }
                )
        resp = await client.bulk(body=body, refresh=False)
        if resp.get('errors'):
            _log_bulk_write_errors('bulk_update_subids', resp)
    try:
        await client.indices.refresh(index=index)
    except Exception as exc:
        logger.warning('bulk_update_subids_refresh_failed', error=str(exc))
    return len(updates)


def _compute_purity(class_names: list[str | None]) -> float:
    """Largest-class share among labelled members (None excluded). 0.0 if all None."""
    labelled = [c for c in class_names if c]
    if not labelled:
        return 0.0
    counts = Counter(labelled)
    top = counts.most_common(1)[0][1]
    return top / len(labelled)


async def refine_members(
    cluster_id: int,
    *,
    count_members: Callable[[], Awaitable[int]],
    fetch_members: Callable[[], Awaitable[list[dict[str, Any]]]],
    write_subids: Callable[[list[tuple[Any, str]]], Awaitable[int]],
    unit: Literal['items', 'boxes'],
    distance_threshold: float = AHC_DISTANCE_THRESHOLD,
    max_members: int = MAX_REFINE_MEMBERS,
) -> dict[str, Any]:
    """The AHC refine core shared by item clusters and region-box clusters.

    ``unit`` is what a member is, and names the response counts
    (``n_<unit>`` members, ``n_<unit>_updated`` members whose stored
    sub-id ``write_subids`` reports changed).

    ``fetch_members`` returns ``{'_id': <opaque key>, 'embedding': [...],
    'class_name': ...}`` dicts; ``write_subids`` receives ``(_id, subid)``
    pairs. Steps:

    1. ``count_members`` first -- a cluster far past ``max_members`` never
       pays for fetching every embedding just to learn it is too large.
    2. Skip if < MIN_REFINE_MEMBERS (4) members (too small) or
       > ``max_members`` (default 8000) members (too expensive).
    3. ``AgglomerativeClustering(linkage='complete', distance_threshold=0.25,
       metric='cosine')`` over the embeddings.
    4. ``write_subids`` the ``"47a"``, ``"47b"`` ... labels; every current
       member gets a fresh one, so re-running overwrites a previous partition.
    5. Purity (largest-class share among labelled members) in the summary.

    Returns ``{cluster_id, n_<unit>, n_subclusters, purity, action, ...}``.
    """
    log = logger.bind(cluster_id=cluster_id)
    log.info('curation_refine_cluster_start')
    n_key = f'n_{unit}'

    precount = await count_members()
    if precount > max_members:
        log.warning(
            'curation_refine_cluster_skipped_too_large_precount',
            n_members=precount,
            max_allowed=max_members,
        )
        return {
            'cluster_id': cluster_id,
            n_key: precount,
            'n_subclusters': 0,
            # Purity isn't computed here -- that would need the same full
            # fetch this precount check exists to avoid paying for.
            'purity': None,
            'action': 'skipped_too_large',
            'reason': (
                f'> {max_members} members ({precount} counted); AHC builds a full '
                '~8*n^2-byte pairwise matrix -- raise OP_MAX_REFINE_MEMBERS / '
                'max_members if RAM allows'
            ),
        }

    members = await fetch_members()
    n_members = len(members)

    if n_members < MIN_REFINE_MEMBERS:
        log.info(
            'curation_refine_cluster_skipped_too_small',
            n_members=n_members,
            min_required=MIN_REFINE_MEMBERS,
        )
        return {
            'cluster_id': cluster_id,
            n_key: n_members,
            'n_subclusters': 0,
            'purity': _compute_purity([m.get('class_name') for m in members]),
            'action': 'skipped_too_small',
            'reason': f'< {MIN_REFINE_MEMBERS} members; AHC offers no benefit',
        }

    if n_members > max_members:
        log.warning(
            'curation_refine_cluster_skipped_too_large',
            n_members=n_members,
            max_allowed=max_members,
        )
        return {
            'cluster_id': cluster_id,
            n_key: n_members,
            'n_subclusters': 0,
            'purity': _compute_purity([m.get('class_name') for m in members]),
            'action': 'skipped_too_large',
            'reason': (
                f'> {max_members} members; AHC builds a full ~8*n^2-byte pairwise '
                'matrix -- raise OP_MAX_REFINE_MEMBERS / max_members if RAM allows'
            ),
        }

    try:
        embeddings = np.asarray([m['embedding'] for m in members], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as e:
        log.error('curation_refine_cluster_embedding_load_failed', error=str(e))
        raise

    # sklearn import is local to keep startup fast and avoid a hard dep for
    # callers that never touch clustering.
    from sklearn.cluster import AgglomerativeClustering

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=AHC_LINKAGE,
        metric=AHC_METRIC,
    )
    # Off-load to a worker thread so a large cluster (close to
    # MAX_REFINE_MEMBERS) doesn't block the FastAPI event loop -- refine runs
    # in the yolo-api process, so a sync fit here would starve every other
    # request.
    sub_labels = await asyncio.to_thread(clusterer.fit_predict, embeddings)
    n_subclusters = int(sub_labels.max() + 1) if len(sub_labels) > 0 else 0

    updates = [
        (member['_id'], f'{cluster_id}{subcluster_label(int(sub_idx))}')
        for member, sub_idx in zip(members, sub_labels, strict=True)
    ]
    n_updated = await write_subids(updates)

    # Per-sub-cluster purity, then weighted-mean as the cluster summary.
    sub_groups: dict[int, list[str | None]] = {}
    for member, sub_idx in zip(members, sub_labels, strict=True):
        sub_groups.setdefault(int(sub_idx), []).append(member.get('class_name'))
    weighted_purity = (
        sum(_compute_purity(names) * len(names) for names in sub_groups.values()) / n_members
    )

    summary: dict[str, Any] = {
        'cluster_id': cluster_id,
        n_key: n_members,
        'n_subclusters': n_subclusters,
        'purity': _compute_purity([m.get('class_name') for m in members]),
        'subcluster_weighted_purity': weighted_purity,
        f'n_{unit}_updated': n_updated,
        'distance_threshold': distance_threshold,
        'linkage': AHC_LINKAGE,
        'metric': AHC_METRIC,
        'action': 'refined',
    }
    log.info('curation_refine_cluster_done', **summary)
    return summary


async def refine_cluster(
    client: AsyncOpenSearch,
    cluster_id: int,
    *,
    distance_threshold: float = AHC_DISTANCE_THRESHOLD,
    max_members: int = MAX_REFINE_MEMBERS,
) -> dict[str, Any]:
    """Per-cluster AHC refinement of an *item* cluster (``pe_embedding`` /
    ``cluster_id`` / ``cluster_subid``); see :func:`refine_members`. Region
    boxes refine through ``region_box_clustering.refine_region_cluster``."""
    index = items_index()

    async def count_members() -> int:
        resp = await client.count(index=index, body={'query': {'term': {'cluster_id': cluster_id}}})
        return int((resp or {}).get('count', 0))

    async def write_subids(updates: list[tuple[Any, str]]) -> int:
        return await _bulk_update_subids(
            client, updates, index=index, expected_cluster_id=cluster_id
        )

    return await refine_members(
        cluster_id,
        count_members=count_members,
        fetch_members=lambda: _fetch_cluster_members(client, cluster_id, index=index),
        write_subids=write_subids,
        unit='items',
        distance_threshold=distance_threshold,
        max_members=max_members,
    )


# auto_promote_clusters moved to src.services.curation.clustering.auto_promote
# on 2026-05-22 — it's opt-in / disabled-by-default in the pipeline
# pending a classifier-confidence-floor rewrite. Re-export the public name here
# so existing callers keep
# working without churn.
from src.services.curation.clustering.auto_promote import auto_promote_clusters  # noqa: E402


async def assign_cluster_to_crop(
    service: ClusteringService,
    embedding: np.ndarray,
) -> tuple[int, float]:
    """Convenience wrapper: assign a single embedding to its items cluster."""
    out = service.assign_cluster(ITEMS_CLUSTER_INDEX, embedding)
    return int(out.cluster_id), float(out.distance)


# ============================================================================
# Residual-pool clustering — dispatches via ClusterMethod registry.
# ============================================================================

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


# Crops parked by the primary-subject clustering gate (too small / too
# blurry to train centroids or be assigned). Distinct from -1 (unassigned /
# noise) and -2 (class_excluded) so the labeler can tell them apart. Parked
# crops keep all metadata + embeddings and are re-included by any looser
# (or off) recluster — they're shelved, not lost.
PARKED_CLUSTER_ID = -3

# Below this fraction of the residual pool carrying both gate fields, a
# gated recluster is unsafe (a range clause silently drops field-less docs).
GATE_MIN_COVERAGE = 0.98


def gate_must_clauses(
    max_rank: int | None,
    min_blur_ratio: float | None,
) -> list[dict[str, Any]]:
    """Build the "passes the primary-subject gate" must-clauses.

    Strict for clustering (unlike the null-safe UI slider): a crop must have
    the field and satisfy the bound to pass, so the parked complement is a
    clean partition. The coverage gate (:func:`residual_gate_coverage`)
    guarantees the fields are populated before a gated run.
    """
    clauses: list[dict[str, Any]] = []
    if max_rank is not None:
        clauses.append({'range': {'crop_rank_in_image': {'lte': max_rank}}})
    if min_blur_ratio is not None:
        clauses.append({'range': {'blur_lap_ratio': {'gte': min_blur_ratio}}})
    return clauses


def _residual_pool_filter() -> dict[str, Any]:
    """The base residual-cohort bool (same gate as the fetchers)."""
    from src.services.curation.clustering import embedding_reduce as _ker

    return {
        'filter': [embedded_clause(_ker.RESIDUAL_EMBEDDING_FIELD)],
        'must_not': [
            {'term': {'class_validated': True}},
            {'terms': {'class_source': list(_ker.CONFIDENT_CLASS_SOURCES)}},
            {'term': {'class_excluded': True}},
        ],
    }


async def residual_gate_coverage(
    client: AsyncOpenSearch,
    *,
    max_rank: int | None,
    min_blur_ratio: float | None,
) -> dict[str, Any]:
    """Fraction of the residual pool carrying the gate fields it needs.

    Returns ``{total, with_fields, coverage, sufficient}``. ``sufficient`` is
    False when coverage < :data:`GATE_MIN_COVERAGE`, signalling the caller to
    block the gated run until the backfill completes.
    """
    base = _residual_pool_filter()
    total_resp = await client.count(index=items_index(), body={'query': {'bool': base}})
    total = int(total_resp.get('count', 0))
    field_filter: list[dict[str, Any]] = list(base['filter'])
    if max_rank is not None:
        field_filter.append({'exists': {'field': 'crop_rank_in_image'}})
    if min_blur_ratio is not None:
        field_filter.append({'exists': {'field': 'blur_lap_ratio'}})
    cov_resp = await client.count(
        index=items_index(),
        body={'query': {'bool': {'filter': field_filter, 'must_not': base['must_not']}}},
    )
    with_fields = int(cov_resp.get('count', 0))
    coverage = (with_fields / total) if total else 1.0
    return {
        'total': total,
        'with_fields': with_fields,
        'coverage': round(coverage, 4),
        'sufficient': coverage >= GATE_MIN_COVERAGE,
    }


async def _park_gated_residuals(
    client: AsyncOpenSearch,
    *,
    max_rank: int | None,
    min_blur_ratio: float | None,
) -> int:
    """Set ``cluster_id=PARKED_CLUSTER_ID`` on residual crops failing the gate.

    The complement of the gated training/assign pool: residual crops that are
    too small (rank > max_rank) or too blurry (blur_lap_ratio < min). Uses
    update_by_query so it scales to the full pool without client-side paging.
    Returns the number of parked docs.
    """
    gate = gate_must_clauses(max_rank, min_blur_ratio)
    if not gate:
        return 0
    base = _residual_pool_filter()
    # "Fails the gate" = residual pool AND NOT(passes all gate clauses).
    # Also exclude docs already parked — rewriting cluster_id=-3 onto
    # a doc that's already -3 (with cluster_subid already null) is a
    # wasted write on every re-run of this gate.
    query = {
        'bool': {
            'filter': base['filter'],
            'must_not': [
                *base['must_not'],
                {'bool': {'filter': gate}},
                {'term': {'cluster_id': PARKED_CLUSTER_ID}},
            ],
        }
    }
    body = {
        'query': query,
        'script': {
            'source': 'ctx._source.cluster_id = params.parked; ctx._source.cluster_subid = null',
            'lang': 'painless',
            'params': {'parked': PARKED_CLUSTER_ID},
        },
    }
    try:
        # Polled, not blocking — see run_update_by_query_polled's
        # docstring (cluster_id_normalize.py): wait_for_completion=True
        # on a large items-index query can fail client-side response
        # parsing ("Too many headers received") even when the operation
        # completes successfully server-side, causing the transport to
        # silently retry the whole multi-minute operation from scratch.
        resp = await run_update_by_query_polled(
            client,
            index=items_index(),
            body=body,
            conflicts='proceed',
            refresh=True,
        )
        return int(resp.get('updated', 0))
    except Exception as exc:
        logger.warning('curation_park_gated_residuals_failed', error=str(exc))
        return 0


async def cluster_residuals(
    client: AsyncOpenSearch,
    *,
    recluster_unvalidated: bool = False,
    clustering_method: str | None = None,
    gate_max_rank: int | None = None,
    gate_min_blur_ratio: float | None = None,
    n_clusters: int | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Cluster the residual pool of unvalidated PE embeddings.

    Dispatches to a :class:`ClusterMethod` from
    :py:mod:`src.services.curation.clustering.methods`:

    * ``"ivf"`` (default) — FAISS k-means partitioning into a fixed K
      buckets, trained on a bounded sample and persisted so ingest +
      assign_only reuse the centroids. No ``-1`` noise; every crop gets
      a bucket. Chosen because the item-embedding manifold is
      continuously dense (no density gaps for HDBSCAN, no geometry for
      UMAP to preserve).
    * ``"ahc"`` — sklearn AgglomerativeClustering complete-linkage on a
      sparse cosine kNN graph. Same algorithm as the refine endpoint;
      retained as a fallback. Don't use on large n — the C-level
      merge loop starves the worker heartbeat coroutine.
    * ``"hdbscan"`` — cuML GPU HDBSCAN; dormant. Collapses on this data
      (one mega-cluster or 100% noise) but kept for a future curated
      subset that has actual density structure.

    Pipeline:

    1. Fetch embeddings for unvalidated crops. Two pool modes:

       * default — items not yet in a candidate cluster.
       * ``recluster_unvalidated=True`` — also pulls items already in
         candidate clusters so smaller candidates can fuse.

    2. Dispatch to the chosen :class:`ClusterMethod`.
    3. Add the ``RESIDUAL_CLUSTER_ID_OFFSET`` to the method's raw labels
       so candidate ids never collide with the class-id namespace
       (0..80). The raw label order is written as-is (no size-based
       renumbering) so this matches :func:`assign_only_residuals` and
       the ingest-time assign path, which both write
       ``RESIDUAL_CLUSTER_ID_OFFSET + raw_centroid_index`` straight from
       :class:`IVFCentroidStore`. A prior version renumbered so
       cluster_id=0 was always the largest bucket, but that renumbering
       was applied only here — never persisted to the centroid store,
       and never applied by ingest/assign_only — so the same crop could
       get a different cluster_id depending on which code path last
       clustered it. If a largest-first ordering is needed again, it
       must be a derived sort (e.g. by cluster size, computed from the
       live data) rather than baked into the id itself.
    4. Bulk-write back to OpenSearch in chunks of 2000.
    """
    from src.services.curation.clustering import embedding_reduce
    from src.services.curation.clustering.backend import detect_cluster_backend
    from src.services.curation.clustering.methods import DEFAULT_METHOD, get_method
    from src.services.curation.strategy_registry import resolve_effective_default

    # DEFAULT_METHOD stays the ultimate fallback (resolve_effective_default
    # falls back to it internally too) -- an explicit ?clustering_method
    # always wins over any shared-settings override, same precedence every
    # other real endpoint's omitted-param resolution uses.
    if clustering_method:
        method_name = clustering_method.lower()
    else:
        resolved_default = await resolve_effective_default('cluster', client)
        method_name = (resolved_default or DEFAULT_METHOD).lower()
    mode_label = 'recluster_unvalidated' if recluster_unvalidated else 'strict_residuals'

    # Primary-subject clustering gate (optional). When set, train + assign
    # only the largest, clear crops; park the rest. Block on insufficient
    # field coverage so a half-backfilled pool can't silently empty itself.
    gate_clauses = gate_must_clauses(gate_max_rank, gate_min_blur_ratio)
    gate_active = bool(gate_clauses)
    if gate_active:
        coverage = await residual_gate_coverage(
            client, max_rank=gate_max_rank, min_blur_ratio=gate_min_blur_ratio
        )
        if not coverage['sufficient']:
            return {
                'status': 'gate_coverage_insufficient',
                'method': method_name,
                'mode': mode_label,
                'gate_max_rank': gate_max_rank,
                'gate_min_blur_ratio': gate_min_blur_ratio,
                'gate_coverage': coverage,
                'hint': (
                    'Run the crop-rank and blur backfill scripts over the '
                    'residual pool before enabling the clustering gate.'
                ),
            }

    # asyncio.to_thread — see embedding_reduce.py's matching call site:
    # detect_cluster_backend()'s CUDA probe has been observed to block
    # 30+ seconds on a no-GPU-passthrough container (driver-mismatch
    # error path), long enough to starve the sibling heartbeat coroutine
    # and trigger a false "worker heartbeat stale" job failure.
    backend_info = await asyncio.to_thread(detect_cluster_backend)
    if progress is not None:
        progress.set_backend(
            name=backend_info.name,
            detail=backend_info.detail,
            free_vram_mb=backend_info.free_vram_mb,
        )

    # PIT + sliced parallel fetch by default (~8x faster than scroll
    # on n=347k). Falls back internally to the sequential scroll
    # variant if PIT isn't available on the cluster.
    ids, embeddings = await embedding_reduce.fetch_residual_embeddings_parallel(
        client,
        include_candidate_clusters=recluster_unvalidated,
        candidate_cluster_id_min=RESIDUAL_CLUSTER_ID_OFFSET,
        extra_must=gate_clauses or None,
        progress=progress,
    )
    if len(ids) < MIN_RESIDUALS_FOR_CLUSTERING:
        return {
            'status': 'no_residuals' if not ids else 'too_few_residuals',
            'method': method_name,
            'mode': mode_label,
            'n_residuals': len(ids),
            'n_clusters': 0,
            'n_noise': 0,
            'cluster_id_offset': RESIDUAL_CLUSTER_ID_OFFSET,
            'backend': backend_info.name,
            'backend_detail': backend_info.detail,
        }

    if progress is not None:
        progress.update(processed=0, total=0)

    method_kwargs: dict[str, Any] = {}
    if n_clusters is not None and method_name == 'ivf':
        # Only IVF takes a fixed cluster count; AHC/HDBSCAN derive their own.
        method_kwargs['n_clusters'] = n_clusters
    method = get_method(method_name, **method_kwargs)
    result = await method.fit_predict(
        embeddings,
        backend_info=backend_info,
        progress=progress,
    )

    if progress is not None:
        progress.raise_if_cancelled()

    # Write the method's raw labels as-is (no size-based renumbering) —
    # see the docstring above for why: this must match assign_only_residuals
    # and the ingest-time assign path bit-for-bit so the same crop can't get
    # a different cluster_id depending on which code path clustered it.
    relabeled = np.asarray(result.labels, dtype=np.int64)
    n_clusters = len({int(x) for x in relabeled if int(x) != -1})
    n_noise = int((relabeled == -1).sum())

    # Per-crop centroid distance, aligned 1:1 with ids/labels. Methods
    # without one (IVF's single-bucket fallback, AHC, HDBSCAN) get the
    # member-mean centroid distance so outlier sorts work for every run.
    distances = result.distances
    if distances is not None:
        dist_list = distances.tolist()
    else:
        from src.services.curation.clustering.centroid_distance import member_centroid_distances

        dist_list = member_centroid_distances(embeddings, relabeled)

    # Chunk the bulk write — at production scale (~350k items) a single
    # bulk call exceeds the OpenSearch client's default 30s timeout.
    # Each chunk publishes incremental progress so the dashboard
    # advances during the write phase.
    BULK_CHUNK = 2000
    pairs = list(zip(ids, relabeled.tolist(), dist_list, strict=True))
    if progress is not None:
        progress.update(processed=0, total=len(pairs))
    n_written = 0
    for start in range(0, len(pairs), BULK_CHUNK):
        chunk = pairs[start : start + BULK_CHUNK]
        bulk_body: list[dict[str, Any]] = []
        for crop_id, label, dist in chunk:
            # Residual labels >= 0 get the offset; -1 (noise) stays as
            # -1 so the API renders it as cluster_kind="unassigned".
            new_cid = int(label)
            if new_cid >= 0:
                new_cid += RESIDUAL_CLUSTER_ID_OFFSET
            bulk_body.append({'update': {'_index': items_index(), '_id': crop_id}})
            # Guarded script, not a blind 'doc' update -- clear
            # cluster_subid (stale refine groupings from the doc's previous
            # cluster have no meaning in the new candidate) and set
            # cluster_distance (present for IVF, else null so a stale
            # distance from a prior method can't mislead), but noop if a
            # human relabeled/validated/excluded the doc since it was
            # fetched for this fit.
            bulk_body.append(
                _guarded_class_cluster_write(new_cid, float(dist) if dist is not None else None)
            )
        br = await client.bulk(body=bulk_body, refresh=False)
        if br.get('errors'):
            _log_bulk_write_errors('cluster_residuals', br)
        n_written += len(chunk)
        if progress is not None:
            progress.update(processed=n_written, total=len(pairs))
            progress.raise_if_cancelled()
    if pairs:
        try:
            await client.indices.refresh(index=items_index())
        except Exception as exc:
            logger.debug('curation_cluster_refresh_failed', error=str(exc))

    # Park the gate complement (residual crops too small / blurry to train or
    # assign) and persist the gate policy so ingest + assign_only apply it.
    # When the gate is off, clear any stale policy so the full pool clusters.
    n_parked = 0
    if gate_active:
        n_parked = await _park_gated_residuals(
            client, max_rank=gate_max_rank, min_blur_ratio=gate_min_blur_ratio
        )
    try:
        from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

        ivf_store = IVFCentroidStore()
        ivf_store.save_gate(max_rank=gate_max_rank, min_blur_ratio=gate_min_blur_ratio)
        # Record which pool mode produced the n_trained_on this run just
        # persisted (via ClusterMethod.fit_predict -> IVFCentroidStore.save)
        # so should_retrain_centroids can compare like-for-like later. See
        # its docstring for why this matters.
        if method_name == 'ivf':
            ivf_store.update_metadata(trained_mode=mode_label)
    except Exception as exc:
        logger.warning('curation_cluster_save_gate_failed', error=str(exc))

    return {
        'status': 'success',
        'method': result.method,
        'mode': mode_label,
        'n_residuals': len(ids),
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_parked': n_parked,
        'gate_active': gate_active,
        'gate_max_rank': gate_max_rank,
        'gate_min_blur_ratio': gate_min_blur_ratio,
        'cluster_id_offset': RESIDUAL_CLUSTER_ID_OFFSET,
        'backend': backend_info.name,
        'backend_detail': backend_info.detail,
        'cluster_method_backend': result.backend,
        'cluster_method_params': result.params,
        'cluster_method_extra': result.extra,
    }


async def assign_only_residuals(
    client: AsyncOpenSearch,
    *,
    chunk_size: int = 2000,
    progress: Any = None,
) -> dict[str, Any]:
    """Assign residual crops against the persisted IVF centroids, streaming.

    No retraining: loads the centroids saved by the last
    :func:`cluster_residuals` (IVF) run and streams the residual pool
    one scroll page at a time — assign + bulk-write per page, then drop
    the page from memory. Peak RAM is one page (~8 MB at chunk_size=2000)
    regardless of pool size, so this is the cheap periodic "re-sort the
    residuals with the current centroids" path (vs a full retrain).

    Returns a summary dict. If no centroids are persisted yet, returns
    ``{'status': 'no_centroids'}`` — the caller should run a full
    :func:`cluster_residuals` first.
    """
    from src.services.curation.clustering import embedding_reduce
    from src.services.curation.clustering.methods.ivf_store import IVFCentroidStore

    store = IVFCentroidStore()
    if not store.load():
        return {'status': 'no_centroids', 'method': 'ivf_assign_only'}

    # Apply the gate the last full recluster trained under (if any), so the
    # incremental re-sort matches the centroids' training distribution.
    gate = store.load_gate()
    gate_max_rank = gate.get('max_rank')
    gate_min_blur_ratio = gate.get('min_blur_ratio')
    gate_clauses = gate_must_clauses(gate_max_rank, gate_min_blur_ratio)

    field = embedding_reduce.RESIDUAL_EMBEDDING_FIELD
    filt: list[dict[str, Any]] = [{'exists': {'field': field}}, *gate_clauses]
    must_not: list[dict[str, Any]] = [
        {'term': {'class_validated': True}},
        {'terms': {'class_source': list(embedding_reduce.CONFIDENT_CLASS_SOURCES)}},
        {'term': {'class_excluded': True}},
    ]
    query = {'bool': {'filter': filt, 'must_not': must_not}}

    total_estimate = 0
    if progress is not None:
        try:
            cnt = await client.count(index=items_index(), body={'query': query})
            total_estimate = int(cnt.get('count', 0))
            progress.update(processed=0, total=total_estimate)
        except Exception as exc:
            logger.debug('curation_ivf_assign_count_failed', error=str(exc))

    body: dict[str, Any] = {'size': chunk_size, '_source': [field], 'query': query}
    resp = await client.search(index=items_index(), body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    n_written = 0
    try:
        while True:
            hits = resp['hits']['hits']
            if not hits:
                break
            chunk_ids: list[str] = []
            chunk_embs: list[np.ndarray] = []
            for h in hits:
                emb = (h.get('_source') or {}).get(field)
                if emb is None:
                    continue
                chunk_ids.append(h['_id'])
                chunk_embs.append(np.asarray(emb, dtype=np.float32))
            if chunk_embs:
                matrix = np.vstack(chunk_embs)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                matrix = matrix / np.where(norms == 0, 1.0, norms)
                labels, dists = store.assign_batch_with_distances(matrix)
                bulk_body: list[dict[str, Any]] = []
                for crop_id, label, dist in zip(
                    chunk_ids, labels.tolist(), dists.tolist(), strict=True
                ):
                    new_cid = int(label) + RESIDUAL_CLUSTER_ID_OFFSET
                    bulk_body.append({'update': {'_index': items_index(), '_id': crop_id}})
                    # Guarded script — see cluster_residuals above.
                    bulk_body.append(_guarded_class_cluster_write(new_cid, float(dist)))
                br = await client.bulk(body=bulk_body, refresh=False)
                if br.get('errors'):
                    _log_bulk_write_errors('assign_only_residuals', br)
                n_written += len(chunk_ids)
                if progress is not None:
                    progress.update(processed=n_written, total=total_estimate or n_written)
                    progress.raise_if_cancelled()
            if not scroll_id:
                break
            resp = await client.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = resp.get('_scroll_id')
    finally:
        if scroll_id:
            try:
                await client.clear_scroll(scroll_id=scroll_id)
            except Exception as exc:
                logger.debug('curation_ivf_assign_clear_scroll_failed', error=str(exc))

    if n_written:
        try:
            await client.indices.refresh(index=items_index())
        except Exception as exc:
            logger.debug('curation_ivf_assign_refresh_failed', error=str(exc))

    # Park the gate complement so newly-gated crops don't linger in stale
    # candidate buckets after an incremental re-sort.
    n_parked = 0
    if gate_clauses:
        n_parked = await _park_gated_residuals(
            client, max_rank=gate_max_rank, min_blur_ratio=gate_min_blur_ratio
        )

    return {
        'status': 'success',
        'method': 'ivf_assign_only',
        'n_assigned': n_written,
        'n_parked': n_parked,
        'gate_active': bool(gate_clauses),
        'n_clusters': store.n_clusters,
        'cluster_id_offset': RESIDUAL_CLUSTER_ID_OFFSET,
        'centroids_trained_at': store.metadata.get('trained_at'),
    }


__all__ = [
    'AHC_DISTANCE_THRESHOLD',
    'AHC_LINKAGE',
    'AHC_METRIC',
    'ITEMS_CLUSTER_INDEX',
    'MAX_REFINE_MEMBERS',
    'MIN_REFINE_MEMBERS',
    'MIN_RESIDUALS_FOR_CLUSTERING',
    'PARKED_CLUSTER_ID',
    'RESIDUAL_CLUSTER_ID_OFFSET',
    'assign_cluster_to_crop',
    'assign_only_residuals',
    'auto_promote_clusters',
    'cluster_residuals',
    'gate_must_clauses',
    'refine_cluster',
    'refine_members',
    'residual_gate_coverage',
    'should_retrain_centroids',
    'subcluster_label',
]
