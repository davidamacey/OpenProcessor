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

1. :py:data:`VEHICLES_CLUSTER_INDEX` — FAISS cluster-index handle.
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
- **Cosine distance** matches the PE / v6 training objective.
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
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config import get_curation_config, get_region_fields
from src.config.region_state import RegionStatus
from src.core.logging import get_logger
from src.services.clustering import ClusterIndex

# Class clusters occupy cluster_id 0..OFFSET-1; candidate clusters produced
# by ``cluster_residuals`` get the offset added so the namespaces never collide.
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.clustering.id_normalize import run_update_by_query_polled


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.clustering import ClusteringService


logger = get_logger(__name__)


VEHICLES_CLUSTER_INDEX: ClusterIndex = ClusterIndex.VEHICLES
"""FAISS cluster-index role for the curation item tenant.

Deliberately reuses the pre-existing, unrelated visual-search
``ClusterIndex.VEHICLES`` role rather than adding a curation-specific
member to ``src/services/clustering.py`` — a naming leftover from the
reference deployment, tracked as a documented gap in
``docs/design/curation_design_rationale.md`` §6.
"""

ITEMS_INDEX = get_curation_config().items_index

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


def _subcluster_label(idx: int) -> str:
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
# F-3: guarded bulk writers.
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


# Mirrors src.clients.occ.is_human_owned_class's class_source check (a
# string containing 'human') plus the class_validated / class_excluded
# guards vlm.py's _class_locked already applies on its own write path. Kept
# as its own clause list (rather than calling is_human_owned_class from
# painless, which isn't possible) -- test_orchestrator_guarded_writes.py
# cross-checks the two stay equivalent.
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
            ),
        }
    }


def _region_write_guard_clauses(F: Any) -> list[GuardClause]:
    """F-3 region-write guard: mirrors :func:`fp_candidate_must_not`'s human
    clauses (``F.label_source``/``F.verifier`` == ``'human'``) plus
    ``F.validated`` -- a VLM-validated region is not ground truth (see
    ``fp_candidate_must_not``'s docstring) but a *human*-validated one is
    final and must never be reshuffled by an automated re-cluster."""
    return [
        (F.label_source, 'eq', 'human'),
        (F.verifier, 'eq', 'human'),
        (F.validated, 'eq', True),
    ]


def _guarded_region_write(F: Any, fields: dict[str, Any]) -> dict[str, Any]:
    """Guarded bulk-update body for the region-cluster writers
    (:func:`cluster_region_residuals`, :func:`auto_assign_fp_from_centroids`):
    noop instead of overwriting a human-verified/validated region. A
    ``None`` value in ``fields`` removes that field instead of nulling it."""
    clauses = _region_write_guard_clauses(F)
    params: dict[str, Any] = {}
    stmts: list[str] = []
    for i, (field, value) in enumerate(fields.items()):
        if value is None:
            stmts.append(f"ctx._source.remove('{field}')")
        else:
            pname = f'v{i}'
            params[pname] = value
            stmts.append(f"ctx._source['{field}'] = params.{pname}")
    source = f'if ({_guard_condition_painless(clauses)}) {{' + " ctx.op = 'noop'; return; }"
    source += ''.join(f' {s};' for s in stmts)
    return {'script': {'lang': 'painless', 'params': params, 'source': source}}


def _log_bulk_write_errors(op: str, resp: dict[str, Any]) -> None:
    """F-3 item 4: log each failed bulk item (id + status + reason) at
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
    index: str = ITEMS_INDEX,
    cluster_id_field: str = 'cluster_id',
    embedding_field: str | None = None,
) -> list[dict[str, Any]]:
    """Pull every doc with ``<cluster_id_field> == cluster_id`` from ``index``.

    Reads ``embedding_field`` (default the vehicle residual field
    ``pe_embedding``; the region path passes ``RegionFields.embedding``),
    1024x4B ≈ 4KB each, so MAX_REFINE_MEMBERS (default 8000) members ≈
    32MB — safe to load into RAM. The
    embedding is normalized to the ``'embedding'`` key so ``refine_cluster``
    stays field-name-agnostic.
    """
    if embedding_field is None:
        from src.services.curation.clustering.embedding_reduce import RESIDUAL_EMBEDDING_FIELD

        embedding_field = RESIDUAL_EMBEDDING_FIELD

    members: list[dict[str, Any]] = []
    body = {
        'size': page_size,
        'query': {'term': {cluster_id_field: cluster_id}},
        '_source': [
            'crop_id',
            'class_name',
            'class_id',
            'class_validated',
            embedding_field,
        ],
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
            logger.warning('legacy_clear_scroll_failed', error=str(e))
    return members


_SUBID_UPDATE_CHUNK = 1000


async def _bulk_update_subids(
    client: AsyncOpenSearch,
    updates: list[tuple[str, str]],
    *,
    index: str = ITEMS_INDEX,
    subid_field: str = 'cluster_subid',
    cluster_id_field: str = 'cluster_id',
    expected_cluster_id: int | None = None,
    chunk_size: int = _SUBID_UPDATE_CHUNK,
) -> int:
    """Bulk-update ``subid_field`` on the supplied (doc_id, subid) pairs.

    F-3: when ``expected_cluster_id`` is given (refine's caller always
    passes it -- the cluster being refined), the write is a guarded
    painless script that noops if the doc's ``cluster_id_field`` no longer
    equals ``expected_cluster_id``. Refine snapshots members, fits AHC
    (can take seconds on a large cluster), then writes; a doc that moved to
    a different cluster in that window (a human relabel, a move endpoint
    call, another clustering job) must not have refine's now-stale
    sub-cluster numbering stamped onto it.

    Chunks into batches of ``chunk_size`` (<=1000) bulk actions with
    ``refresh=False`` per chunk, then issues one explicit index refresh at
    the end -- avoids refreshing the index once per chunk on a large
    refine.
    """
    if not updates:
        return 0
    now = datetime.now(UTC).isoformat()
    for start in range(0, len(updates), chunk_size):
        chunk = updates[start : start + chunk_size]
        body: list[dict[str, Any]] = []
        for doc_id, subid in chunk:
            body.append({'update': {'_index': index, '_id': doc_id}})
            if expected_cluster_id is None:
                body.append({'doc': {subid_field: subid, 'updated_at': now}})
            else:
                body.append(
                    {
                        'script': {
                            'lang': 'painless',
                            'params': {'cid': expected_cluster_id, 'subid': subid, 'now': now},
                            'source': (
                                f"if (ctx._source['{cluster_id_field}'] != params.cid)"
                                " { ctx.op = 'noop'; return; }"
                                f" ctx._source['{subid_field}'] = params.subid;"
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


async def refine_cluster(
    client: AsyncOpenSearch,
    cluster_id: int,
    *,
    distance_threshold: float = AHC_DISTANCE_THRESHOLD,
    index: str = ITEMS_INDEX,
    cluster_id_field: str = 'cluster_id',
    embedding_field: str | None = None,
    subid_field: str = 'cluster_subid',
    max_members: int = MAX_REFINE_MEMBERS,
) -> dict[str, Any]:
    """Run per-cluster AHC refinement on the supplied ``cluster_id``.

    Steps:
    1. Pull all crops in the cluster from ``index``.
    2. Skip if < MIN_REFINE_MEMBERS (4) members (too small) or
       > MAX_REFINE_MEMBERS (default 8000) members (too expensive).
    3. ``AgglomerativeClustering(linkage='complete', distance_threshold=0.25,
       metric='cosine')`` over the embeddings.
    4. Bulk-write ``subid_field`` (e.g. ``"47a"``, ``"47b"``) back to each doc.
    5. Compute purity (largest-class share among labelled members) and return
       a summary.

    Defaults refine vehicle clusters over ``pe_embedding`` / ``cluster_id`` /
    ``cluster_subid``. The region path passes
    ``cluster_id_field=RegionFields.cluster_id``,
    ``embedding_field=RegionFields.embedding``,
    ``subid_field=RegionFields.cluster_subid`` (see :func:`refine_region_cluster`)
    to refine region buckets without touching vehicle clustering.

    Returns:
        ``{cluster_id, n_members, n_subclusters, purity, action, ...}``.
    """
    log = logger.bind(cluster_id=cluster_id, index=index, cluster_id_field=cluster_id_field)
    log.info('legacy_refine_cluster_start')

    # F-16: count before scrolling every member's embedding — a cluster
    # far past max_members should never pay for that fetch just to
    # discover it's too large to refine.
    count_resp = await client.count(
        index=index, body={'query': {'term': {cluster_id_field: cluster_id}}}
    )
    precount = int((count_resp or {}).get('count', 0))
    if precount > max_members:
        log.warning(
            'legacy_refine_cluster_skipped_too_large_precount',
            n_members=precount,
            max_allowed=max_members,
        )
        return {
            'cluster_id': cluster_id,
            'n_members': precount,
            'n_subclusters': 0,
            # Purity isn't computed here — that would need the same full
            # fetch this precount check exists to avoid paying for.
            'purity': None,
            'action': 'skipped_too_large',
            'reason': (
                f'> {max_members} members ({precount} counted); AHC builds a full '
                '~8*n^2-byte pairwise matrix — raise OP_MAX_REFINE_MEMBERS / '
                'max_members if RAM allows'
            ),
        }

    members = await _fetch_cluster_members(
        client,
        cluster_id,
        index=index,
        cluster_id_field=cluster_id_field,
        embedding_field=embedding_field,
    )
    n_members = len(members)

    if n_members < MIN_REFINE_MEMBERS:
        log.info(
            'legacy_refine_cluster_skipped_too_small',
            n_members=n_members,
            min_required=MIN_REFINE_MEMBERS,
        )
        return {
            'cluster_id': cluster_id,
            'n_members': n_members,
            'n_subclusters': 0,
            'purity': _compute_purity([m.get('class_name') for m in members]),
            'action': 'skipped_too_small',
            'reason': f'< {MIN_REFINE_MEMBERS} members; AHC offers no benefit',
        }

    if n_members > max_members:
        log.warning(
            'legacy_refine_cluster_skipped_too_large',
            n_members=n_members,
            max_allowed=max_members,
        )
        return {
            'cluster_id': cluster_id,
            'n_members': n_members,
            'n_subclusters': 0,
            'purity': _compute_purity([m.get('class_name') for m in members]),
            'action': 'skipped_too_large',
            'reason': (
                f'> {max_members} members; AHC builds a full ~8*n^2-byte pairwise '
                'matrix — raise OP_MAX_REFINE_MEMBERS / max_members if RAM allows'
            ),
        }

    # Stack embeddings.
    try:
        embeddings = np.asarray(
            [m['embedding'] for m in members],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError) as e:
        log.error('legacy_refine_cluster_embedding_load_failed', error=str(e))
        raise

    # AHC — sklearn import is local to keep startup fast and avoid a hard dep
    # for callers that never touch clustering.
    import asyncio as _asyncio

    from sklearn.cluster import AgglomerativeClustering

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=AHC_LINKAGE,
        metric=AHC_METRIC,
    )
    # Off-load to a worker thread so a large cluster (close to
    # MAX_REFINE_MEMBERS, default 8000) doesn't block the FastAPI event loop —
    # refine_cluster runs in the yolo-api process, not the dedicated
    # worker container, so a sync fit_predict here would starve every
    # other request.
    sub_labels = await _asyncio.to_thread(clusterer.fit_predict, embeddings)
    n_subclusters = int(sub_labels.max() + 1) if len(sub_labels) > 0 else 0

    # Map each doc to its subid string. Every current member of the
    # cluster gets a fresh subid, so any previous subid value is
    # overwritten — re-running refine on the same cluster produces a
    # clean partition. Crops that left this cluster between runs are
    # handled by the move/label endpoints clearing cluster_subid when
    # they change cluster_id.
    updates: list[tuple[str, str]] = []
    for member, sub_idx in zip(members, sub_labels, strict=True):
        subid = f'{cluster_id}{_subcluster_label(int(sub_idx))}'
        updates.append((member['_id'], subid))

    n_updated = await _bulk_update_subids(
        client,
        updates,
        index=index,
        subid_field=subid_field,
        cluster_id_field=cluster_id_field,
        expected_cluster_id=cluster_id,
    )

    # Per-sub-cluster purity, then weighted-mean as the cluster summary.
    sub_groups: dict[int, list[str | None]] = {}
    for member, sub_idx in zip(members, sub_labels, strict=True):
        sub_groups.setdefault(int(sub_idx), []).append(member.get('class_name'))
    weighted_purity = (
        sum(_compute_purity(names) * len(names) for names in sub_groups.values()) / n_members
    )
    overall_purity = _compute_purity([m.get('class_name') for m in members])

    summary: dict[str, Any] = {
        'cluster_id': cluster_id,
        'n_members': n_members,
        'n_subclusters': n_subclusters,
        'purity': overall_purity,
        'subcluster_weighted_purity': weighted_purity,
        'n_updated': n_updated,
        'distance_threshold': distance_threshold,
        'linkage': AHC_LINKAGE,
        'metric': AHC_METRIC,
        'action': 'refined',
    }
    log.info('legacy_refine_cluster_done', **summary)
    return summary


# auto_promote_clusters moved to src.services.curation.clustering.auto_promote
# on 2026-05-22 — it's opt-in / disabled-by-default in the pipeline
# pending a v6-confidence-floor rewrite. Re-export the public name here
# so existing callers (legacy_pipeline, tests/test_legacy_clustering) keep
# working without churn.
from src.services.curation.clustering.auto_promote import auto_promote_clusters  # noqa: E402


async def assign_cluster_to_crop(
    service: ClusteringService,
    embedding: np.ndarray,
) -> tuple[int, float]:
    """Convenience wrapper: assign a single embedding to its legacy_vehicles cluster."""
    out = service.assign_cluster(VEHICLES_CLUSTER_INDEX, embedding)
    return int(out.cluster_id), float(out.distance)


# ============================================================================
# Residual-pool clustering — dispatches via ClusterMethod registry.
# ============================================================================

# Minimum residual count below which clustering is a no-op.
MIN_RESIDUALS_FOR_CLUSTERING = 32

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

    filt: list[dict[str, Any]] = [{'exists': {'field': _ker.RESIDUAL_EMBEDDING_FIELD}}]
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
    resp = await client.count(index=ITEMS_INDEX, body={'query': query})
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
    idle-worker trigger (``legacy_auto_label_worker.py``) always requests
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
        'filter': [{'exists': {'field': _ker.RESIDUAL_EMBEDDING_FIELD}}],
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
    total_resp = await client.count(index=ITEMS_INDEX, body={'query': {'bool': base}})
    total = int(total_resp.get('count', 0))
    field_filter: list[dict[str, Any]] = list(base['filter'])
    if max_rank is not None:
        field_filter.append({'exists': {'field': 'crop_rank_in_image'}})
    if min_blur_ratio is not None:
        field_filter.append({'exists': {'field': 'blur_lap_ratio'}})
    cov_resp = await client.count(
        index=ITEMS_INDEX,
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
    # F-29: also exclude docs already parked — rewriting cluster_id=-3 onto
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
        # on a large legacy_vehicle_crops query can fail client-side response
        # parsing ("Too many headers received") even when the operation
        # completes successfully server-side, causing the transport to
        # silently retry the whole multi-minute operation from scratch.
        resp = await run_update_by_query_polled(
            client,
            index=ITEMS_INDEX,
            body=body,
            conflicts='proceed',
            refresh=True,
        )
        return int(resp.get('updated', 0))
    except Exception as exc:
        logger.warning('legacy_park_gated_residuals_failed', error=str(exc))
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
      a bucket. Chosen because the vehicle-embedding manifold is
      continuously dense (no density gaps for HDBSCAN, no geometry for
      UMAP to preserve) — see docs/design/clustering_methods.md.
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
                    'Run legacy_backfill_crop_rank.py and legacy_backfill_blur.py over the '
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
    ids, embeddings = await embedding_reduce.fetch_residual_v6_embeddings_parallel(
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
            bulk_body.append({'update': {'_index': ITEMS_INDEX, '_id': crop_id}})
            # F-3: guarded script, not a blind 'doc' update -- clear
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
            await client.indices.refresh(index=ITEMS_INDEX)
        except Exception as exc:
            logger.debug('legacy_cluster_refresh_failed', error=str(exc))

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
        logger.warning('legacy_cluster_save_gate_failed', error=str(exc))

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
            cnt = await client.count(index=ITEMS_INDEX, body={'query': query})
            total_estimate = int(cnt.get('count', 0))
            progress.update(processed=0, total=total_estimate)
        except Exception as exc:
            logger.debug('legacy_ivf_assign_count_failed', error=str(exc))

    body: dict[str, Any] = {'size': chunk_size, '_source': [field], 'query': query}
    resp = await client.search(index=ITEMS_INDEX, body=body, scroll='2m')
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
                    bulk_body.append({'update': {'_index': ITEMS_INDEX, '_id': crop_id}})
                    # F-3: guarded script — see cluster_residuals above.
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
                logger.debug('legacy_ivf_assign_clear_scroll_failed', error=str(exc))

    if n_written:
        try:
            await client.indices.refresh(index=ITEMS_INDEX)
        except Exception as exc:
            logger.debug('legacy_ivf_assign_refresh_failed', error=str(exc))

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


# ============================================================================
# Region clustering — coarse partition + per-bucket AHC refine over the
# RegionFields embedding (a deployment overlay may point this at an
# existing plate_pe_embedding field). Regions are all one
# class (e.g. license plate), so this is OUTLIER discovery: similar regions
# group together and false-positives / bad boxes fall out as sub-cluster
# outliers under refine. Writes the independent RegionFields cluster fields
# (not the vehicle-level cluster_*).
# ============================================================================

F = get_region_fields()

REGION_TARGET_BUCKET_SIZE = 800
# Target members per coarse bucket. K is chosen so buckets land well under
# MAX_REFINE_MEMBERS (2000), keeping per-bucket AHC refine cheap.
MIN_REGIONS_FOR_CLUSTERING = 32

# Permanent region false-positive bucket (e.g. license-plate FPs).
# Negative so it never collides with the flat KMeans namespace (0..K-1).
# Human FP marks park crops here; cluster_region_residuals excludes them so
# the good buckets' centroids stay clean. FPs vary widely (lights, bumpers,
# stickers, brackets) so build_region_fp_centroids sub-types this bucket.
FALSE_POSITIVE_REGION_CLUSTER_ID = -100
FP_TARGET_SUBTYPE_SIZE = 150  # target members per FP sub-type
FP_MIN_FOR_SUBTYPES = 32  # below this, one whole-bucket centroid


async def cluster_region_residuals(
    client: AsyncOpenSearch,
    *,
    max_rank: int | None = None,
    page_size: int = 2000,
) -> dict[str, Any]:
    """Coarse-partition boxed regions into ``RegionFields.cluster_id`` buckets.

    Scrolls every crop that carries ``RegionFields.embedding`` (optionally
    gated to top-N largest crops via ``max_rank`` over
    ``crop_rank_in_image``), runs MiniBatchKMeans over the unit-norm
    vectors, and writes ``RegionFields.cluster_id`` +
    ``RegionFields.cluster_distance``. Each bucket can then be AHC-refined
    via :func:`refine_region_cluster` to surface outliers.

    No confident-class gate and no RESIDUAL_CLUSTER_ID_OFFSET — regions are
    a single flat namespace in ``RegionFields.cluster_id`` (0..K-1).
    """
    filt: list[dict[str, Any]] = [{'exists': {'field': F.embedding}}]
    if max_rank is not None:
        filt.append({'range': {'crop_rank_in_image': {'lte': int(max_rank)}}})
    # FPs live in the permanent FALSE_POSITIVE_REGION_CLUSTER_ID bucket.
    # Exclude them so KMeans never reshuffles them back into good buckets
    # and the good buckets' centroids recompute clean.
    query = {
        'bool': {
            'filter': filt,
            'must_not': [{'term': {F.status: RegionStatus.FALSE_POSITIVE}}],
        }
    }

    ids: list[str] = []
    vecs: list[list[float]] = []
    body = {'size': page_size, 'query': query, '_source': [F.embedding]}
    resp = await client.search(index=ITEMS_INDEX, body=body, scroll='5m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    while hits:
        for h in hits:
            emb = (h.get('_source') or {}).get(F.embedding)
            if emb is not None:
                ids.append(h['_id'])
                vecs.append(emb)
        resp = await client.scroll(scroll_id=scroll_id, scroll='5m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
    if scroll_id:
        try:
            await client.clear_scroll(scroll_id=scroll_id)
        except Exception as e:
            logger.warning('legacy_clear_scroll_failed', error=str(e))

    n = len(ids)
    if n < MIN_REGIONS_FOR_CLUSTERING:
        return {'status': 'skipped', 'reason': 'too_few_regions', 'n_regions': n, 'n_clusters': 0}

    from sklearn.cluster import MiniBatchKMeans

    x = np.asarray(vecs, dtype=np.float32)
    # CM-3: re-normalize defensively. The k-means/cosine-distance math
    # below assumes unit-norm rows, but this reads region_embedding
    # straight off the index with no guarantee the writer's normalization
    # survived (or that every historical row was written by a
    # normalizing writer). A norm drift here silently breaks the
    # "cosine-ish distance to centroid" comment two lines down.
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    k = max(8, round(n / REGION_TARGET_BUCKET_SIZE))
    k = min(k, n)  # never more clusters than points

    def _fit() -> tuple[Any, Any]:
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=4096)
        labels = km.fit_predict(x)
        # Cosine-ish distance to assigned centroid (vectors are unit-norm).
        dists = np.linalg.norm(x - km.cluster_centers_[labels], axis=1)
        return labels, dists

    labels, dists = await asyncio.to_thread(_fit)

    now = datetime.now(UTC).isoformat()
    bulk: list[dict[str, Any]] = []
    n_written = 0
    for doc_id, lab, dist in zip(ids, labels, dists, strict=True):
        bulk.append({'update': {'_index': ITEMS_INDEX, '_id': doc_id}})
        # F-3: guarded script — noop instead of overwriting a
        # human-verified/validated region; a fresh coarse partition
        # invalidates any prior refine, so cluster_subid is removed.
        bulk.append(
            _guarded_region_write(
                F,
                {
                    F.cluster_id: int(lab),
                    F.cluster_distance: float(dist),
                    F.cluster_subid: None,
                    'updated_at': now,
                },
            )
        )
        if len(bulk) >= 1000:
            br = await client.bulk(body=bulk, refresh=False)
            if br.get('errors'):
                _log_bulk_write_errors('cluster_region_residuals', br)
            n_written += len(bulk) // 2
            bulk = []
    if bulk:
        br = await client.bulk(body=bulk, refresh=False)
        if br.get('errors'):
            _log_bulk_write_errors('cluster_region_residuals', br)
        n_written += len(bulk) // 2
    try:
        await client.indices.refresh(index=ITEMS_INDEX)
    except Exception as exc:
        logger.debug('legacy_region_cluster_refresh_failed', error=str(exc))

    logger.info(
        'legacy_cluster_region_residuals_done', n_regions=n, n_clusters=int(k), assigned=n_written
    )
    return {
        'status': 'success',
        'method': 'minibatch_kmeans',
        'n_regions': n,
        'n_clusters': int(k),
        'assigned': n_written,
        'max_rank': max_rank,
    }


async def refine_region_cluster(
    client: AsyncOpenSearch,
    region_cluster_id: int,
    *,
    distance_threshold: float = AHC_DISTANCE_THRESHOLD,
) -> dict[str, Any]:
    """AHC-refine one region bucket; writes ``RegionFields.cluster_subid``.

    Thin wrapper over :func:`refine_cluster` pinned to the region fields, so
    outlier regions (false-positives / bad boxes) split into their own
    sub-clusters exactly like vehicle-class refine.
    """
    return await refine_cluster(
        client,
        region_cluster_id,
        distance_threshold=distance_threshold,
        index=ITEMS_INDEX,
        cluster_id_field=F.cluster_id,
        embedding_field=F.embedding,
        subid_field=F.cluster_subid,
    )


# Background region-clustering job state, persisted to a small file. The API
# runs many uvicorn workers in one container; in-process state would make the
# status poll hit a worker that knows nothing about the job. A file on the
# shared container fs is consistent across all workers. Clustering 50k+
# regions scrolls hundreds of MB + writes every assignment (~7-8 min), far
# too long to hold an HTTP request — so the endpoint fires it and the UI
# polls this file.
_JOB_FILE = Path(
    os.getenv('OP_REGION_CLUSTER_JOB_FILE')
    or str(Path(tempfile.gettempdir()) / 'region_cluster_job.json')
)
_JOB_STALE_S = 1800.0  # a 'running' flag older than this is treated as dead
_DEFAULT_JOB: dict[str, Any] = {
    'running': False,
    'started_at': None,
    'finished_at': None,
    'result': None,
    'error': None,
}
_job_tasks: set[asyncio.Task[None]] = set()


def _read_region_cluster_job() -> dict[str, Any]:
    try:
        state: dict[str, Any] = json.loads(_JOB_FILE.read_text())
    except Exception:
        return dict(_DEFAULT_JOB)
    # Stale-guard: a worker that died mid-run would otherwise leave the flag
    # stuck on 'running' forever, wedging the button.
    if state.get('running') and state.get('started_at'):
        try:
            started = datetime.fromisoformat(state['started_at'])
            if (datetime.now(UTC) - started).total_seconds() > _JOB_STALE_S:
                state['running'] = False
                state['error'] = 'job timed out or worker died'
        except ValueError:
            pass
    return state


def _write_region_cluster_job(state: dict[str, Any]) -> None:
    try:
        _JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _JOB_FILE.with_suffix('.tmp')
        tmp.write_text(json.dumps(state))
        tmp.replace(_JOB_FILE)  # atomic rename
    except Exception as exc:
        logger.warning('legacy_plate_cluster_job_write_failed', error=str(exc))


def region_cluster_job_status() -> dict[str, Any]:
    """Cross-worker snapshot of the background plate-clustering job."""
    return _read_region_cluster_job()


# A manual AHC refine of a good plate bucket writes per-crop sub-ids that a
# full re-partition would wipe (sub-ids are cluster-local). We record the last
# refine time so the one-click pipeline can skip the destructive re-partition
# while recent refine work is still fresh (a SHORT TTL), unless the caller forces
# it OR a substantial batch of new FPs has accumulated since the last partition
# (which busts the TTL — the good-plate pool changed enough to be worth it).
REGION_REPARTITION_REFINE_TTL_S = 600.0  # 10 min — short; just protects in-progress refines
FP_REPARTITION_BUST_DELTA = 200  # this many new FPs since last partition busts the TTL
_REGION_REFINE_MARKER = Path(
    os.getenv('OP_REGION_REFINE_MARKER')
    or str(Path(tempfile.gettempdir()) / 'region_refine_marker.json')
)
_REGION_PARTITION_MARKER = Path(
    os.getenv('OP_REGION_PARTITION_MARKER')
    or str(Path(tempfile.gettempdir()) / 'region_partition_marker.json')
)


def mark_region_refine(cluster_id: int) -> None:
    """Record that a good plate bucket was just manually refined (TTL anchor)."""
    try:
        _REGION_REFINE_MARKER.parent.mkdir(parents=True, exist_ok=True)
        tmp = _REGION_REFINE_MARKER.with_suffix('.tmp')
        tmp.write_text(
            json.dumps({'last_refine_at': datetime.now(UTC).isoformat(), 'cluster_id': cluster_id})
        )
        tmp.replace(_REGION_REFINE_MARKER)
    except Exception as exc:
        logger.warning('legacy_plate_refine_marker_write_failed', error=str(exc))


def _read_region_refine_marker() -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads(_REGION_REFINE_MARKER.read_text())
        return data
    except Exception:
        return {}


def _write_region_partition_marker(fp_count: int) -> None:
    try:
        _REGION_PARTITION_MARKER.parent.mkdir(parents=True, exist_ok=True)
        tmp = _REGION_PARTITION_MARKER.with_suffix('.tmp')
        tmp.write_text(
            json.dumps({'last_partition_at': datetime.now(UTC).isoformat(), 'fp_count': fp_count})
        )
        tmp.replace(_REGION_PARTITION_MARKER)
    except Exception as exc:
        logger.warning('legacy_plate_partition_marker_write_failed', error=str(exc))


def _read_region_partition_marker() -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads(_REGION_PARTITION_MARKER.read_text())
        return data
    except Exception:
        return {}


async def _count_false_positives(client: AsyncOpenSearch) -> int:
    """Current count of false-positive region crops (the FP bucket population)."""
    try:
        resp = await client.count(
            index=ITEMS_INDEX,
            body={'query': {'term': {F.status: RegionStatus.FALSE_POSITIVE}}},
        )
        return int(resp.get('count', 0))
    except Exception as exc:
        logger.warning('legacy_count_fp_failed', error=str(exc))
        return 0


async def start_region_cluster_job(
    client: AsyncOpenSearch,
    *,
    max_rank: int | None = None,
    auto_fp_threshold: float | None = 0.20,
    rebuild_fp_centroids: bool = True,
    repartition_ttl_s: float = REGION_REPARTITION_REFINE_TTL_S,
    fp_bust_delta: int = FP_REPARTITION_BUST_DELTA,
    force_repartition: bool = False,
) -> dict[str, Any]:
    """Launch the full plate-clustering pipeline in the background (single-flight).

    Pipeline (FP-prep steps are best-effort so a failure can't block the main
    re-partition):
      1. ``rebuild_fp_centroids``: re-sub-type the FP bucket + rebuild its
         sub-type centroids from the current false positives.
      2. ``auto_fp_threshold`` > 0: auto-move plates within that L2 distance of an
         FP sub-centroid into the FP bucket (the tight, near-certain matches).
      3. re-partition the good plates (FPs — including the just-moved ones —
         excluded), so the good buckets' centroids stay clean. **Skipped** when a
         manual plate refine happened within ``repartition_ttl_s`` (a short TTL),
         to avoid wiping that fresh cluster-local sub-id work — UNLESS
         ``force_repartition`` or at least ``fp_bust_delta`` new FPs have
         accumulated since the last partition (a substantial change busts the TTL).

    Returns the job snapshot immediately so the caller never blocks. If a run is
    already in flight, returns its snapshot without starting another.
    """
    state = _read_region_cluster_job()
    if state.get('running'):
        return state
    started = datetime.now(UTC).isoformat()
    _write_region_cluster_job(
        {'running': True, 'started_at': started, 'finished_at': None, 'result': None, 'error': None}
    )

    async def _run() -> None:
        result: dict[str, Any] | None = None
        error: str | None = None
        extra: dict[str, Any] = {}
        try:
            if rebuild_fp_centroids:
                try:
                    extra['fp_centroids'] = await build_region_fp_centroids(client)
                except Exception as exc:
                    extra['fp_centroids'] = {'status': 'error', 'error': str(exc)}
                    logger.error('legacy_plate_job_fp_build_failed', error=str(exc))
            if auto_fp_threshold and auto_fp_threshold > 0:
                try:
                    extra['auto_fp'] = await auto_assign_fp_from_centroids(
                        client, threshold=auto_fp_threshold
                    )
                except Exception as exc:
                    extra['auto_fp'] = {'status': 'error', 'error': str(exc)}
                    logger.error('legacy_plate_job_auto_fp_failed', error=str(exc))
            # TTL gate: a re-partition clears good-plate sub-ids, so skip it
            # while a recent manual refine is still fresh — UNLESS forced, or a
            # substantial batch of FPs accumulated since the last partition
            # (then the good-plate pool changed enough to be worth re-clustering).
            current_fp = await _count_false_positives(client)
            fp_at_last = int(_read_region_partition_marker().get('fp_count', 0))
            fp_delta = current_fp - fp_at_last
            refine_at = _read_region_refine_marker().get('last_refine_at')
            refine_fresh = False
            if refine_at and not force_repartition:
                try:
                    age = (datetime.now(UTC) - datetime.fromisoformat(refine_at)).total_seconds()
                    refine_fresh = age < repartition_ttl_s
                except ValueError:
                    refine_fresh = False
            busts_ttl = fp_delta >= fp_bust_delta
            do_repartition = force_repartition or busts_ttl or not refine_fresh
            if do_repartition:
                result = await cluster_region_residuals(client, max_rank=max_rank)
                _write_region_partition_marker(current_fp)
            else:
                result = {
                    'status': 'skipped_repartition_ttl',
                    'reason': 'recent manual plate refine within TTL; sub-clusters preserved',
                    'last_refine_at': refine_at,
                    'repartition_ttl_s': repartition_ttl_s,
                    'fp_delta_since_partition': fp_delta,
                    'fp_bust_delta': fp_bust_delta,
                }
            result = {**result, **extra}
        except Exception as exc:
            error = str(exc)
            logger.error('legacy_plate_cluster_job_failed', error=str(exc))
        finally:
            _write_region_cluster_job(
                {
                    'running': False,
                    'started_at': started,
                    'finished_at': datetime.now(UTC).isoformat(),
                    'result': result,
                    'error': error,
                }
            )

    task = asyncio.create_task(_run())
    _job_tasks.add(task)
    task.add_done_callback(_job_tasks.discard)
    return _read_region_cluster_job()


# ============================================================================
# Plate false-positive centroids — sub-type the permanent FP bucket and
# persist one centroid per sub-type, in a single pass, so the sub-clusters an
# operator shift-selects and the centroids used by the suspected-FP matcher
# always agree. MiniBatchKMeans (not AHC) — no 50/2000-member bounds, since the
# FP bucket starts tiny and grows large.
# ============================================================================


async def build_region_fp_centroids(client: AsyncOpenSearch) -> dict[str, Any]:
    """Sub-type the FP bucket via MiniBatchKMeans + persist one centroid/sub-type.

    Scrolls every ``RegionFields.status='false_positive'`` crop carrying a
    ``RegionFields.embedding``, partitions them into ``k`` sub-types, writes
    ``RegionFields.cluster_subid`` (``'-100a'`` …) + ``RegionFields.cluster_distance`` per crop,
    and saves the ``k`` centroids to :class:`FalsePositiveCentroidStore`. FP crops
    without an embedding still carry the FP cluster id (set on mark) and still
    export — they are simply not sub-typed here.
    """
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    query = {
        'bool': {
            'filter': [
                {'term': {F.status: RegionStatus.FALSE_POSITIVE}},
                {'exists': {'field': F.embedding}},
            ]
        }
    }
    ids: list[str] = []
    vecs: list[list[float]] = []
    body = {'size': 2000, 'query': query, '_source': [F.embedding]}
    resp = await client.search(index=ITEMS_INDEX, body=body, scroll='5m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    while hits:
        for h in hits:
            emb = (h.get('_source') or {}).get(F.embedding)
            if emb is not None:
                ids.append(h['_id'])
                vecs.append(emb)
        resp = await client.scroll(scroll_id=scroll_id, scroll='5m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
    if scroll_id:
        try:
            await client.clear_scroll(scroll_id=scroll_id)
        except Exception as exc:
            logger.warning('legacy_fp_centroid_clear_scroll_failed', error=str(exc))

    n = len(ids)
    if n == 0:
        return {'status': 'skipped', 'reason': 'no_fp_embeddings', 'n_members': 0}

    from sklearn.cluster import MiniBatchKMeans

    x = np.asarray(vecs, dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    k = 1 if n < FP_MIN_FOR_SUBTYPES else min(max(1, round(n / FP_TARGET_SUBTYPE_SIZE)), n)

    def _fit() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if k == 1:
            c = x.mean(axis=0, keepdims=True)
            c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
            labels = np.zeros(n, dtype=int)
            return c.astype(np.float32), labels, np.linalg.norm(x - c[labels], axis=1)
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=4096)
        labels = km.fit_predict(x)
        # CM-3: k-means centroids (an arithmetic mean of unit-norm
        # members) are not themselves unit-norm. FalsePositiveCentroidStore
        # persists these into an IndexFlatL2 that fp_store.search() maps
        # to cosine similarity assuming every stored vector is unit-norm
        # -- an un-normalized centroid silently shifts that mapping.
        centers = km.cluster_centers_
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
        dists = np.linalg.norm(x - centers[labels], axis=1)
        return centers.astype(np.float32), labels, dists

    centroids, labels, dists = await asyncio.to_thread(_fit)

    now = datetime.now(UTC).isoformat()
    bulk: list[dict[str, Any]] = []
    for doc_id, lab, dist in zip(ids, labels, dists, strict=True):
        subid = f'{FALSE_POSITIVE_REGION_CLUSTER_ID}{_subcluster_label(int(lab))}'
        bulk.append({'update': {'_index': ITEMS_INDEX, '_id': doc_id}})
        bulk.append(
            {
                'doc': {
                    F.cluster_id: FALSE_POSITIVE_REGION_CLUSTER_ID,
                    F.cluster_subid: subid,
                    F.cluster_distance: float(dist),
                    'updated_at': now,
                }
            }
        )
        if len(bulk) >= 1000:
            await client.bulk(body=bulk, refresh=False)
            bulk = []
    if bulk:
        await client.bulk(body=bulk, refresh=False)
    try:
        await client.indices.refresh(index=ITEMS_INDEX)
    except Exception as exc:
        logger.debug('legacy_fp_centroid_refresh_failed', error=str(exc))

    subids = [f'{FALSE_POSITIVE_REGION_CLUSTER_ID}{_subcluster_label(i)}' for i in range(int(k))]
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
    logger.info('legacy_build_plate_fp_centroids_done', n_members=n, k=int(k))
    return {'status': 'success', 'n_members': n, 'k': int(k), 'subids': subids}


def fp_candidate_must_not() -> list[dict[str, Any]]:
    """OpenSearch must-not clauses for the FP-centroid candidate pool.

    Excludes crops that are already FP, test-holdout, or carry a HUMAN verdict
    (``RegionFields.label_source`` / ``RegionFields.verifier`` == ``human``).
    VLM-validated crops are deliberately NOT excluded: ``RegionFields.validated=true``
    is set by the VLM verifier for the large majority of regions, and its calls
    aren't trusted as ground truth — so a VLM "validated/detected" verdict must
    not shield a real false positive. Only a human's decision is final.
    """
    return [
        {'term': {F.status: RegionStatus.FALSE_POSITIVE}},
        {'term': {'test_holdout': True}},
        {'term': {F.label_source: 'human'}},
        {'term': {F.verifier: 'human'}},
    ]


async def auto_assign_fp_from_centroids(
    client: AsyncOpenSearch, *, threshold: float = 0.20
) -> dict[str, Any]:
    """Auto-move tight FP-centroid matches into the permanent FP bucket.

    Scans non-FP, non-human-validated region crops; any whose
    ``RegionFields.embedding`` is within ``threshold`` (L2 on unit-norm vectors)
    of a persisted FP sub-type centroid is flipped to
    ``RegionFields.status='false_positive'`` and parked in
    ``FALSE_POSITIVE_REGION_CLUSTER_ID`` with the matched sub-id. Looser matches
    (above ``threshold``) are left for the human ``suspected_false_positives``
    review. No-op when no centroids exist.
    ``RegionFields.label_source='auto_fp_centroid'`` marks the moves as
    auditable + reversible (un-marking releases the crop).
    """
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    store = FalsePositiveCentroidStore()
    if not store.load():
        return {'status': 'skipped', 'reason': 'no_centroids', 'n_moved': 0, 'threshold': threshold}

    query = {
        'bool': {
            'filter': [{'exists': {'field': F.embedding}}],
            'must_not': fp_candidate_must_not(),
        }
    }
    subids = store.metadata.get('subids', [])

    now = datetime.now(UTC).isoformat()
    n_scanned = 0
    moved: list[tuple[str, str | None, float]] = []
    body = {'size': 2000, 'query': query, '_source': [F.embedding]}
    resp = await client.search(index=ITEMS_INDEX, body=body, scroll='5m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    while hits:
        embs = np.asarray(
            [(h.get('_source') or {}).get(F.embedding) for h in hits],
            dtype=np.float32,
        )
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        dist, idx = store.search(embs)
        for h, d, ci in zip(hits, dist, idx, strict=True):
            n_scanned += 1
            if float(d) <= threshold:
                sub = subids[int(ci)] if 0 <= int(ci) < len(subids) else None
                moved.append((h['_id'], sub, float(d)))
        resp = await client.scroll(scroll_id=scroll_id, scroll='5m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
    if scroll_id:
        try:
            await client.clear_scroll(scroll_id=scroll_id)
        except Exception as exc:
            logger.warning('legacy_auto_fp_clear_scroll_failed', error=str(exc))

    bulk: list[dict[str, Any]] = []
    for doc_id, sub, d in moved:
        bulk.append({'update': {'_index': ITEMS_INDEX, '_id': doc_id}})
        # F-3: guarded script — the query above already excludes
        # human-verified regions via fp_candidate_must_not() at scroll
        # time, but a human write between the scroll and this write
        # (the scan + distance search can take a while) must still not
        # be clobbered, hence the same defense-in-depth guard.
        bulk.append(
            _guarded_region_write(
                F,
                {
                    F.status: RegionStatus.FALSE_POSITIVE,
                    F.label_source: 'auto_fp_centroid',
                    F.cluster_id: FALSE_POSITIVE_REGION_CLUSTER_ID,
                    F.cluster_subid: sub,
                    F.cluster_distance: d,
                    'updated_at': now,
                },
            )
        )
        if len(bulk) >= 1000:
            br = await client.bulk(body=bulk, refresh=False)
            if br.get('errors'):
                _log_bulk_write_errors('auto_assign_fp_from_centroids', br)
            bulk = []
    if bulk:
        br = await client.bulk(body=bulk, refresh=False)
        if br.get('errors'):
            _log_bulk_write_errors('auto_assign_fp_from_centroids', br)
    if moved:
        try:
            await client.indices.refresh(index=ITEMS_INDEX)
        except Exception as exc:
            logger.debug('legacy_auto_fp_refresh_failed', error=str(exc))

    logger.info(
        'legacy_auto_assign_fp_done', n_scanned=n_scanned, n_moved=len(moved), threshold=threshold
    )
    return {
        'status': 'success',
        'n_scanned': n_scanned,
        'n_moved': len(moved),
        'threshold': threshold,
    }


# Background FP-centroid job state — same cross-worker file pattern as the
# plate-clustering job above (own file so the two can run independently).
_FP_JOB_FILE = Path(
    os.getenv('OP_REGION_FP_JOB_FILE') or str(Path(tempfile.gettempdir()) / 'region_fp_job.json')
)


def _read_region_fp_job() -> dict[str, Any]:
    try:
        state: dict[str, Any] = json.loads(_FP_JOB_FILE.read_text())
    except Exception:
        return dict(_DEFAULT_JOB)
    if state.get('running') and state.get('started_at'):
        try:
            started = datetime.fromisoformat(state['started_at'])
            if (datetime.now(UTC) - started).total_seconds() > _JOB_STALE_S:
                state['running'] = False
                state['error'] = 'job timed out or worker died'
        except ValueError:
            pass
    return state


def _write_region_fp_job(state: dict[str, Any]) -> None:
    try:
        _FP_JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _FP_JOB_FILE.with_suffix('.tmp')
        tmp.write_text(json.dumps(state))
        tmp.replace(_FP_JOB_FILE)
    except Exception as exc:
        logger.warning('legacy_plate_fp_job_write_failed', error=str(exc))


def region_fp_centroid_job_status() -> dict[str, Any]:
    """Cross-worker snapshot of the background FP-centroid build job.

    Merges in the persisted centroid metadata (``trained_at``/``k``/``n_members``)
    so the UI can warn when the centroids are stale.
    """
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    state = _read_region_fp_job()
    store = FalsePositiveCentroidStore()
    if store.load():
        state['centroids'] = {
            'trained_at': store.metadata.get('trained_at'),
            'k': store.metadata.get('k'),
            'n_members': store.metadata.get('n_members'),
        }
    else:
        state['centroids'] = None
    return state


async def start_region_fp_centroid_job(client: AsyncOpenSearch) -> dict[str, Any]:
    """Launch :func:`build_region_fp_centroids` in the background (single-flight)."""
    state = _read_region_fp_job()
    if state.get('running'):
        return state
    started = datetime.now(UTC).isoformat()
    _write_region_fp_job(
        {'running': True, 'started_at': started, 'finished_at': None, 'result': None, 'error': None}
    )

    async def _run() -> None:
        result: dict[str, Any] | None = None
        error: str | None = None
        try:
            result = await build_region_fp_centroids(client)
        except Exception as exc:
            error = str(exc)
            logger.error('legacy_plate_fp_job_failed', error=str(exc))
        finally:
            _write_region_fp_job(
                {
                    'running': False,
                    'started_at': started,
                    'finished_at': datetime.now(UTC).isoformat(),
                    'result': result,
                    'error': error,
                }
            )

    task = asyncio.create_task(_run())
    _job_tasks.add(task)
    task.add_done_callback(_job_tasks.discard)
    return region_fp_centroid_job_status()


__all__ = [
    'AHC_DISTANCE_THRESHOLD',
    'AHC_LINKAGE',
    'AHC_METRIC',
    'FALSE_POSITIVE_REGION_CLUSTER_ID',
    'ITEMS_INDEX',
    'MAX_REFINE_MEMBERS',
    'MIN_REFINE_MEMBERS',
    'MIN_REGIONS_FOR_CLUSTERING',
    'MIN_RESIDUALS_FOR_CLUSTERING',
    'PARKED_CLUSTER_ID',
    'RESIDUAL_CLUSTER_ID_OFFSET',
    'VEHICLES_CLUSTER_INDEX',
    'assign_cluster_to_crop',
    'assign_only_residuals',
    'auto_assign_fp_from_centroids',
    'auto_promote_clusters',
    'build_region_fp_centroids',
    'cluster_region_residuals',
    'cluster_residuals',
    'fp_candidate_must_not',
    'gate_must_clauses',
    'mark_region_refine',
    'refine_cluster',
    'refine_region_cluster',
    'region_cluster_job_status',
    'region_fp_centroid_job_status',
    'residual_gate_coverage',
    'should_retrain_centroids',
    'start_region_cluster_job',
    'start_region_fp_centroid_job',
]
