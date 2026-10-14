"""Residual-pool embedding fetch for clustering and the UMAP visualization.

Pulls classifier/PE embeddings for residual crops (no confident class
assigned) from the items index, L2-normalized. Two variants share one
residual-pool gate: a sequential scroll
(:func:`fetch_residual_embeddings`) and a PIT + sliced parallel read
(:func:`fetch_residual_embeddings_parallel`) that
:func:`~src.services.curation.clustering.orchestrator.cluster_residuals`
calls for its embedding fetch.

Override the embedding field via ``OP_RESIDUAL_EMBEDDING_FIELD`` if
experimenting.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from opensearchpy.exceptions import OpenSearchException

from src.config.curation import items_index
from src.config.env import env_str
from src.core.logging import get_logger
from src.domain.embedding_state import embedded_clause
from src.domain.ingest_class_sources import confident_class_sources
from src.services.curation.clustering.page_rows import hits_to_rows, stack_pages


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


# Embedding field to reduce. ``pe_embedding`` is PE-Core-L14-336's
# foundation visual encoder (1024-d), trained on broad web imagery —
# better at grouping out-of-distribution items by semantic similarity
# than ``backbone_embedding`` (the secondary classifier's penultimate layer, which
# clusters by framing/lighting outside its confident range). A visual
# audit picked pe_embedding as the residual default.
RESIDUAL_EMBEDDING_FIELD = env_str('OP_RESIDUAL_EMBEDDING_FIELD', 'pe_embedding')

# Crops with these class_source values are *confidently* labeled and
# must NEVER be pulled into the residual pool — clustering would
# overwrite their cluster_id=class_id mapping with a candidate id and
# destroy the class-cluster grouping (regression 2026-05-23: an IVF
# run on the unfiltered query overwrote ~129k labeled crops' cluster
# ids before being cancelled).
#
# Specifically (src.domain.ingest_class_sources, derived from
# the configured ingest profiles -- resolved once at import like other
# OP_* config):
#   {secondary}_model — the ingest classifier labeled it above its floor
#                       ({primary}_model too when the primary assigns_class)
#   vlm               — the VLM matched a registered class name
#   human             — human-validated label
#
# Everything else (low-conf / unlabeled proposals, vlm_unmatched,
# vlm_new_class_pending, null) is residual and goes into the cluster.
CONFIDENT_CLASS_SOURCES = confident_class_sources()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


async def fetch_residual_embeddings(
    client: AsyncOpenSearch,
    *,
    include_candidate_clusters: bool = False,
    candidate_cluster_id_min: int | None = None,
    extra_must: list[dict[str, Any]] | None = None,
    progress: Any | None = None,
) -> tuple[list[str], np.ndarray]:
    """Pull unvalidated crops with embeddings for the AHC residual pool.

    Two pools, selected by ``include_candidate_clusters``:

    * **Default (False)** — crops with ``RESIDUAL_EMBEDDING_FIELD`` set,
      ``class_validated`` not true, AND either no ``cluster_id`` yet
      OR a ``cluster_id`` below ``candidate_cluster_id_min``. The intent
      is "everything unvalidated that isn't already in a candidate
      cluster" — fresh items (no cluster_id) plus items still sitting
      in classifier class buckets (cluster_id < candidate_cluster_id_min).
      Existing candidate clusters are left alone so they aren't churned
      every run.
    * **Broad (True)** — same, but candidate-bucketed crops are also
      included. Use this when the operator wants to re-pool candidate
      clusters so smaller ones can merge into bigger ones.

    Despite the name's historical prefix (kept for back-compat with the
    pipeline stages), the actual field read is governed by
    :py:data:`RESIDUAL_EMBEDDING_FIELD` — defaults to PE embeddings.

    When ``progress`` is supplied, the scroll loop updates
    ``processed`` after each page and checks the cancel sentinel so an
    operator-initiated cancel takes effect within a single scroll
    batch (~50 ms) instead of waiting for the full fetch to finish.
    """
    filt: list[dict[str, Any]] = [embedded_clause(RESIDUAL_EMBEDDING_FIELD)]
    # Exclude confidently-labeled crops from the residual pool. The
    # previous filter (class_validated != true) only caught the 102
    # human-validated rows because item_model and vlm writers don't
    # set class_validated; the 2026-05-23 IVF run pulled all 347k
    # crops including 219k labeled ones and started overwriting their
    # cluster_id=class_id mappings before cancel. CONFIDENT_CLASS_SOURCES
    # is the right gate: only crops with no confident class_source go
    # into clustering.
    must_not: list[dict[str, Any]] = [
        {'term': {'class_validated': True}},
        {'terms': {'class_source': list(CONFIDENT_CLASS_SOURCES)}},
        # Human-ignored crops (blurry / unidentifiable) stay out of the
        # residual pool so they're never re-clustered.
        {'term': {'class_excluded': True}},
    ]
    if not include_candidate_clusters and candidate_cluster_id_min is not None:
        # Include items with no cluster_id OR cluster_id below the
        # candidate threshold. Items already in a candidate cluster
        # (>= threshold) are excluded so they aren't re-pooled.
        filt.append(
            {
                'bool': {
                    'should': [
                        {'bool': {'must_not': [{'exists': {'field': 'cluster_id'}}]}},
                        {'range': {'cluster_id': {'lt': candidate_cluster_id_min}}},
                    ],
                    'minimum_should_match': 1,
                },
            },
        )
    if extra_must:
        # Primary-subject clustering gate (rank / blur). Narrows the pool
        # to the crops that should train centroids + be assigned.
        filt.extend(extra_must)

    # Pre-count so the progress bar has a real total. A separate count
    # query is cheap (no _source, no scroll) and lets the dashboard show
    # 12,000 / 90,000 instead of "working…" during the scroll.
    total_estimate = 0
    if progress is not None:
        try:
            count_resp = await client.count(
                index=items_index(),
                body={'query': {'bool': {'filter': filt, 'must_not': must_not}}},
            )
            total_estimate = int(count_resp.get('count', 0))
            progress.update(processed=0, total=total_estimate)
        except Exception as exc:
            logger.debug('curation_embedding_count_failed', error=str(exc))

    body: dict[str, Any] = {
        'size': 1000,
        '_source': [RESIDUAL_EMBEDDING_FIELD],
        'query': {'bool': {'filter': filt, 'must_not': must_not}},
    }
    ids: list[str] = []
    pages: list[np.ndarray] = []
    resp = await client.search(index=items_index(), body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    try:
        while True:
            hits = resp['hits']['hits']
            if not hits:
                break
            page_ids, page = hits_to_rows(hits, RESIDUAL_EMBEDDING_FIELD)
            if page is not None:
                ids.extend(page_ids)
                pages.append(page)
            if progress is not None:
                # Scroll-loop checkpoint: emit progress and honour cancel
                # after each page. Per-page is ~50 ms so a cancel is felt
                # within a fraction of a second of the click.
                progress.update(processed=len(ids), total=total_estimate or len(ids))
                progress.raise_if_cancelled()
            if not scroll_id:
                break
            resp = await client.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = resp.get('_scroll_id')
    finally:
        if scroll_id:
            try:
                await client.clear_scroll(scroll_id=scroll_id)
            except OpenSearchException as exc:
                logger.debug('curation_embedding_clear_scroll_failed', error=str(exc))

    if not pages:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, _normalize_rows(stack_pages(pages))


# ============================================================================
# PIT + sliced parallel fetch — N-way concurrent variant of the scroll
# loop above. ~8x faster on 347k docs (10 min -> ~1.5 min) by letting
# OpenSearch coordinate N async slices over a frozen Point-In-Time view.
#
# Use this for cluster_residuals; keep the scroll variant for callers
# that don't have a sortable index or want simpler semantics.
# ============================================================================


PARALLEL_FETCH_DEFAULT_SLICES = 8
PARALLEL_FETCH_PAGE_SIZE = 2000
PARALLEL_FETCH_PIT_KEEPALIVE = '5m'


async def fetch_residual_embeddings_parallel(
    client: AsyncOpenSearch,
    *,
    include_candidate_clusters: bool = False,
    candidate_cluster_id_min: int | None = None,
    extra_must: list[dict[str, Any]] | None = None,
    n_slices: int = PARALLEL_FETCH_DEFAULT_SLICES,
    page_size: int = PARALLEL_FETCH_PAGE_SIZE,
    progress: Any | None = None,
) -> tuple[list[str], np.ndarray]:
    """Same shape as :func:`fetch_residual_embeddings`, PIT + slice impl.

    Freezes a Point-In-Time view of the index and dispatches
    ``n_slices`` parallel async scrolls (each owning ``slice {id: i,
    max: n_slices}``). OpenSearch partitions by shard+doc-id so the
    slices return disjoint result sets. PIT keeps results stable even
    if the ingest path is writing concurrently.

    Falls back to the sequential scroll-based fetcher on any error
    (older OpenSearch, missing _shard_doc sort support, etc.) so the
    pipeline never breaks on a fetch-layer issue.
    """
    filt: list[dict[str, Any]] = [embedded_clause(RESIDUAL_EMBEDDING_FIELD)]
    # Same residual-pool gate as the scroll variant — exclude
    # confidently-labeled crops so clustering only touches the truly
    # residual cohort. See CONFIDENT_CLASS_SOURCES for the rule.
    must_not: list[dict[str, Any]] = [
        {'term': {'class_validated': True}},
        {'terms': {'class_source': list(CONFIDENT_CLASS_SOURCES)}},
        # Human-ignored crops (blurry / unidentifiable) stay out of the
        # residual pool so they're never re-clustered.
        {'term': {'class_excluded': True}},
    ]
    if not include_candidate_clusters and candidate_cluster_id_min is not None:
        filt.append(
            {
                'bool': {
                    'should': [
                        {'bool': {'must_not': [{'exists': {'field': 'cluster_id'}}]}},
                        {'range': {'cluster_id': {'lt': candidate_cluster_id_min}}},
                    ],
                    'minimum_should_match': 1,
                },
            },
        )
    if extra_must:
        # Primary-subject clustering gate (rank / blur).
        filt.extend(extra_must)
    query = {'bool': {'filter': filt, 'must_not': must_not}}

    # Pre-count for progress.
    total_estimate = 0
    if progress is not None:
        try:
            count_resp = await client.count(index=items_index(), body={'query': query})
            total_estimate = int(count_resp.get('count', 0))
            progress.update(processed=0, total=total_estimate)
        except Exception as exc:
            logger.debug('curation_embedding_count_failed', error=str(exc))

    # Create a Point-In-Time. Without PIT we'd need scroll cursors per
    # slice -- PIT is the supported pattern for sliced parallel reads.
    try:
        pit_resp = await client.create_pit(
            index=items_index(),
            keep_alive=PARALLEL_FETCH_PIT_KEEPALIVE,
        )
        pit_id = pit_resp.get('pit_id') or pit_resp.get('pit')
        if not pit_id:
            raise RuntimeError(f'create_pit returned no pit_id: {pit_resp!r}')
    except Exception as exc:
        logger.warning('curation_embedding_pit_unavailable_fallback_scroll', error=str(exc))
        return await fetch_residual_embeddings(
            client,
            include_candidate_clusters=include_candidate_clusters,
            candidate_cluster_id_min=candidate_cluster_id_min,
            extra_must=extra_must,
            progress=progress,
        )

    # Shared progress counter across slices; integer mutation under the
    # GIL is atomic enough for an int-bucket counter.
    counter = {'n': 0}
    lock = asyncio.Lock()

    async def _emit_progress(added: int) -> None:
        if progress is None:
            return
        async with lock:
            counter['n'] += added
            progress.update(processed=counter['n'], total=total_estimate or counter['n'])
            progress.raise_if_cancelled()

    async def _fetch_one_slice(slice_id: int) -> tuple[list[str], list[np.ndarray]]:
        ids_local: list[str] = []
        embs_local: list[np.ndarray] = []
        search_after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                'size': page_size,
                '_source': [RESIDUAL_EMBEDDING_FIELD],
                'query': query,
                'pit': {'id': pit_id, 'keep_alive': PARALLEL_FETCH_PIT_KEEPALIVE},
                # _shard_doc is a synthetic sort key OpenSearch exposes
                # specifically for PIT pagination; doesn't require a
                # mapping change. Same sort across all slices keeps the
                # cursors comparable.
                'sort': [{'_shard_doc': 'asc'}],
            }
            if n_slices > 1:
                body['slice'] = {'id': slice_id, 'max': n_slices}
            if search_after is not None:
                body['search_after'] = search_after

            resp = await client.search(body=body)
            hits = resp.get('hits', {}).get('hits') or []
            if not hits:
                break
            page_ids, page = hits_to_rows(hits, RESIDUAL_EMBEDDING_FIELD)
            if page is not None:
                ids_local.extend(page_ids)
                embs_local.append(page)
            await _emit_progress(len(hits))
            # search_after cursor = sort key of the last hit. Required
            # to continue paging within this slice.
            last_sort = hits[-1].get('sort')
            if not last_sort:
                break
            search_after = last_sort
        return ids_local, embs_local

    try:
        slice_results = await asyncio.gather(
            *(_fetch_one_slice(i) for i in range(n_slices)),
        )
    finally:
        try:
            await client.delete_pit(body={'pit_id': [pit_id]})
        except OpenSearchException as exc:
            logger.debug('curation_embedding_pit_delete_failed', error=str(exc))

    ids: list[str] = []
    embs: list[np.ndarray] = []
    for sl_ids, sl_embs in slice_results:
        ids.extend(sl_ids)
        embs.extend(sl_embs)

    if not embs:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, _normalize_rows(stack_pages(embs))
