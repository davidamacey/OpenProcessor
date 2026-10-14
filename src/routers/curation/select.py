"""``POST /curation/select/diverse`` and its ``status`` / ``cancel`` lifecycle.

Pool-scale k-center-greedy selection over crop ``pe_embedding`` vectors; the
ranking, the flag and the compute budgets live in
:mod:`src.services.curation.diverse_order`. Read-only: nothing is written to
OpenSearch.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.domain.embedding_state import embedded_clause
from src.routers.curation._common import OpenSearchDep, items_index, router
from src.services.curation import review_queries
from src.services.curation.diverse_limits import DIVERSE_SELECT_MAX_K
from src.services.curation.diverse_order import diverse_enabled, job_max_n, max_n, sync_max_ops
from src.services.curation.index_bootstrap import ensure_indexes
from src.services.curation.selection import k_center_greedy
from src.services.curation.selection.pool_fetch import (
    EMBEDDING_FIELD,
    fetch_pool_embeddings,
    l2_normalize,
)


# =============================================================================
# POST /curation/select/diverse
# =============================================================================


class SelectDiverseScope(BaseModel):
    """Which crops the selection draws from. Every field optional;
    combining ``cluster_id`` + ``review_tab`` + ``filters`` intersects
    them (AND), matching ``GET /curation/crops``'s own filter-combination style."""

    cluster_id: int | None = None
    review_tab: str | None = Field(
        default=None,
        description='A GET /curation/review/{tab} tab id — reuses its match logic verbatim.',
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            'Generic exact-match term filters keyed by OpenSearch field name '
            '(e.g. {"class_id": 7}). List values become a `terms` clause.'
        ),
    )


class SelectDiverseRequest(BaseModel):
    scope: SelectDiverseScope = Field(default_factory=SelectDiverseScope)
    k: int = Field(..., ge=1, le=DIVERSE_SELECT_MAX_K)
    seed_crop_id: str | None = None


def _build_scope_query(scope: SelectDiverseScope) -> dict[str, Any]:
    """Translate a ``SelectDiverseScope`` into an OpenSearch bool query.

    ``test_holdout=true`` crops are always excluded — no scope can opt
    back in; this is a training-selection-adjacent
    operation and must never leak the frozen holdout into what gets
    labeled next).
    """
    must: list[dict[str, Any]] = [embedded_clause(EMBEDDING_FIELD)]
    must_not: list[dict[str, Any]] = [{'term': {'test_holdout': True}}]

    if scope.cluster_id is not None:
        must.append({'term': {'cluster_id': scope.cluster_id}})

    if scope.review_tab:
        # Reuses the exact tab match logic GET /curation/review/{tab} uses — no
        # duplicated/drifting definition of what a tab means. Raises
        # HTTPException(400) for an unknown tab, same as that endpoint.
        tab_must, tab_must_not, _reason = review_queries.build_tab_query(
            scope.review_tab, include_test=False, text=None, max_rank=None
        )
        must.extend(tab_must)
        must_not.extend(tab_must_not)

    if scope.filters:
        for field_name, value in scope.filters.items():
            if isinstance(value, list):
                must.append({'terms': {field_name: value}})
            else:
                must.append({'term': {field_name: value}})

    return {'bool': {'must': must, 'must_not': must_not}}


@router.post('/select/diverse')
async def select_diverse(
    payload: SelectDiverseRequest,
    opensearch: OpenSearchDep,
) -> Any:
    """Pool-scale diversity selection — k-center-greedy over an arbitrary
    scope (cluster / review tab / generic filters), bigger than what fits
    in one ``GET /curation/crops`` page.

    Answers inline (``200``, ``{crop_ids, method, version, n_pool}``) when
    the scoped pool is small enough per ``OP_SELECT_SYNC_MAX_OPS``;
    otherwise starts a background job and returns ``202`` with
    ``{job_id, status: 'running', ...}`` — poll ``GET /curation/select/status``.
    Never writes anything to OpenSearch (read-only selection).
    """
    if not diverse_enabled():
        raise HTTPException(
            status_code=400,
            detail='diversity selection is disabled (set OP_SELECT_DIVERSE_ENABLED=1 to enable)',
        )
    await ensure_indexes(opensearch)

    query = _build_scope_query(payload.scope)

    # Peek at the pool size first (a cheap count, not a fetch) so we can
    # decide sync-vs-job without paying for an embeddings fetch we might
    # throw away. Falling back to "assume large" on a count failure is the
    # safe direction — worst case we start an unnecessary job, never block
    # the request on an OpenSearch outage.
    try:
        count_resp = await opensearch.count(index=items_index(), body={'query': query})
        n_estimate = int(count_resp.get('count', 0))
    except Exception:
        n_estimate = job_max_n() + 1  # force the job path

    fetch_cap = max_n()
    sync_ops = sync_max_ops()
    fits_sync_fetch = n_estimate <= fetch_cap
    fits_sync_ops = n_estimate * payload.k <= sync_ops

    if fits_sync_fetch and fits_sync_ops:
        ids, embeddings, truncated = await fetch_pool_embeddings(
            opensearch, items_index(), query, cap=fetch_cap
        )
        if truncated:
            # The count-based estimate underestimated (race with concurrent
            # writes) — fall through to the job path rather than silently
            # answering over a partial pool.
            pass
        else:
            seed_idx: int | None = None
            if payload.seed_crop_id is not None and payload.seed_crop_id in ids:
                seed_idx = ids.index(payload.seed_crop_id)
            if not ids:
                return {'crop_ids': [], 'method': 'kcenter_greedy', 'version': 'v1', 'n_pool': 0}
            order_idx = k_center_greedy(l2_normalize(embeddings), payload.k, seed_idx=seed_idx)
            crop_ids = [ids[i] for i in order_idx.tolist()]
            return {
                'crop_ids': crop_ids,
                'method': 'kcenter_greedy',
                'version': 'v1',
                'n_pool': len(ids),
            }

    from src.services.curation.selection import job as select_job

    try:
        state = select_job.start_job(
            opensearch,
            index=items_index(),
            query=query,
            k=payload.k,
            seed_crop_id=payload.seed_crop_id,
            scope=payload.scope.model_dump(),
            max_n=job_max_n(),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(status_code=202, content=state)


@router.get('/select/status')
async def select_status() -> dict[str, Any]:
    """Poll a backgrounded ``POST /curation/select/diverse`` job. ``result`` is
    populated once ``status == 'completed'`` with the same
    ``{crop_ids, method, version, n_pool}`` shape the sync path returns
    directly."""
    from src.services.curation.selection import job as select_job

    return select_job.get_state()


@router.post('/select/cancel')
async def select_cancel() -> dict[str, Any]:
    from src.services.curation.selection import job as select_job

    cancelled = select_job.cancel_job()
    return {'cancelled': cancelled, **select_job.get_state()}


__all__ = ['SelectDiverseRequest', 'SelectDiverseScope']
