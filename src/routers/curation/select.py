"""Diversity / core-set selection overlay (curation-strategy plan
§2.6/§3.4/§7 Phase 4): ``GET /curation/crops?order=diverse``'s helper +
``POST /curation/select/diverse``.

k-center-greedy (:func:`src.services.curation.selection.k_center_greedy`,
Sener & Savarese ICLR 2018) over crop ``pe_embedding`` vectors. This is a
pure **selection** overlay — like every ``crop_scores`` scorer, it never
writes ``cluster_id``/``cluster_subid``/``cluster_distance`` or any other
crop field (plan §8 non-goal #3); unlike a scorer it doesn't write
anything to OpenSearch *at all* — it only ever reads embeddings and
returns an ordering/subset of ids.

**Validation status (docs/design/curation_scores.md §6):** the cheap
pre-screen passed on real data (k-center-greedy vs. random vs.
cluster-representatives, 1.50x label-coverage ratio, bar >= 1.3x) — that
unblocked writing this module. The **full gate** (a real training A/B via
``/curation/bakeoff``, mAP50-95 >= +1.0pt) has NOT run. Per plan §6/§10.2 this
means ``diverse`` stays ``experimental`` at most in ``strategy_registry.py``
— never promotable to ``stable`` by this module or by flipping
``OP_SELECT_DIVERSE_ENABLED`` alone.

Gated end-to-end by ``OP_SELECT_DIVERSE_ENABLED`` (default off, inline
``os.getenv`` per house convention):

* ``GET /curation/crops?order=diverse`` with the flag off behaves exactly like
  today's handling of any other unrecognized ``order`` value — silently
  ignored, default sort — because ``kb_crops.py``'s ``order`` branch simply
  never matches when this module reports "not enabled" (see
  :func:`compute_diverse_order` — same early-return shape
  ``compute_outlier_order`` uses for "too large", just for "disabled").
* ``POST /curation/select/diverse`` with the flag off returns ``400`` (matches
  ``scores.py``'s ``scores_compute`` convention for its own disabled
  flag, not a bespoke 404).

**Compute budget & the sync-vs-job split (plan §3 compute budget /
curation_scores.md §6's 93.6s @ n=30,000,k=5,000 real benchmark):**

k-center-greedy is O(n*k*d). Two very different call shapes need two very
different budgets:

1. ``GET /curation/crops?order=diverse`` needs a **full ranking of the whole
   pool** (exactly like ``compute_outlier_order``, so any page can be
   sliced out of one stable order) — i.e. ``k == n``, which makes the cost
   O(n^2*d). This is quadratic, not linear, so it needs a much smaller
   inline pool cap than ``OP_SELECT_MAX_N`` (20,000, the *fetch* cap that
   mirrors ``cluster_outliers._MAX_MEMBERS``'s precedent). Solving
   ``n^2 <= OP_SELECT_SYNC_MAX_OPS`` for ``n`` gives the GET-only inline
   cap (``_get_diverse_inline_max()`` below) — at the 3,000,000-op default
   that's ~1,732 crops, comfortably "a few thousand" and a couple of
   seconds of BLAS matvecs on CPU. Above that cap, the endpoint returns
   ``None`` and ``kb_crops.py`` falls through to its default sort — the
   *exact* same fallback contract ``order=outliers`` already uses above
   its own cap (mirrored, not reinvented).
2. ``POST /curation/select/diverse`` is user-``k``-bounded (O(n*k*d), linear in
   k, not quadratic) — this is the "give me 1,000 diverse crops out of a
   30,000-crop cluster" pool-scale case the plan's own compute budget
   flags as ~1-2 min CPU at n~128k/k~1000, i.e. too slow to block an HTTP
   request. ``OP_SELECT_SYNC_MAX_OPS`` (default 3,000,000 — derived from
   the real 93.6s-at-n*k=1.5e8 k-center-greedy benchmark in
   ``docs/design/curation_scores.md`` §6, budgeting for a ~2s wall-clock
   ceiling on an inline request: 1.5e8 selections / 93.6s ~= 1.6e6
   selections/sec => 2s budget ~= 3.2e6) decides sync vs. job: if
   ``n_pool * k <= OP_SELECT_SYNC_MAX_OPS`` (and the pool itself fit under
   ``OP_SELECT_MAX_N``, the fetch cap), answer inline with the documented
   ``{crop_ids, method, version, n_pool}`` body. Otherwise this starts a
   background job (:mod:`src.services.curation.selection.job`, mirroring
   ``crop_scores.job``'s singleton state.json/heartbeat/cancel.flag
   pattern) and returns ``202`` with ``{job_id, status: 'running', ...}``;
   poll ``GET /curation/select/status`` for the result. The job path fetches the
   pool uncapped by ``OP_SELECT_MAX_N`` (bounded only by the much larger
   ``OP_SELECT_JOB_MAX_N``, default 200,000 — a generous safety ceiling
   above the largest real pool this repo has ever seen, ~347,837 total /
   ~124,920 residual per ``docs/design/curation_scores.md`` §0, chosen so
   a background job can actually answer the "whole residual pool" use
   case the plan's compute-budget section describes, which the 20,000
   sync cap would otherwise make impossible to ever service).

Judgment call, not spelled out verbatim in the plan: the plan's §3 flag
table lists a single ``OP_SELECT_MAX_N=20000``. This module treats that
value as *the sync-path fetch cap* (matching its literal
``cluster_outliers._MAX_MEMBERS`` precedent, used by both the GET path and
as the "is this small enough to even consider answering inline" gate for
POST) and introduces ``OP_SELECT_SYNC_MAX_OPS``/``OP_SELECT_JOB_MAX_N`` as
two new, undocumented-in-the-plan constants to resolve the tension between
"20,000 is the sync cap" and "the plan's own POST example asks for
diversity over a 30,000-crop cluster" — see module docstring above.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    _ensure_indexes,
    router,
)
from src.services.curation import review_queries
from src.services.curation.selection import k_center_greedy
from src.services.curation.selection.pool_fetch import (
    EMBEDDING_FIELD,
    fetch_pool_embeddings,
    l2_normalize,
)


DEFAULT_MAX_N = 20_000
DEFAULT_SYNC_MAX_OPS = 3_000_000
DEFAULT_JOB_MAX_N = 200_000


def _diverse_enabled() -> bool:
    """Read fresh each call (not a module constant) — same reasoning as
    every other feature flag in this codebase (e.g.
    ``strategy_registry._scores_enabled``): tests ``monkeypatch.setenv``
    without reimporting."""
    return os.environ.get('OP_SELECT_DIVERSE_ENABLED', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def _max_n() -> int:
    try:
        return max(1, int(os.environ.get('OP_SELECT_MAX_N', str(DEFAULT_MAX_N))))
    except ValueError:
        return DEFAULT_MAX_N


def _sync_max_ops() -> int:
    try:
        return max(1, int(os.environ.get('OP_SELECT_SYNC_MAX_OPS', str(DEFAULT_SYNC_MAX_OPS))))
    except ValueError:
        return DEFAULT_SYNC_MAX_OPS


def _job_max_n() -> int:
    try:
        return max(1, int(os.environ.get('OP_SELECT_JOB_MAX_N', str(DEFAULT_JOB_MAX_N))))
    except ValueError:
        return DEFAULT_JOB_MAX_N


def _get_diverse_inline_max() -> int:
    """Full-pool ranking is O(n^2*d) (k==n) — solve n^2 <= sync-ops-budget
    for n. Also bounded above by the fetch cap (never relevant in
    practice: sqrt of the default op budget is ~1,732, far below the
    20,000 fetch cap)."""
    return max(1, min(_max_n(), math.isqrt(_sync_max_ops())))


# =============================================================================
# GET /curation/crops?order=diverse helper
# =============================================================================


_ORDER_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_TTL_S = float(os.getenv('OP_SELECT_CACHE_TTL_S', '600'))


def _cache_key(index: str, query: dict[str, Any]) -> str:
    return f'diverse|{index}|{EMBEDDING_FIELD}|{json.dumps(query, sort_keys=True)}'


async def compute_diverse_order(
    client: Any,
    index: str,
    query: dict[str, Any],
    *,
    current_count: int | None = None,
) -> list[str] | None:
    """Return every matching crop's ``_id``, ordered by k-center-greedy
    selection order (index 0 = the seed, the rest in decreasing
    "how much new coverage did picking this add" order).

    Returns ``None`` when diversity selection is disabled, the pool
    exceeds the inline cap, or there are no embeddings to rank — callers
    (``kb_crops.py``) fall back to their default sort in every ``None``
    case, exactly mirroring ``compute_outlier_order``'s contract.
    """
    if not _diverse_enabled():
        return None

    inline_max = _get_diverse_inline_max()
    key = _cache_key(index, query)
    now = time.monotonic()
    cached = _ORDER_CACHE.get(key)
    if (
        cached is not None
        and (now - cached['at']) < _CACHE_TTL_S
        and (current_count is None or cached['count'] == current_count)
    ):
        return cached['order']  # type: ignore[no-any-return]

    ids, embeddings, truncated = await fetch_pool_embeddings(client, index, query, cap=inline_max)
    if truncated:
        return None
    if not ids:
        return []

    order_idx = k_center_greedy(l2_normalize(embeddings), len(ids))
    order = [ids[i] for i in order_idx.tolist()]
    _ORDER_CACHE[key] = {'order': order, 'count': len(order), 'at': now}
    return order


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
    k: int = Field(..., ge=1, le=50_000)
    seed_crop_id: str | None = None


def _build_scope_query(scope: SelectDiverseScope) -> dict[str, Any]:
    """Translate a ``SelectDiverseScope`` into an OpenSearch bool query.

    ``test_holdout=true`` crops are always excluded (plan §8 non-goal #12
    — no scope can opt back in; this is a training-selection-adjacent
    operation and must never leak the frozen holdout into what gets
    labeled next).
    """
    must: list[dict[str, Any]] = [{'exists': {'field': EMBEDDING_FIELD}}]
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
    Never writes anything to OpenSearch (read-only selection, plan §8
    non-goal #3).
    """
    if not _diverse_enabled():
        raise HTTPException(
            status_code=400,
            detail='diversity selection is disabled (set OP_SELECT_DIVERSE_ENABLED=1 to enable)',
        )
    await _ensure_indexes(opensearch)

    query = _build_scope_query(payload.scope)

    # Peek at the pool size first (a cheap count, not a fetch) so we can
    # decide sync-vs-job without paying for an embeddings fetch we might
    # throw away. Falling back to "assume large" on a count failure is the
    # safe direction — worst case we start an unnecessary job, never block
    # the request on an OpenSearch outage.
    try:
        count_resp = await opensearch.count(index=CURATION_ITEMS_INDEX, body={'query': query})
        n_estimate = int(count_resp.get('count', 0))
    except Exception:
        n_estimate = _job_max_n() + 1  # force the job path

    max_n = _max_n()
    sync_ops = _sync_max_ops()
    fits_sync_fetch = n_estimate <= max_n
    fits_sync_ops = n_estimate * payload.k <= sync_ops

    if fits_sync_fetch and fits_sync_ops:
        ids, embeddings, truncated = await fetch_pool_embeddings(
            opensearch, CURATION_ITEMS_INDEX, query, cap=max_n
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
            index=CURATION_ITEMS_INDEX,
            query=query,
            k=payload.k,
            seed_crop_id=payload.seed_crop_id,
            scope=payload.scope.model_dump(),
            max_n=_job_max_n(),
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


__all__ = [
    'DEFAULT_JOB_MAX_N',
    'DEFAULT_MAX_N',
    'DEFAULT_SYNC_MAX_OPS',
    'SelectDiverseRequest',
    'SelectDiverseScope',
    'compute_diverse_order',
]
