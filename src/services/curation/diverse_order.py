"""Diversity / core-set selection: the ``order=diverse`` ranking behind ``GET
/curation/crops`` and the budgets ``POST /curation/select/diverse`` shares.

k-center-greedy (:func:`src.services.curation.selection.k_center_greedy`,
Sener & Savarese ICLR 2018) over crop ``pe_embedding`` vectors. This is a
pure **selection** overlay — like every ``crop_scores`` scorer, it never
writes ``cluster_id``/``cluster_subid``/``cluster_distance`` or any other
crop field; unlike a scorer it doesn't write anything to OpenSearch *at
all* — it only ever reads embeddings and returns an ordering/subset of
ids.

**Validation status:** the cheap pre-screen passed on real data
(k-center-greedy vs. random vs. cluster-representatives, 1.50x
label-coverage ratio, bar >= 1.3x) — that unblocked writing this module.
The **full gate** (a real training A/B via ``/curation/bakeoff``,
mAP50-95 >= +1.0pt) has NOT run. Until it does, ``diverse`` stays
``experimental`` at most in ``strategy_registry.py`` — never promotable
to ``stable`` by this module or by flipping ``OP_SELECT_DIVERSE_ENABLED``
alone.

Gated end-to-end by ``OP_SELECT_DIVERSE_ENABLED`` (default off, read with
``env_bool``):

* ``GET /curation/crops?order=diverse`` with the flag off behaves exactly like
  today's handling of any other unrecognized ``order`` value — silently
  ignored, default sort — because the crops router's ``order`` branch simply
  never matches when this module reports "not enabled" (see
  :func:`compute_diverse_order` — same early-return shape
  ``compute_outlier_order`` uses for "too large", just for "disabled").
* ``POST /curation/select/diverse`` with the flag off returns ``400`` (matches
  ``scores.py``'s ``scores_compute`` convention for its own disabled
  flag, not a bespoke 404).

**Compute budget & the sync-vs-job split** (based on a 93.6s
@ n=30,000,k=5,000 real benchmark):

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
   ``None`` and the crops router falls through to its default sort — the
   *exact* same fallback contract ``order=outliers`` already uses above
   its own cap (mirrored, not reinvented).
2. ``POST /curation/select/diverse`` is user-``k``-bounded (O(n*k*d), linear in
   k, not quadratic) — this is the "give me 1,000 diverse crops out of a
   30,000-crop cluster" pool-scale case, measured at ~1-2 min CPU at
   n~128k/k~1000, i.e. too slow to block an HTTP request.
   ``OP_SELECT_SYNC_MAX_OPS`` (default 3,000,000 — derived from the real
   93.6s-at-n*k=1.5e8 k-center-greedy benchmark, budgeting for a ~2s
   wall-clock ceiling on an inline request: 1.5e8 selections / 93.6s ~=
   1.6e6 selections/sec => 2s budget ~= 3.2e6) decides sync vs. job: if
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
   ~124,920 residual, chosen so a background job can actually answer the
   "whole residual pool" use case, which the 20,000 sync cap would
   otherwise make impossible to ever service).

Judgment call: ``OP_SELECT_MAX_N=20000`` is treated as *the sync-path
fetch cap* (matching its literal ``cluster_outliers._MAX_MEMBERS``
precedent, used by both the GET path and as the "is this small enough to
even consider answering inline" gate for POST); ``OP_SELECT_SYNC_MAX_OPS``
/``OP_SELECT_JOB_MAX_N`` are two additional constants that resolve the
tension between "20,000 is the sync cap" and "diversity may be requested
over a 30,000-crop cluster" — see above.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from src.config.env import env_bool, env_float, env_int
from src.services.curation.selection import k_center_greedy
from src.services.curation.selection.pool_fetch import (
    EMBEDDING_FIELD,
    fetch_pool_embeddings,
    l2_normalize,
)


DEFAULT_MAX_N = 20_000
DEFAULT_SYNC_MAX_OPS = 3_000_000
DEFAULT_JOB_MAX_N = 200_000


def diverse_enabled() -> bool:
    """Read fresh each call (not a module constant) — same reasoning as
    every other feature flag in this codebase (e.g.
    ``strategy_registry._scores_enabled``): tests ``monkeypatch.setenv``
    without reimporting."""
    return env_bool('OP_SELECT_DIVERSE_ENABLED')


def max_n() -> int:
    try:
        return env_int('OP_SELECT_MAX_N', DEFAULT_MAX_N, minimum=1)
    except ValueError:
        return DEFAULT_MAX_N


def sync_max_ops() -> int:
    try:
        return env_int('OP_SELECT_SYNC_MAX_OPS', DEFAULT_SYNC_MAX_OPS, minimum=1)
    except ValueError:
        return DEFAULT_SYNC_MAX_OPS


def job_max_n() -> int:
    try:
        return env_int('OP_SELECT_JOB_MAX_N', DEFAULT_JOB_MAX_N, minimum=1)
    except ValueError:
        return DEFAULT_JOB_MAX_N


def _get_diverse_inline_max() -> int:
    """Full-pool ranking is O(n^2*d) (k==n) — solve n^2 <= sync-ops-budget
    for n. Also bounded above by the fetch cap (never relevant in
    practice: sqrt of the default op budget is ~1,732, far below the
    20,000 fetch cap)."""
    return max(1, min(max_n(), math.isqrt(sync_max_ops())))


_ORDER_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_TTL_S = env_float('OP_SELECT_CACHE_TTL_S', 600)


def _cache_key(index: str, query: dict[str, Any], k: int | None = None) -> str:
    return f'diverse|{index}|{EMBEDDING_FIELD}|k={k}|{json.dumps(query, sort_keys=True)}'


async def compute_diverse_order(
    client: Any,
    index: str,
    query: dict[str, Any],
    *,
    current_count: int | None = None,
    k: int | None = None,
) -> list[str] | None:
    """Return matching crop ``_id``s ordered by k-center-greedy selection
    order (index 0 = the seed, the rest in decreasing "how much new
    coverage did picking this add" order) — every match, or only the
    first ``k`` picks when ``k`` is given.

    Returns ``None`` when diversity selection is disabled, the pool
    exceeds the inline cap, or there are no embeddings to rank — callers
    (the reference crops router) fall back to their default sort in every ``None``
    case, exactly mirroring ``compute_outlier_order``'s contract.
    """
    if not diverse_enabled():
        return None

    inline_max = _get_diverse_inline_max()
    key = _cache_key(index, query, k)
    now = time.monotonic()
    cached = _ORDER_CACHE.get(key)
    if (
        cached is not None
        and (now - cached['at']) < _CACHE_TTL_S
        and (current_count is None or cached['count'] == current_count)
    ):
        return cached['order']

    ids, embeddings, truncated = await fetch_pool_embeddings(client, index, query, cap=inline_max)
    if truncated:
        return None
    if not ids:
        return []

    # Cache 'count' must be the pool size (len(ids)), the same value
    # a fresh call compares against via current_count — NOT len(order),
    # which is capped at k and so (for k < pool size, the common case)
    # could never match current_count on a later lookup, making the cache
    # permanently miss.
    pool_count = len(ids)
    n_pick = len(ids) if k is None else min(k, len(ids))
    order_idx = k_center_greedy(l2_normalize(embeddings), n_pick)
    order = [ids[i] for i in order_idx.tolist()]
    _ORDER_CACHE[key] = {'order': order, 'count': pool_count, 'at': now}
    return order
