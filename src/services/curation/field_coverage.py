"""Per-field coverage counts for ``GET /curation/methods``.

TTL-cached per items index so the methods endpoint stays an O(1)-per-request
call; a failed count is cached as ``None`` (unknown), never ``0``.
"""

from __future__ import annotations

import time
from typing import Any

from src.config.env import env_float
from src.core.logging import get_logger


logger = get_logger(__name__)


# Keyed by the bound project's items index: one project's counts must
# never answer another project's ``/methods`` (projects_plan.md §2.1).
# Value: (fetched_at monotonic, counts).
_COVERAGE_CACHE: dict[str, tuple[float, dict[str, int | None]]] = {}
_COVERAGE_TTL_S = env_float('OP_FIELD_COVERAGE_TTL_S', 60)
_COVERAGE_TOTAL_KEY = '__total__'
"""Sentinel key the pool-size count is cached under, alongside the
per-field exists counts, in the same dict -- avoids a second cache
structure just to hold one extra number."""


async def _compute_field_coverage(opensearch: Any, fields: frozenset[str]) -> dict[str, int | None]:
    """``{field: exists_count}`` for every field in ``fields``, plus the
    pool total under :data:`_COVERAGE_TOTAL_KEY`. Mirrors
    ``crop_scores/job.py::compute_coverage``'s query shape (one
    ``opensearch.count`` per field, one ``match_all`` count for the
    denominator) but keyed by raw field name across every axis, not just
    the three ``crop_scores`` scorer fields -- ``/curation/scores/coverage`` has
    no answer for ``probe_pred_entropy``/``cluster_distance``/
    ``region_score``/``crop_area_norm``, which is exactly where the inert
    sorts live (see this module's Phase 6 note above).

    TTL-cached at module scope per items index (``_COVERAGE_CACHE`` / ``time.monotonic()``
    freshness check), the same pattern the select router's ``_ORDER_CACHE``
    and ``cluster_outliers.py``'s ``_CACHE`` use -- ``GET /curation/methods``
    must stay an O(1)-per-request endpoint (its own docstring promises it
    "never fails or blocks"), not an O(distinct-fields) OpenSearch round
    trip on every page load.

    A field whose count query fails is cached as ``None`` -- "unknown",
    never ``0`` ("empty") -- so a transient OpenSearch hiccup can never
    make a working sort/chip disappear ("fall back to None
    (not 0) on OpenSearch failure, so a transient error hides nothing that
    already works"). The whole helper never raises; callers get a dict
    with real ints, ``None``s, or (only if OpenSearch is totally
    unreachable) a ``None`` total too.

    On a cache miss this used to be N+1 round trips (one
    ``opensearch.count`` per field plus one for the total). Now it's a
    single ``_search`` (``size: 0``, ``track_total_hits: true``, one
    ``filter: {exists}`` sub-agg per field) -- same answer, one request.
    """
    from src.config.curation_accessors import get_curation_config
    from src.config.curation_roles import IndexRole, index_name

    index = index_name(get_curation_config(), IndexRole.ITEMS)
    now = time.monotonic()
    entry = _COVERAGE_CACHE.get(index)
    cache = entry[1] if entry is not None and (now - entry[0]) < _COVERAGE_TTL_S else None
    fresh = cache is not None
    if cache is not None and fields <= (cache.keys() - {_COVERAGE_TOTAL_KEY}):
        return cache

    # Callers ask for different field sets (GET /methods: every sort field;
    # a review tab: its own fallback chain), so a fresh cache is extended
    # rather than replaced -- otherwise they evict each other every call.
    counts: dict[str, int | None] = dict(cache) if cache is not None else {}
    missing = sorted(fields - counts.keys())
    need_total = cache is None
    try:
        resp = await opensearch.search(
            index=index,
            body={
                'size': 0,
                'track_total_hits': need_total,
                'aggs': {field: {'filter': {'exists': {'field': field}}} for field in missing},
            },
        )
        if need_total:
            counts[_COVERAGE_TOTAL_KEY] = int(
                ((resp.get('hits') or {}).get('total') or {}).get('value', 0)
            )
        aggs = resp.get('aggregations') or {}
        for field in missing:
            bucket = aggs.get(field)
            counts[field] = int(bucket['doc_count']) if bucket is not None else None
    except Exception as exc:
        logger.warning('curation_methods_field_coverage_failed', error=str(exc))
        if need_total:
            counts[_COVERAGE_TOTAL_KEY] = None
        for field in missing:
            counts[field] = None

    _COVERAGE_CACHE[index] = (entry[0] if fresh and entry is not None else now, counts)
    return counts


def invalidate_field_coverage() -> None:
    """Drop the bound project's cached coverage so its next lookup counts
    live."""
    from src.config.curation_accessors import get_curation_config
    from src.config.curation_roles import IndexRole, index_name

    _COVERAGE_CACHE.pop(index_name(get_curation_config(), IndexRole.ITEMS), None)


async def field_coverage(opensearch: Any, fields: frozenset[str]) -> dict[str, int | None]:
    """``{field: exists_count}`` over the items index (``None`` = unknown,
    never ``0``); TTL-cached. Public face of :func:`_compute_field_coverage`."""
    return await _compute_field_coverage(opensearch, fields)


def _reset_field_coverage_cache() -> None:
    """Test-only escape hatch -- the module-level TTL cache otherwise leaks
    across test cases that monkeypatch a fake OpenSearch client per-test."""
    _COVERAGE_CACHE.clear()
