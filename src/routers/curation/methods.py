"""``GET /curation/methods`` — capability-discovery endpoint.

Separate module from ``review.py`` / ``pipeline.py``. Side-effect
import: registers the ``@router`` handler on the shared ``_common.router``.

Thin wrapper over :mod:`src.services.curation.strategy_registry`, which
adds a real, TTL-cached ``field_coverage`` OpenSearch lookup per entry
(:func:`strategy_registry._compute_field_coverage`) — this endpoint is no
longer pure/I/O-free, but the frontend's 404-fallback design still means it
must never itself be the reason a request fails, so a failure anywhere in
the coverage lookup degrades to ``field_coverage: None`` (or, in the
unlikely event ``get_registry`` itself raises against a broken client, the
un-annotated config/flag-only registry) rather than a 5xx.
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.routers.curation._common import OpenSearchDep, router


logger = get_logger(__name__)


@router.get('/methods')
async def get_methods(opensearch: OpenSearchDep) -> dict[str, Any]:
    """Every strategy across every axis (cluster / score / sort / overlay),
    plus the feature flags that gated each entry's status and a real
    ``field_coverage`` (exists-count) / ``field_coverage_total`` (pool
    size) per entry that declares a ``requires_field``. Frontend renders
    only ``stable``/``experimental`` entries; ``shadow``/``disabled`` are
    computed+logged (or not computed at all) but never offered as a
    ``?sort=`` value. Frontend also hides an entry whose ``field_coverage``
    is exactly ``0`` (genuinely inert — no data has ever been backfilled)
    while still showing one whose coverage is ``None`` (unknown, e.g. a
    transient OpenSearch failure) — see ``StrategyBar.svelte``'s
    ``hasFieldCoverage``.
    """
    from src.services.curation.strategy_registry import get_registry

    try:
        return await get_registry(opensearch)
    except Exception as exc:
        logger.warning('curation_methods_registry_failed', error=str(exc))
        return await get_registry(None)
