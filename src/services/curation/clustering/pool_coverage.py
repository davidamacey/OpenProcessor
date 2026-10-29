"""Whether a cluster pass covered the pool it was meant to cluster (#357).

Another writer (the cluster-refresh daemon assigning a new project's items, ingest
assigning at write time) can take items out of the strict residual pool between the
moment the pass counts it and the moment it reads it, or before the caller started it.
A pass that clustered a fraction of the pool is not a success: its centroids and
assignments describe the fraction, and a caller that trusts ``success`` records the
project as trained.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


# Fraction of the expected pool a pass must have read to count as covering it. The slack
# is for items that left the pool through a legitimate path while the pass ran.
POOL_COVERAGE_MIN = 0.99

PARTIAL = 'partial'


def covers_pool(n_fetched: int, expected: int) -> bool:
    """True when ``n_fetched`` reaches :data:`POOL_COVERAGE_MIN` of ``expected``."""
    return n_fetched >= math.ceil(expected * POOL_COVERAGE_MIN)


def coverage_fields(n_fetched: int, expected: int) -> dict[str, Any]:
    """The wire fields that say how much of the pool a pass read."""
    return {
        'n_residuals': n_fetched,
        'expected_pool': expected,
        'coverage': round(n_fetched / expected, 4) if expected else 1.0,
    }


async def count_pool(
    client: AsyncOpenSearch, query: dict[str, Any], expected: int | None = None
) -> int:
    """``expected`` when the caller knows its pool, else the documents the residual-pool
    ``query`` matches right now (a failed count raises)."""
    if expected is not None:
        return expected
    resp = await client.count(index=items_index(), body={'query': query})
    return int(resp.get('count', 0))


def short_read_result(
    *, n_fetched: int, expected_pool: int, method: str, mode: str, backend_info: Any
) -> dict[str, Any]:
    """The result of a pass that stopped because it read too little of its pool."""
    logger.warning(
        'curation_cluster_pass_partial',
        reason='short_read',
        n_residuals=n_fetched,
        expected_pool=expected_pool,
    )
    return {
        'status': PARTIAL,
        'reason': 'short_read',
        'method': method,
        'mode': mode,
        **coverage_fields(n_fetched, expected_pool),
        'n_clusters': 0,
        'n_noise': 0,
        'cluster_id_offset': RESIDUAL_CLUSTER_ID_OFFSET,
        'backend': backend_info.name,
        'backend_detail': backend_info.detail,
    }


def write_outcome(n_failed: int) -> tuple[str, dict[str, Any]]:
    """``(status, extra fields)`` of a pass whose write-back failed ``n_failed`` items."""
    if not n_failed:
        return 'success', {}
    logger.warning('curation_cluster_pass_partial', reason='write_failures', n_failed=n_failed)
    return PARTIAL, {'reason': 'write_failures', 'n_failed_writes': n_failed}
