"""Primary-subject clustering gate: coverage check and parking of the complement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.curation import items_index
from src.core.logging import get_logger
from src.services.curation.clustering.id_normalize import run_update_by_query_polled
from src.services.curation.embedding_state import embedded_clause


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


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
