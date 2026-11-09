"""Dataset stats endpoints for the curation labeler dashboard.

Two endpoints; the aggregations and rollups live in
:mod:`src.services.curation.stats_classes` and :mod:`src.services.curation.stats_dataset`:

* ``GET /curation/stats/classes`` — per-class total/validated breakdown.
* ``GET /curation/stats/dataset`` — overall dataset balance + label-source +
  pipeline breakdown. Single ``_search?size=0`` powering the labeler's
  pipeline dashboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the annotation
from src.db.session import async_session_factory
from src.routers._errors import server_error
from src.routers.curation._common import OpenSearchDep, get_class_registry, router
from src.routers.curation._item_filter_params import ItemFilterQuery  # noqa: TC001 - FastAPI
from src.routers.curation._stats_models import DatasetStatsResponse
from src.services.curation.item_filter import ItemFilter
from src.services.curation.stats_classes import ClassRegistryUnavailableError, class_stats
from src.services.curation.stats_dataset import build_dataset_stats
from src.services.curation.stats_dataset_query import InvalidItemFilterError, StatsQueryError


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@router.get('/stats/classes')
async def stats_classes(opensearch: OpenSearchDep, item_filter: ItemFilterQuery) -> dict[str, Any]:
    """Per-class total/validated breakdown.

    Returns both the raw ``by_class`` aggregation and a flattened ``classes``
    array (``{class_id, class_name, count, validated_count, trainable,
    trainable_gap, adequacy, aug_target, aug_gap}``, plus the ``thresholds``
    behind them) joined against the registry for the names. The labeler's
    ``getStats()`` (Export + Home pages) consumes ``classes`` — the
    aggregation alone has no class names, so without this join the Export
    dataset table renders empty.

    ``trainable`` = validated minus frozen test-holdout minus human-excluded
    crops — the frontend used to compute ``validated - holdout`` itself
    client-side; this serves the real number (also excluding
    ``class_excluded`` crops, which the client-side version didn't
    account for) so every page agrees. ``trainable_gap`` is the shortfall
    against :func:`dataset_thresholds`'s per-class minimum, floored at 0.
    """
    try:
        return await class_stats(opensearch, item_filter, get_class_registry)
    except InvalidItemFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ClassRegistryUnavailableError as exc:
        raise server_error(503, 'class registry unavailable', exc) from exc
    except StatsQueryError as exc:
        raise server_error(503, 'opensearch error', exc) from exc


async def stats_dataset(opensearch: Any) -> dict[str, Any]:
    """Unfiltered dataset stats (what the live stats stream serves)."""
    async with async_session_factory()() as session:
        return await dataset_stats(opensearch, ItemFilter(), session)


@router.get('/stats/dataset', response_model=None, responses={200: {'model': DatasetStatsResponse}})
async def stats_dataset_route(
    opensearch: OpenSearchDep, item_filter: ItemFilterQuery, session: DbSessionDep
) -> dict[str, Any]:
    """``GET /stats/dataset`` over the items the shared item filter selects (every
    item when no filter parameter is set); see :func:`dataset_stats`."""
    return await dataset_stats(opensearch, item_filter, session)


async def dataset_stats(
    opensearch: Any, item_filter: ItemFilter, session: AsyncSession
) -> dict[str, Any]:
    """Overall dataset balance + label-source + pipeline breakdown over the items
    ``item_filter`` selects; see :func:`~src.services.curation.stats_dataset.build_dataset_stats`
    for the payload."""
    from src.services.triton_control import TritonControlService

    try:
        return await build_dataset_stats(
            opensearch,
            item_filter,
            session,
            repository_index=TritonControlService().get_repository_index,
        )
    except InvalidItemFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StatsQueryError as exc:
        raise server_error(503, 'opensearch error', exc) from exc
