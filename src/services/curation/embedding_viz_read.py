"""Read side of the UMAP viz projection: ``GET /viz/projection`` serves
cached ``viz_x``/``viz_y`` coordinates only and never fits (split out of
:mod:`src.services.curation.embedding_viz` for the file-size ceiling)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import NotFoundError

from src.config.curation import items_index, umap_viz_state_index
from src.core.logging import get_logger
from src.services.curation.embedding_state import embedded_clause
from src.services.curation.embedding_viz import EMBEDDING_FIELD


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

_VIZ_PROJECTION_PAGE_SIZE = 5000


class ProjectionNotBuiltError(Exception):
    """No projection has been fit yet (the state document does not exist)."""


class ProjectionUnavailableError(Exception):
    """The projection state could not be read (OpenSearch failure); it may exist."""


async def _load_run_metadata(opensearch: AsyncOpenSearch) -> dict[str, Any]:
    """The latest fit's state document.

    Raises :class:`ProjectionNotBuiltError` only when the document is
    genuinely absent; any other read failure is
    :class:`ProjectionUnavailableError`, so "never built" and "read failed"
    stay distinguishable.
    """
    try:
        resp = await opensearch.get(index=umap_viz_state_index(), id='current')
    except NotFoundError as exc:
        raise ProjectionNotBuiltError from exc
    except Exception as exc:
        logger.warning('curation_umap_viz_state_read_failed', error=str(exc))
        raise ProjectionUnavailableError(str(exc)) from exc
    meta = resp.get('_source')
    if not meta:
        raise ProjectionNotBuiltError
    return meta


def _point_from_hit(h: dict[str, Any]) -> dict[str, Any]:
    src = h.get('_source') or {}
    return {
        'crop_id': h['_id'],
        'x': src.get('viz_x'),
        'y': src.get('viz_y'),
        'cluster_id': src.get('cluster_id'),
        'class_name': src.get('class_name'),
        'class_source': src.get('class_source'),
    }


async def get_cached_projection(
    opensearch: AsyncOpenSearch,
    *,
    cluster_id: int | None = None,
    class_id: int | None = None,
    max_points: int = 50_000,
) -> dict[str, Any]:
    """Serve **cached coordinates only** — imports nothing UMAP-related,
    calls no fit function, does one plain ``search`` over already-written
    ``viz_x``/``viz_y`` fields. This is the entire GET
    ``/curation/viz/projection`` contract (module docstring point 2).

    Raises :class:`ProjectionNotBuiltError` when no projection has ever been
    fit and :class:`ProjectionUnavailableError` when the state read failed
    (also for a failed points search). Otherwise returns ``{points, projection_version, fitted_at, stale}`` where
    ``stale`` is True iff there exist crops matching the requested scope
    (embedding present, not test_holdout) whose cached
    ``viz_projection_version`` doesn't match the latest fit's version --
    i.e. some in-scope crops are missing/outdated coordinates, the same
    "partial coverage is visible" philosophy ``/curation/scores/coverage`` uses.
    """
    meta = await _load_run_metadata(opensearch)

    scope_must: list[dict[str, Any]] = [embedded_clause(EMBEDDING_FIELD)]
    scope_must_not: list[dict[str, Any]] = [{'term': {'test_holdout': True}}]
    if cluster_id is not None:
        scope_must.append({'term': {'cluster_id': cluster_id}})
    if class_id is not None:
        scope_must.append({'term': {'class_id': class_id}})

    points_query = {
        'bool': {
            'filter': [*scope_must, {'exists': {'field': 'viz_x'}}],
            'must_not': scope_must_not,
        }
    }
    points: list[dict[str, Any]] = []
    search_after: list[Any] | None = None
    while len(points) < max_points:
        page_size = min(_VIZ_PROJECTION_PAGE_SIZE, max_points - len(points))
        body: dict[str, Any] = {
            'size': page_size,
            'query': points_query,
            '_source': ['viz_x', 'viz_y', 'cluster_id', 'class_name', 'class_source'],
            'sort': [{'crop_id': 'asc'}],
            'track_total_hits': False,
        }
        if search_after is not None:
            body['search_after'] = search_after
        try:
            resp = await opensearch.search(index=items_index(), body=body)
        except Exception as exc:
            logger.warning('curation_viz_projection_points_read_failed', error=str(exc))
            raise ProjectionUnavailableError(str(exc)) from exc
        hits = resp.get('hits', {}).get('hits') or []
        if not hits:
            break
        points.extend(_point_from_hit(h) for h in hits)
        if len(hits) < page_size:
            break
        search_after = hits[-1]['sort']

    stale = False
    try:
        missing_query = {
            'bool': {
                'filter': scope_must,
                'must_not': [
                    *scope_must_not,
                    {'term': {'viz_projection_version': meta.get('projection_version', '')}},
                ],
            }
        }
        count_resp = await opensearch.count(index=items_index(), body={'query': missing_query})
        stale = int(count_resp.get('count', 0)) > 0
    except Exception as exc:
        logger.warning('curation_viz_projection_staleness_check_failed', error=str(exc))

    return {
        'points': points,
        'projection_version': meta.get('projection_version'),
        'fitted_at': meta.get('fitted_at'),
        'stale': stale,
    }


__all__ = ['ProjectionNotBuiltError', 'ProjectionUnavailableError', 'get_cached_projection']
