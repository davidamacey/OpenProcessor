"""Read side of the UMAP viz projection: ``GET /viz/projection`` serves
cached ``viz_x``/``viz_y`` coordinates only and never fits (split out of
:mod:`src.services.curation.embedding_viz` for the file-size ceiling)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException
from sqlalchemy.exc import SQLAlchemyError

from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.db.repos import settings as settings_repo
from src.db.session import async_session_factory, project_tx
from src.domain.embedding_spaces import Target
from src.domain.embedding_state import embedded_clause
from src.services.config_store.bound_project import bound_project_id
from src.services.embedding_spaces.resolve import embedding_space


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

_VIZ_PROJECTION_PAGE_SIZE = 5000


class ProjectionNotBuiltError(Exception):
    """No projection has been fit yet (the state document does not exist)."""


class ProjectionUnavailableError(Exception):
    """The projection state could not be read (OpenSearch failure); it may exist."""


async def _load_run_metadata() -> dict[str, Any]:
    """The latest fit's view state.

    Raises :class:`ProjectionNotBuiltError` only when the project has no state row; any other
    read failure is :class:`ProjectionUnavailableError`, so "never built" and "read failed" stay
    distinguishable.
    """
    try:
        project_id = await bound_project_id()
        async with async_session_factory()() as session, project_tx(session, project_id):
            row = await settings_repo.get_projection_state(session, project_id)
    except SQLAlchemyError as exc:
        logger.warning('curation_umap_viz_state_read_failed', error=str(exc))
        raise ProjectionUnavailableError(str(exc)) from exc
    if row is None:
        raise ProjectionNotBuiltError
    return {
        'projection_version': f'umap_viz_v{row.projection_version}',
        'scope': row.scope,
        'cluster_uid': None if row.cluster_uid is None else str(row.cluster_uid),
        'n_points': row.n_points,
        'fitted_at': row.fitted_at.isoformat(),
        'n_components': row.n_components,
        'metric': row.metric,
    }


def _point_from_hit(h: dict[str, Any]) -> dict[str, Any]:
    src = h.get('_source') or {}
    return {
        'crop_id': h['_id'],
        'x': src.get('viz_x'),
        'y': src.get('viz_y'),
        'cluster_uid': src.get('cluster_uid'),
        'class_name': src.get('class_name'),
        'class_source': src.get('class_source'),
        'matched': True,
    }


async def _matched_ids(
    opensearch: AsyncOpenSearch, clauses: list[dict[str, Any]], crop_ids: list[str]
) -> set[str]:
    """The ``crop_ids`` of one page of points that the filter ``clauses`` also match."""
    body = {
        'size': len(crop_ids),
        'query': {'bool': {'filter': [*clauses, {'terms': {'crop_id': crop_ids}}]}},
        '_source': False,
        'track_total_hits': False,
    }
    try:
        resp = await opensearch.search(index=items_index(), body=body)
    except OpenSearchException as exc:
        logger.warning('curation_viz_projection_match_read_failed', error=str(exc))
        raise ProjectionUnavailableError(str(exc)) from exc
    return {str(h['_id']) for h in resp.get('hits', {}).get('hits') or []}


def _legend(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The classes of ``points`` by size, each with the colour slot its points use.

    A point with no class (or an empty one) is the ``class_name: None`` entry. The slot is the
    entry's rank, so the biggest class gets slot 0; the frontend maps a slot to a theme colour.
    """
    counts = Counter(p['class_name'] or None for p in points)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0] is None, kv[0] or ''))
    slots = {name: i for i, (name, _) in enumerate(ranked)}
    for p in points:
        p['colour_index'] = slots[p['class_name'] or None]
    return [{'class_name': n, 'colour_index': slots[n], 'count': c} for n, c in ranked]


async def get_cached_projection(
    opensearch: AsyncOpenSearch,
    *,
    filter_clauses: list[dict[str, Any]] | None = None,
    max_points: int = 50_000,
) -> dict[str, Any]:
    """Serve **cached coordinates only** — imports nothing UMAP-related,
    calls no fit function, does plain ``search`` reads over already-written
    ``viz_x``/``viz_y`` fields. This is the entire GET
    ``/curation/viz/projection`` contract (module docstring point 2).

    Every embedded, non-holdout item with coordinates is a point. ``filter_clauses`` (the shared
    item filter's clauses, built by the route) only decide ``matched``: a second
    ``terms crop_id`` pass over each page of points keeps the matching ones, so the points the
    filter leaves out are served with ``matched: false`` (drawn dimmed), never dropped, and the
    pass is bounded by ``max_points`` like the points.

    Raises :class:`ProjectionNotBuiltError` when no projection has ever been
    fit and :class:`ProjectionUnavailableError` when the state read failed
    (also for a failed points search). Otherwise returns ``{points, legend, projection_version,
    fitted_at, stale}`` where ``stale`` is True iff there exist crops (embedding present, not
    test_holdout) whose cached ``viz_projection_version`` doesn't match the latest fit's version
    -- i.e. some crops are missing/outdated coordinates, the same "partial coverage is visible"
    philosophy ``/curation/scores/coverage`` uses.
    """
    meta = await _load_run_metadata()

    embedding_field = (await embedding_space(Target.ITEM)).vector_field
    scope_must: list[dict[str, Any]] = [embedded_clause(embedding_field)]
    scope_must_not: list[dict[str, Any]] = [{'term': {'test_holdout': True}}]

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
            '_source': ['viz_x', 'viz_y', 'cluster_uid', 'class_name', 'class_source'],
            'sort': [{'crop_id': 'asc'}],
            'track_total_hits': False,
        }
        if search_after is not None:
            body['search_after'] = search_after
        try:
            resp = await opensearch.search(index=items_index(), body=body)
        except OpenSearchException as exc:
            logger.warning('curation_viz_projection_points_read_failed', error=str(exc))
            raise ProjectionUnavailableError(str(exc)) from exc
        hits = resp.get('hits', {}).get('hits') or []
        if not hits:
            break
        page = [_point_from_hit(h) for h in hits]
        if filter_clauses:
            kept = await _matched_ids(opensearch, filter_clauses, [p['crop_id'] for p in page])
            for p in page:
                p['matched'] = p['crop_id'] in kept
        points.extend(page)
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
    except OpenSearchException as exc:
        logger.warning('curation_viz_projection_staleness_check_failed', error=str(exc))

    return {
        'points': points,
        'legend': _legend(points),
        'projection_version': meta.get('projection_version'),
        'fitted_at': meta.get('fitted_at'),
        'stale': stale,
    }


__all__ = ['ProjectionNotBuiltError', 'ProjectionUnavailableError', 'get_cached_projection']
