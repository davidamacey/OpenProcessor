"""2-D UMAP visualization-only projection overlay: ``POST
{prefix}/viz/projection/rebuild``,
``GET {prefix}/viz/projection/status``, ``POST {prefix}/viz/projection/cancel``,
``GET {prefix}/viz/projection``.

New module (not touching ``review.py`` / ``pipeline.py`` — both already
at or near the 700-LOC pre-commit ceiling). Side-effect import:
registers ``@router`` handlers on the shared ``_common.router``.

This is a **visualization-only** overlay, distinct from and never sharing
state with the retired clustering UMAP path
(``umap.py``/``embedding_reduce.py``, mounted at
``POST {prefix}/cluster/umap/rebuild``) — see
:mod:`src.services.curation.embedding_viz`'s module docstring for the full
non-negotiable-rules list (own state slot, never fits on a request path,
color decorates from the served ``cluster_uid`` rather than inventing a
second one, additive-only fields).

Gated end-to-end by ``OP_VIZ_PROJECTION_ENABLED`` (default off, read with
``env_bool``). Both endpoints 400 when the flag is
off, same convention this codebase uses for other disabled-by-default
feature flags.

**Ship-mode status** — see ``strategy_flags.py``'s ``viz_projection``
entry for how the measured purity maps to this entry's
``status``/``purity``/``requires_banner`` fields.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config.env import env_bool, env_int
from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._item_filter_params import ItemFilterQuery
from src.routers.curation._latest_job_models import VizProjectionCancelResponse, VizProjectionState
from src.services.curation.image_join import resolve_filter_join
from src.services.curation.item_filter import item_filter_clauses
from src.services.embedding_spaces.readers import item_write_fields


DEFAULT_MAX_N = 20_000


def _viz_enabled() -> bool:
    """Read fresh each call (not a module constant) — same reasoning as
    every other feature flag in this codebase: tests
    ``monkeypatch.setenv`` without reimporting."""
    return env_bool('OP_VIZ_PROJECTION_ENABLED')


def _max_n() -> int:
    try:
        return env_int('OP_VIZ_MAX_N', DEFAULT_MAX_N, minimum=1)
    except ValueError:
        return DEFAULT_MAX_N


@router.post(
    '/viz/projection/rebuild',
    status_code=202,
    response_model=None,
    responses={202: {'model': VizProjectionState}},
)
async def viz_projection_rebuild(
    session: DbSessionDep,
    scope: str = Query('residual', pattern='^(residual|cluster)$'),
    cluster_uid: uuid.UUID | None = Query(None),
) -> Any:
    """Queue the UMAP-viz fit job. Never fits inline: answers `202` with the job's state (`409
    job_busy` while one is active); poll ``GET {prefix}/viz/projection/status``."""
    if not _viz_enabled():
        raise HTTPException(
            status_code=400,
            detail='UMAP viz projection is disabled (set OP_VIZ_PROJECTION_ENABLED=1 to enable)',
        )
    if scope == 'cluster' and cluster_uid is None:
        raise HTTPException(status_code=400, detail='cluster_uid is required when scope=cluster')

    from src.services.curation import embedding_viz

    await embedding_viz.start(
        session, current_project_id(), scope=scope, cluster_uid=cluster_uid, max_n=_max_n()
    )
    return JSONResponse(status_code=202, content=await _state(session))


async def _state(session: Any) -> dict[str, Any]:
    from src.services.curation import embedding_viz
    from src.services.curation.latest_job import latest_job

    return embedding_viz.state_dict(await latest_job(session, embedding_viz.KIND))


@router.get('/viz/projection/status', response_model=VizProjectionState)
async def viz_projection_status(session: DbSessionDep) -> dict[str, Any]:
    return await _state(session)


@router.post('/viz/projection/cancel', response_model=VizProjectionCancelResponse)
async def viz_projection_cancel(session: DbSessionDep) -> dict[str, Any]:
    from src.services.curation import embedding_viz
    from src.services.curation.latest_job import cancel_latest

    cancelled = await cancel_latest(session, embedding_viz.KIND)
    return {'cancelled': cancelled, **await _state(session)}


class ProjectionPoint(BaseModel):
    crop_id: str | None = None
    x: float | None = None
    y: float | None = None
    cluster_uid: str | None = None
    class_name: str | None = None
    class_source: str | None = None
    # False when the shared filter leaves the point out; the scatter draws it dimmed.
    matched: bool = True
    # The legend slot of the point's class; the frontend maps a slot to a theme colour.
    colour_index: int | None = None


class ProjectionLegendEntry(BaseModel):
    """One class of the scatter: ``class_name`` is null for points with no class."""

    class_name: str | None = None
    colour_index: int
    count: int


class ProjectionResponse(BaseModel):
    """``GET /viz/projection`` 200 body (cached coordinates)."""

    points: list[ProjectionPoint]
    legend: list[ProjectionLegendEntry]
    projection_version: str | None = None
    fitted_at: str | None = None
    stale: bool


class ProjectionError(BaseModel):
    """404 / 503 body: ``error`` is the machine-readable discriminator."""

    error: Literal['projection_not_built', 'projection_unavailable']
    message: str


@router.get(
    '/viz/projection',
    response_model=ProjectionResponse,
    responses={404: {'model': ProjectionError}, 503: {'model': ProjectionError}},
)
async def viz_projection(
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    max_points: int = Query(50_000, ge=1, le=200_000),
) -> Any:
    """Serve cached ``viz_x``/``viz_y`` coordinates only — never triggers
    a fit (see ``embedding_viz_read.get_cached_projection``'s docstring and
    ``tests/curation/test_curation_viz_router.py::
    test_get_projection_never_imports_or_calls_fit`` for the enforcement).
    The shared item filter marks the points it leaves out ``matched: false`` (they are not
    dropped); ``legend`` lists the classes of the served points with their colour slots.
    ``404`` ``projection_not_built``: nothing has been fit yet (rebuild it).
    ``503`` ``projection_unavailable``: the read failed; a projection may
    exist, so retry rather than rebuild.
    """
    if not _viz_enabled():
        raise HTTPException(
            status_code=400,
            detail='UMAP viz projection is disabled (set OP_VIZ_PROJECTION_ENABLED=1 to enable)',
        )

    from src.services.curation import embedding_viz_read

    try:
        clauses = item_filter_clauses(
            item_filter,
            embedding=await item_write_fields(),
            image_join=await resolve_filter_join(opensearch, item_filter),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return await embedding_viz_read.get_cached_projection(
            opensearch, filter_clauses=clauses, max_points=max_points
        )
    except embedding_viz_read.ProjectionNotBuiltError:
        return JSONResponse(
            status_code=404,
            content=ProjectionError(
                error='projection_not_built',
                message='no projection has been built yet; POST /viz/projection/rebuild',
            ).model_dump(),
        )
    except embedding_viz_read.ProjectionUnavailableError:
        return JSONResponse(
            status_code=503,
            content=ProjectionError(
                error='projection_unavailable',
                message='the projection could not be read; retry shortly',
            ).model_dump(),
        )


__all__ = ['DEFAULT_MAX_N']
