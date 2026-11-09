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

import uuid  # noqa: TC003 - FastAPI resolves the query type at runtime
from typing import Any, Literal

from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config.env import env_bool, env_int
from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._common import OpenSearchDep, router


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


@router.post('/viz/projection/rebuild', status_code=202)
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


@router.get('/viz/projection/status')
async def viz_projection_status(session: DbSessionDep) -> dict[str, Any]:
    return await _state(session)


@router.post('/viz/projection/cancel')
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


class ProjectionResponse(BaseModel):
    """``GET /viz/projection`` 200 body (cached coordinates)."""

    points: list[ProjectionPoint]
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
    cluster_uid: uuid.UUID | None = Query(None),
    class_id: int | None = Query(None),
    max_points: int = Query(50_000, ge=1, le=200_000),
) -> Any:
    """Serve cached ``viz_x``/``viz_y`` coordinates only — never triggers
    a fit (see ``embedding_viz_read.get_cached_projection``'s docstring and
    ``tests/curation/test_curation_viz_router.py::
    test_get_projection_never_imports_or_calls_fit`` for the enforcement).
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
        return await embedding_viz_read.get_cached_projection(
            opensearch,
            cluster_uid=cluster_uid,
            class_id=class_id,
            max_points=max_points,
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
