"""2-D UMAP visualization-only projection overlay: ``POST
{prefix}/viz/projection/rebuild``,
``GET {prefix}/viz/projection/status``, ``POST {prefix}/viz/projection/cancel``,
``GET {prefix}/viz/projection``.

New module (not touching ``review.py`` / ``pipeline.py`` — both already
at or near the 700-LOC pre-commit ceiling). Side-effect import:
registers ``@router`` handlers on the shared ``_common.router``.

This is a **visualization-only** overlay, distinct from and never sharing
state with the retired clustering UMAP path
(``curation_umap.py``/``embedding_reduce.py``, mounted at
``POST {prefix}/cluster/umap/rebuild``) — see
:mod:`src.services.curation.embedding_viz`'s module docstring for the full
non-negotiable-rules list (own state slot, never fits on a request path,
color decorates from the real ``cluster_id`` rather than inventing a
second one, additive-only fields).

Gated end-to-end by ``OP_VIZ_PROJECTION_ENABLED`` (default off, inline
``os.getenv`` per house convention). Both endpoints 400 when the flag is
off, same convention this codebase uses for other disabled-by-default
feature flags.

**Ship-mode status** — see ``strategy_registry.py``'s ``viz_projection``
entry for how the measured purity maps to this entry's
``status``/``purity``/``requires_banner`` fields.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.routers.curation._common import OpenSearchDep, _ensure_indexes, router


DEFAULT_MAX_N = 20_000


def _viz_enabled() -> bool:
    """Read fresh each call (not a module constant) — same reasoning as
    every other feature flag in this codebase: tests
    ``monkeypatch.setenv`` without reimporting."""
    return os.environ.get('OP_VIZ_PROJECTION_ENABLED', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def _max_n() -> int:
    try:
        return max(1, int(os.environ.get('OP_VIZ_MAX_N', str(DEFAULT_MAX_N))))
    except ValueError:
        return DEFAULT_MAX_N


@router.post('/viz/projection/rebuild')
async def viz_projection_rebuild(
    opensearch: OpenSearchDep,
    scope: str = Query('residual', pattern='^(residual|cluster)$'),
    cluster_id: int | None = Query(None),
) -> Any:
    """Start the background UMAP-viz fit job. Never fits inline — always
    returns immediately with the job's initial ``{status: 'running', ...}``
    state (``202``); poll ``GET {prefix}/viz/projection/status``."""
    if not _viz_enabled():
        raise HTTPException(
            status_code=400,
            detail='UMAP viz projection is disabled (set OP_VIZ_PROJECTION_ENABLED=1 to enable)',
        )
    if scope == 'cluster' and cluster_id is None:
        raise HTTPException(status_code=400, detail='cluster_id is required when scope=cluster')
    await _ensure_indexes(opensearch)

    from src.services.curation import embedding_viz

    try:
        state = embedding_viz.start_job(
            opensearch, scope=scope, cluster_id=cluster_id, max_n=_max_n()
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(status_code=202, content=state)


@router.get('/viz/projection/status')
async def viz_projection_status() -> dict[str, Any]:
    from src.services.curation import embedding_viz

    return embedding_viz.get_state()


@router.post('/viz/projection/cancel')
async def viz_projection_cancel() -> dict[str, Any]:
    from src.services.curation import embedding_viz

    cancelled = embedding_viz.cancel_job()
    return {'cancelled': cancelled, **embedding_viz.get_state()}


class ProjectionPoint(BaseModel):
    crop_id: str | None = None
    x: float | None = None
    y: float | None = None
    cluster_id: int | None = None
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
    cluster_id: int | None = Query(None),
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
    await _ensure_indexes(opensearch)

    from src.services.curation import embedding_viz_read

    try:
        return await embedding_viz_read.get_cached_projection(
            opensearch,
            cluster_id=cluster_id,
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
