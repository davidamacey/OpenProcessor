"""The FastAPI path dependency that binds a project for the rest of a
request (§3.2). It **must** be ``async def`` -- a sync dependency runs
in a threadpool, and a ``ContextVar`` set there never reaches the
endpoint (see ``src.config.project_context``'s module docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Path, Request
from fastapi.responses import JSONResponse

from src.config.project_context import set_bound_project, try_current_project
from src.config.projects import ProjectRecord  # noqa: TC001 - resolved at runtime by FastAPI
from src.core.logging import get_logger
from src.routers.curation._config_common_models import ConfigErrorDetail, api_error
from src.services.projects.registry import get_project_registry


if TYPE_CHECKING:
    from fastapi import FastAPI


logger = get_logger(__name__)

# M2: a read-only bind (archived, or a stale registry) refuses every
# non-safe method BEFORE the handler runs -- the guard's ProjectReadOnly
# stays as defence in depth for the OpenSearch calls, but file-backed
# writes (e.g. POST /classes rewriting class_registry.json) never went
# through the guard at all, so they need this earlier gate.
_SAFE_METHODS = frozenset({'GET', 'HEAD', 'OPTIONS'})


async def _resolve_and_bind(slug: str) -> ProjectRecord:
    registry = get_project_registry()
    record = await registry.lookup(slug)
    if record is None or record.status == 'deleted':
        raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)
    if record.status == 'building':
        raise api_error(
            409, 'project_building', f"project '{slug}' is still being created", project=slug
        )
    if record.status == 'deleting':
        raise api_error(409, 'project_deleting', f"project '{slug}' is being deleted", project=slug)
    if record.status == 'failed':
        # Its index set may be half-created: never bind it, not even to read.
        raise api_error(
            409, 'project_failed', f"project '{slug}' failed to build; delete it", project=slug
        )
    # P1R minor 10: if the last ensure_fresh() failed, this snapshot's
    # `status` may be stale (e.g. another instance flipped this project
    # active -> deleting after our last successful read) -- bind
    # read-only rather than trust a possibly-stale `active`.
    read_only = record.status == 'archived' or registry.stale
    set_bound_project(record, read_only=read_only)
    return record


async def bind_path_project(request: Request, project: Annotated[str, Path()]) -> ProjectRecord:
    """A malformed slug names no project: 404 ``project_not_found`` like
    any unknown slug, never a bare 422.

    M2: a read-only bind refuses every non-safe method here, before the
    route handler runs -- a file-backed write (``class_registry.json``,
    a settings snapshot, ...) never reaches the OpenSearch guard at all,
    so relying on :class:`ProjectReadOnly` alone let those through on an
    archived or stale-registry project."""
    record = await _resolve_and_bind(project)
    if request.method not in _SAFE_METHODS:
        bound = try_current_project()
        if bound is not None and bound.read_only:
            raise api_error(
                409,
                'project_archived' if record.status == 'archived' else 'project_read_only',
                f"project '{project}' is bound read-only; refusing a {request.method}",
                project=project,
            )
    return record


def install_project_exception_handlers(app: FastAPI) -> None:
    """Map the isolation exceptions to structured responses (§2.4): a
    cross-project access or an unbound project-scoped read is a server
    bug -> 500 ``internal_isolation_error`` (never partial data); a write
    to a read-only binding -> 409 ``project_read_only``."""
    from src.config.project_context import ProjectNotBound
    from src.services.projects.guard import CrossProjectAccess, ProjectReadOnly

    async def _isolation_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.error('internal_isolation_error', error_type=type(exc).__name__, error=str(exc))
        detail = ConfigErrorDetail(
            error='internal_isolation_error',
            message='The request was refused to keep projects isolated.',
        )
        return JSONResponse(status_code=500, content={'detail': detail.model_dump()})

    async def _read_only(_request: Request, exc: Exception) -> JSONResponse:
        bound = try_current_project()
        detail = ConfigErrorDetail(
            error='project_read_only',
            message=str(exc),
            project=bound.record.slug if bound is not None else None,
        )
        return JSONResponse(status_code=409, content={'detail': detail.model_dump()})

    app.add_exception_handler(CrossProjectAccess, _isolation_error)
    app.add_exception_handler(ProjectNotBound, _isolation_error)
    app.add_exception_handler(ProjectReadOnly, _read_only)
