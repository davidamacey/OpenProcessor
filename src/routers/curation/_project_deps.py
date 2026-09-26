"""FastAPI path dependencies that bind a project for the rest of a
request (§3.2). Both **must** be ``async def`` -- a sync dependency runs
in a threadpool, and a ``ContextVar`` set there never reaches the
endpoint (see ``src.config.project_context``'s module docstring).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Path

from src.config.project_context import set_bound_project
from src.config.projects import DEFAULT_SLUG, PROJECT_SLUG_RE, ProjectRecord
from src.routers.curation._config_common_models import api_error
from src.services.projects.registry import get_project_registry


async def _resolve_and_bind(slug: str) -> ProjectRecord:
    registry = get_project_registry()
    await registry.ensure_fresh()
    record = registry.get(slug)
    if record is None or record.status == 'deleted':
        raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)
    if record.status == 'building':
        raise api_error(
            409, 'project_building', f"project '{slug}' is still being created", project=slug
        )
    if record.status == 'deleting':
        raise api_error(409, 'project_deleting', f"project '{slug}' is being deleted", project=slug)
    read_only = record.status == 'archived'
    set_bound_project(record, read_only=read_only)
    return record


async def bind_path_project(
    project: Annotated[str, Path(pattern=PROJECT_SLUG_RE)],
) -> ProjectRecord:
    return await _resolve_and_bind(project)


async def bind_default_project() -> ProjectRecord:
    return await _resolve_and_bind(DEFAULT_SLUG)
