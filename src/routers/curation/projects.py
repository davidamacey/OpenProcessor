"""``GET /projects`` and ``GET /projects/{project}`` (§4 rows 1 and 3).
P1 scope only -- create/patch/archive/clone_settings/delete/stats are P3.
Mounted on ``global_router`` (never scoped).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query

from src.config import IndexRole
from src.config.projects import DEFAULT_SLUG, PROJECT_SLUG_RE
from src.core.logging import get_logger
from src.routers.curation._config_common_models import api_error
from src.routers.curation._project_models import (
    ProjectCounts,
    ProjectLimits,
    ProjectRecordResponse,
    ProjectsResponse,
    capacity_wire,
    list_membership,
    resources_wire,
    summarize,
)
from src.services.projects.guard import bind_registry_admin, make_curation_opensearch
from src.services.projects.registry import get_project_registry


logger = get_logger(__name__)

global_router = APIRouter(tags=['Projects'])


async def _fetch_counts(client: Any, snapshot: dict[str, Any]) -> dict[str, ProjectCounts]:
    """One ``_cat/indices`` call covering every project's images/items
    index (§4), the one cross-project read the guard allows, inside
    :func:`bind_registry_admin`. ``validated`` is not computed in this
    pass (it needs a per-project term query on the items index) and is
    served as ``null``."""
    index_names: set[str] = set()
    for record in snapshot.values():
        index_names.add(record.resources.indexes[IndexRole.IMAGES])
        index_names.add(record.resources.indexes[IndexRole.ITEMS])
    if not index_names:
        return {}
    pattern = ','.join(sorted(index_names))
    try:
        with bind_registry_admin():
            rows = await client.transport.perform_request(
                'GET',
                f'/_cat/indices/{pattern}',
                params={'h': 'index,docs.count', 'format': 'json'},
            )
    except Exception as exc:
        logger.warning('project_counts_unavailable', error=str(exc))
        rows = []
    doc_counts = {
        row['index']: int(row.get('docs.count') or 0) for row in rows if isinstance(row, dict)
    }
    result: dict[str, ProjectCounts] = {}
    for slug, record in snapshot.items():
        images_idx = record.resources.indexes[IndexRole.IMAGES]
        items_idx = record.resources.indexes[IndexRole.ITEMS]
        result[slug] = ProjectCounts(
            images=doc_counts.get(images_idx, 0),
            items=doc_counts.get(items_idx, 0),
        )
    return result


@global_router.get('/projects', response_model=ProjectsResponse)
async def list_projects(
    include_archived: Annotated[bool, Query()] = False,
) -> ProjectsResponse:
    registry = get_project_registry()
    await registry.ensure_fresh()
    snapshot = dict(registry.snapshot())

    listed = {
        slug: record
        for slug, record in snapshot.items()
        if list_membership(record.status, include_archived=include_archived)
    }

    client: Any = None
    counts: dict[str, ProjectCounts] = {}
    try:
        client = await make_curation_opensearch()
        counts = await _fetch_counts(client, listed)
    except Exception as exc:
        logger.warning('project_counts_unavailable', error=str(exc))
    summaries = [
        summarize(record, counts.get(slug, ProjectCounts()))
        for slug, record in sorted(listed.items())
    ]

    capacity = None
    try:
        from src.services.projects.capacity import capacity_status

        if client is not None:
            capacity = capacity_wire(await capacity_status(client))
    except Exception:
        capacity = None

    retired = sorted(slug for slug, record in snapshot.items() if record.status == 'deleted')
    return ProjectsResponse(
        default_slug=DEFAULT_SLUG,
        projects=summaries,
        capacity=capacity,
        limits=ProjectLimits(retired_slugs=retired),
        include_archived=include_archived,
    )


@global_router.get('/projects/{project}', response_model=ProjectRecordResponse)
async def get_project(
    project: Annotated[str, Path(pattern=PROJECT_SLUG_RE)],
) -> ProjectRecordResponse:
    registry = get_project_registry()
    await registry.ensure_fresh()
    record = registry.get(project)
    if record is None or record.status == 'deleted':
        raise api_error(404, 'project_not_found', f"no project named '{project}'", project=project)

    counts = ProjectCounts()
    try:
        client = await make_curation_opensearch()
        counts = (await _fetch_counts(client, {project: record})).get(project, counts)
    except Exception as exc:
        logger.warning('project_counts_unavailable', project=project, error=str(exc))
    summary = summarize(record, counts)
    return ProjectRecordResponse(
        **summary.model_dump(),
        resources=resources_wire(record.resources),
        error=None,
    )
