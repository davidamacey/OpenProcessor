"""The full §4 project lifecycle API: ``GET/POST /projects``,
``GET/PATCH {prefix}``, archive/unarchive/clone_settings, guarded delete
(dry-run + real, background-completing per delta 10), and the scoped
``/stats`` route. Everything except ``/stats`` is mounted on
``global_router`` (never scoped) -- lifecycle operations act *on* a
project, not *within* one, and (owner decision, 2026-09-26: no
backwards compatibility) there is no unscoped alias anywhere in this
module.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Response

from src.config import IndexRole
from src.config.projects import DEFAULT_SLUG, PROJECT_SLUG_RE
from src.core.logging import get_logger
from src.routers.curation._config_common_models import api_error
from src.routers.curation._project_deps import bind_path_project
from src.routers.curation._project_models import (
    ArchiveRequest,
    CloneSettingsRequest,
    CreateProjectRequest,
    DeleteDryRunResponse,
    PatchProjectRequest,
    ProjectCounts,
    ProjectLifecycleResponse,
    ProjectLimits,
    ProjectRecordResponse,
    ProjectsResponse,
    ProjectStatsResponse,
    capacity_wire,
    list_membership,
    resources_wire,
    summarize,
)
from src.services.projects import lifecycle
from src.services.projects.guard import make_curation_opensearch
from src.services.projects.registry import get_project_registry


logger = get_logger(__name__)

global_router = APIRouter(tags=['Projects'])


async def _fetch_counts(client: Any, snapshot: dict[str, Any]) -> dict[str, ProjectCounts]:
    """One ``_cat/indices`` call covering every project's images/items
    index (§4). ``validated`` is not computed in this pass -- it needs a
    per-project term query on the items index and is left at 0; a
    documented gap, not silently faked as accurate."""
    index_names: set[str] = set()
    for record in snapshot.values():
        index_names.add(record.resources.indexes[IndexRole.IMAGES])
        index_names.add(record.resources.indexes[IndexRole.ITEMS])
    if not index_names:
        return {}
    pattern = ','.join(sorted(index_names))
    try:
        rows = await client.transport.perform_request(
            'GET', f'/_cat/indices/{pattern}', params={'h': 'index,docs.count', 'format': 'json'}
        )
    except Exception:
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
            validated=0,
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

    client = await make_curation_opensearch()
    counts = await _fetch_counts(client, listed)
    summaries = [
        summarize(record, counts.get(slug, ProjectCounts()))
        for slug, record in sorted(listed.items())
    ]

    capacity = None
    try:
        from src.services.projects.capacity import capacity_status

        capacity_result = await capacity_status(client)
        capacity = capacity_wire(capacity_result)
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


async def _summary_response(
    record: Any, warnings: list[dict[str, str]] | None = None
) -> ProjectLifecycleResponse:
    client = await make_curation_opensearch()
    counts = (await _fetch_counts(client, {record.slug: record})).get(record.slug, ProjectCounts())
    return ProjectLifecycleResponse(project=summarize(record, counts), warnings=warnings or [])


@global_router.post('/projects', response_model=ProjectLifecycleResponse, status_code=201)
async def create_project(body: CreateProjectRequest) -> ProjectLifecycleResponse:
    client = await make_curation_opensearch()
    record, warnings = await lifecycle.create_project(
        client,
        slug=body.slug,
        display_name=body.display_name,
        description=body.description,
        clone_settings_from=body.clone_settings_from,
        clone_axes=body.clone_axes,
    )
    return await _summary_response(record, warnings)


@global_router.patch('/projects/{project}', response_model=ProjectLifecycleResponse)
async def patch_project(
    project: Annotated[str, Path(pattern=PROJECT_SLUG_RE)],
    body: PatchProjectRequest,
) -> ProjectLifecycleResponse:
    client = await make_curation_opensearch()
    record = await lifecycle.patch_project(
        client,
        slug=project,
        display_name=body.display_name,
        description=body.description,
        expected_revision=body.expected_revision,
    )
    return await _summary_response(record)


@global_router.post('/projects/{project}/archive', response_model=ProjectLifecycleResponse)
async def archive_project(
    project: Annotated[str, Path(pattern=PROJECT_SLUG_RE)],
    body: ArchiveRequest,
) -> ProjectLifecycleResponse:
    client = await make_curation_opensearch()
    record = await lifecycle.archive_project(
        client, slug=project, expected_revision=body.expected_revision
    )
    return await _summary_response(record)


@global_router.post('/projects/{project}/unarchive', response_model=ProjectLifecycleResponse)
async def unarchive_project(
    project: Annotated[str, Path(pattern=PROJECT_SLUG_RE)],
    body: ArchiveRequest,
) -> ProjectLifecycleResponse:
    client = await make_curation_opensearch()
    record = await lifecycle.unarchive_project(
        client, slug=project, expected_revision=body.expected_revision
    )
    return await _summary_response(record)


@global_router.post('/projects/{project}/clone_settings', response_model=ProjectLifecycleResponse)
async def clone_settings_route(
    project: Annotated[str, Path(pattern=PROJECT_SLUG_RE)],
    body: CloneSettingsRequest,
) -> ProjectLifecycleResponse:
    client = await make_curation_opensearch()
    record = await lifecycle.patch_project(
        client,
        slug=project,
        display_name=None,
        description=None,
        expected_revision=body.expected_revision,
    )
    await lifecycle.clone_settings(
        client, target_record=record, from_slug=body.from_, axes=body.axes
    )
    registry = get_project_registry()
    await registry.ensure_fresh()
    refreshed = registry.get(project) or record
    return await _summary_response(refreshed)


@global_router.delete('/projects/{project}')
async def delete_project(
    project: Annotated[str, Path(pattern=PROJECT_SLUG_RE)],
    response: Response,
    dry_run: Annotated[bool, Query()] = False,
    confirm: Annotated[str | None, Query()] = None,
    force: Annotated[bool, Query()] = False,
) -> Any:
    """Dry run (200, writes nothing) or a guarded delete. A real delete
    answers **202** with the ``deleting`` record (delta 10): the drain
    wait, index/dir removal and tombstone run in the background so this
    always returns well under a proxy read timeout, regardless of how
    long the drain takes. Follow ``project.deleted`` on the global
    stream, or poll ``GET /projects/{project}`` -> 404, to know when it
    finished."""
    client = await make_curation_opensearch()
    if dry_run:
        report = await lifecycle.dry_run_delete(client, slug=project)
        return DeleteDryRunResponse(**report)
    if confirm is None:
        raise api_error(422, 'confirm_mismatch', 'confirm is required for a real delete')
    record = await lifecycle.delete_project(client, slug=project, confirm=confirm, force=force)

    async def _finish() -> None:
        try:
            await lifecycle.delete_project_finish(client, slug=project)
        except Exception as exc:
            logger.error('project_delete_finish_failed', project=project, error=str(exc))

    asyncio.create_task(_finish())  # noqa: RUF006 - fire-and-forget delete completion (delta 10)
    response.status_code = 202
    return await _summary_response(record)


@global_router.get('/projects/{project}/stats', response_model=ProjectStatsResponse)
async def get_project_stats(
    _project_record: Annotated[Any, Depends(bind_path_project)],
) -> ProjectStatsResponse:
    from src.services.projects.stats import project_stats

    client = await make_curation_opensearch()
    stats = await project_stats(client)
    return ProjectStatsResponse(**stats)
