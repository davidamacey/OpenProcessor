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

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response

from src.config import IndexRole
from src.config.project_context import bind_project
from src.config.projects import DEFAULT_SLUG
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves route annotations at runtime
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.routers.curation._api_error import api_error
from src.routers.curation._project_deps import bind_path_project
from src.routers.curation._project_views import (
    capacity_wire,
    list_membership,
    resources_wire,
    summarize,
)
from src.schemas.curation.config_common import ApiErrorResponse
from src.schemas.curation.projects import (
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
)
from src.services.projects import lifecycle
from src.services.projects.guard import make_curation_opensearch
from src.services.projects.registry import get_project_registry


logger = get_logger(__name__)

# Every error here is api_error()'s typed body. A path slug is matched as
# a plain string: a malformed slug names no project, so it is a 404
# project_not_found like any unknown slug, never a 422.
global_router = APIRouter(
    tags=['Projects'],
    responses={
        404: {'model': ApiErrorResponse, 'description': 'project_not_found'},
        409: {'model': ApiErrorResponse, 'description': 'Refused (see detail.error)'},
    },
)


async def _publish_lifecycle_event(event_type: str, record: Any, **extra: Any) -> None:
    """M5 step 9: every project.* lifecycle event on the global stream
    (never scoped -- these routes act *on* a project, not *within* one),
    so any open Cropwright tab (not just the one that made the request)
    learns about create/patch/archive/unarchive/delete."""
    from src.services.events.publish import apublish_global_event

    await apublish_global_event(
        event_type,
        target=record.slug,
        status=record.status,
        revision=record.revision,
        **extra,
    )


async def _fetch_counts(client: Any, snapshot: dict[str, Any]) -> dict[str, ProjectCounts]:
    """Per project, under that project's own read-only binding: how many
    images and items it holds (``_count``, which counts top-level documents --
    ``_cat/indices`` ``docs.count`` also counts every nested region box and
    box embedding, and skips what an idle shard has not refreshed yet) and
    how many are validated (``null`` when it could not be counted, the same
    rule ``/stats`` uses)."""
    from src.services.projects.stats import embedded_count, index_count, validated_count

    result: dict[str, ProjectCounts] = {}
    for slug, record in snapshot.items():
        images_idx = record.resources.indexes[IndexRole.IMAGES]
        items_idx = record.resources.indexes[IndexRole.ITEMS]
        with bind_project(record, read_only=True):
            images = await index_count(client, images_idx)
            items = await index_count(client, items_idx)
            validated = await validated_count(client, items_idx)
            embedded = await embedded_count(client, items_idx)
        result[slug] = ProjectCounts(
            images=images, items=items, validated=validated, items_embedded=embedded
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
    from src.services.projects.capacity import capacity_status

    if client is not None:
        capacity = capacity_wire(await capacity_status(client))

    retired = sorted(registry.retired_slugs())
    return ProjectsResponse(
        default_slug=DEFAULT_SLUG,
        projects=summaries,
        capacity=capacity,
        limits=ProjectLimits(retired_slugs=retired),
        include_archived=include_archived,
    )


@global_router.get('/projects/{project}', response_model=ProjectRecordResponse)
async def get_project(
    project: str,
) -> ProjectRecordResponse:
    record = await get_project_registry().lookup(project)
    if record is None:
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
    record: Any,
    warnings: list[dict[str, Any]] | None = None,
    keymap_clone_conflicts: list[dict[str, Any]] | None = None,
) -> ProjectLifecycleResponse:
    client = await make_curation_opensearch()
    counts = (await _fetch_counts(client, {record.slug: record})).get(record.slug, ProjectCounts())
    return ProjectLifecycleResponse(
        project=summarize(record, counts),
        warnings=warnings or [],
        keymap_clone_conflicts=keymap_clone_conflicts or [],
    )


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
    await _publish_lifecycle_event('project.created', record)
    return await _summary_response(record, warnings)


@global_router.patch('/projects/{project}', response_model=ProjectLifecycleResponse)
async def patch_project(
    project: str,
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
    await _publish_lifecycle_event('project.updated', record)
    return await _summary_response(record)


@global_router.post('/projects/{project}/archive', response_model=ProjectLifecycleResponse)
async def archive_project(
    project: str,
    body: ArchiveRequest,
) -> ProjectLifecycleResponse:
    client = await make_curation_opensearch()
    record = await lifecycle.archive_project(
        client, slug=project, expected_revision=body.expected_revision
    )
    await _publish_lifecycle_event('project.archived', record)
    return await _summary_response(record)


@global_router.post('/projects/{project}/unarchive', response_model=ProjectLifecycleResponse)
async def unarchive_project(
    project: str,
    body: ArchiveRequest,
) -> ProjectLifecycleResponse:
    client = await make_curation_opensearch()
    record = await lifecycle.unarchive_project(
        client, slug=project, expected_revision=body.expected_revision
    )
    await _publish_lifecycle_event('project.unarchived', record)
    return await _summary_response(record)


@global_router.post('/projects/{project}/clone_settings', response_model=ProjectLifecycleResponse)
async def clone_settings_route(
    project: str,
    body: CloneSettingsRequest,
) -> ProjectLifecycleResponse:
    client = await make_curation_opensearch()
    record, keymap_clone_conflicts = await lifecycle.clone_settings_into(
        client,
        slug=project,
        from_slug=body.from_,
        axes=body.axes,
        expected_revision=body.expected_revision,
    )
    await _publish_lifecycle_event('project.updated', record)
    return await _summary_response(record, keymap_clone_conflicts=keymap_clone_conflicts)


@global_router.delete(
    '/projects/{project}',
    response_model=None,
    responses={
        200: {'model': DeleteDryRunResponse, 'description': 'Dry run (writes nothing)'},
        202: {'model': ProjectLifecycleResponse, 'description': 'Delete accepted; finishing'},
    },
)
async def delete_project(
    project: str,
    response: Response,
    session: DbSessionDep,
    dry_run: Annotated[bool, Query()] = False,
    confirm: Annotated[str | None, Query()] = None,
    force: Annotated[bool, Query()] = False,
) -> DeleteDryRunResponse | ProjectLifecycleResponse:
    """Dry run (200, writes nothing) or a guarded delete. A real delete
    answers **202** with the ``deleting`` record (delta 10): the drain
    wait, index/dir removal and tombstone run in the background so this
    always returns well under a proxy read timeout, regardless of how
    long the drain takes. ``job_id`` names the ``project_delete`` job; follow
    ``project.deleted`` on the global stream, or poll ``GET /projects/{project}``
    -> 404, to know when it finished."""
    client = await make_curation_opensearch()
    if dry_run:
        report = await lifecycle.dry_run_delete(client, slug=project)
        return DeleteDryRunResponse(**report)
    record = await lifecycle.delete_project(slug=project, confirm=confirm, force=force)
    # Starting is idempotent on the project row: a re-DELETE of a `deleting` project (the retry
    # path) answers 202 and leaves the finish that is already queued or running to finish.
    try:
        started = await lifecycle.start_project_delete(session, project)
        job_id: str | None = str(started.job.id)
    except RequestRejectedError as exc:
        if exc.detail.get('error') != 'job_busy':
            raise
        job_id = exc.detail.get('active_job_id')
    response.status_code = 202
    summary = await _summary_response(record)
    summary.job_id = job_id
    return summary


@global_router.get('/projects/{project}/stats', response_model=ProjectStatsResponse)
async def get_project_stats(
    _project_record: Annotated[Any, Depends(bind_path_project)],
) -> ProjectStatsResponse:
    from src.services.projects.stats import project_stats

    client = await make_curation_opensearch()
    stats = await project_stats(client)
    return ProjectStatsResponse(**stats)
