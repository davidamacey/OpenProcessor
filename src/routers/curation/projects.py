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

from fastapi import APIRouter, Depends, Query, Response

from src.config import IndexRole
from src.config.project_context import bind_project
from src.config.projects import DEFAULT_SLUG
from src.core.logging import get_logger
from src.routers.curation._config_common_models import ApiErrorResponse, api_error
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
from src.services.projects.guard import bind_registry_admin, make_curation_opensearch
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

# m11: a strong reference for delete's fire-and-forget finish task, so it
# is never garbage-collected mid-run (a documented asyncio caveat) --
# discarded automatically once the task completes. P3F pass-3 MA1
# probe 2: keyed by slug (not a bare set) so a re-DELETE issued while a
# finish for the SAME slug is still running (the M4 retry path) never
# schedules a second, redundant finish task racing the first one --
# `delete_project_finish` itself also refuses a concurrent run for the
# same slug (`delete._FINISH_IN_PROGRESS`), so this is belt-and-braces
# against wasting a task, not the only guard.
_BACKGROUND_DELETE_TASKS: dict[str, asyncio.Task[None]] = {}


def _publish_lifecycle_event(event_type: str, record: Any) -> None:
    """M5 step 9: every project.* lifecycle event on the global stream
    (never scoped -- these routes act *on* a project, not *within* one),
    so any open Cropwright tab (not just the one that made the request)
    learns about create/patch/archive/unarchive/delete."""
    from src.services.curation.event_hub import publish_global_event

    publish_global_event(
        event_type,
        target=record.slug,
        status=record.status,
        revision=record.revision,
    )


async def _fetch_counts(client: Any, snapshot: dict[str, Any]) -> dict[str, ProjectCounts]:
    """One ``_cat/indices`` call covering every project's images/items
    index (§4), the one cross-project read the guard allows, inside
    :func:`bind_registry_admin`; plus one ``validated`` count per project
    under that project's own read-only binding (``null`` when it could
    not be counted, the same rule ``/stats`` uses)."""
    from src.services.projects.stats import validated_count

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
    # TODO(P3F m12): one validated_count `count` query per project here
    # is N+1 on top of the single `_cat` call above. A real fix batches
    # it into one aggregation query (bucket by `_index`, term-filtered
    # on class_validated) across every project's items index, the same
    # `_cat` pattern already builds -- deferred this pass: it needs a
    # cross-index terms aggregation the existing fakes (FakeLifecycleOpenSearch
    # and the leak sweep's _FakeTransport) don't model, so verifying it
    # wouldn't be a real red->green fix in the time this pass allows.
    # Acceptable per finish-pass input 6; low severity (project counts
    # are small-cardinality, cached-adjacent reads, not a hot path).
    for slug, record in snapshot.items():
        images_idx = record.resources.indexes[IndexRole.IMAGES]
        items_idx = record.resources.indexes[IndexRole.ITEMS]
        with bind_project(record, read_only=True):
            validated = await validated_count(client, items_idx)
        result[slug] = ProjectCounts(
            images=doc_counts.get(images_idx, 0),
            items=doc_counts.get(items_idx, 0),
            validated=validated,
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
    project: str,
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
    record: Any,
    warnings: list[dict[str, str]] | None = None,
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
    _publish_lifecycle_event('project.created', record)
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
    _publish_lifecycle_event('project.updated', record)
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
    _publish_lifecycle_event('project.archived', record)
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
    _publish_lifecycle_event('project.unarchived', record)
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
    _publish_lifecycle_event('project.updated', record)
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
    dry_run: Annotated[bool, Query()] = False,
    confirm: Annotated[str | None, Query()] = None,
    force: Annotated[bool, Query()] = False,
) -> DeleteDryRunResponse | ProjectLifecycleResponse:
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
    record = await lifecycle.delete_project(client, slug=project, confirm=confirm, force=force)

    async def _finish() -> None:
        try:
            finished = await lifecycle.delete_project_finish(client, slug=project)
        except Exception as exc:
            logger.error('project_delete_finish_failed', project=project, error=str(exc))
        else:
            # M5 step 9: the completion signal Cropwright polls for
            # (docstring above; delta 10) -- published only once the
            # tombstone write itself succeeded, never on a busy/failed
            # retry (M3/M4 leave the record retryable with no event).
            _publish_lifecycle_event('project.deleted', finished)

    # MA1 probe 2 / M11: only schedule a new finish task for this slug if
    # none is already running -- a re-DELETE on an already-'deleting'
    # record (the M4 retry path, just above) must not race a second
    # finish against the first one's still-in-flight drain wait.
    # M11 (unstarted background task with no strong reference can be
    # GC'd mid-run): held on the router module so it survives until it
    # completes, and discarded from the map once done (only if this
    # exact task is still the one mapped -- a stale done-callback must
    # never evict a newer task that replaced it).
    existing_task = _BACKGROUND_DELETE_TASKS.get(project)
    if existing_task is None or existing_task.done():
        task = asyncio.create_task(_finish())
        _BACKGROUND_DELETE_TASKS[project] = task

        def _discard(finished_task: asyncio.Task[None], *, _slug: str = project) -> None:
            if _BACKGROUND_DELETE_TASKS.get(_slug) is finished_task:
                _BACKGROUND_DELETE_TASKS.pop(_slug, None)

        task.add_done_callback(_discard)
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
