"""Project lifecycle service: create / patch / archive / unarchive /
clone_settings / delete (dry-run and guarded).

See the project-isolation design §4, §7, the
owner decisions in §13 (D4 capacity, D5 ``default`` is undeletable), and
the Cropwright rev-3 deltas 2 (envelope + revision), 3 (list membership,
implemented in P1's projects wire models), 10 (delete fits a proxy
timeout), 11 (structured refusals) and 12 (full ``capacity`` on the
409). There is no unscoped alias and no back-compat shim anywhere in
this module (owner decision, 2026-09-26): every route this backs lives
only under the global router or a project's scoped prefix.
"""

from __future__ import annotations

from typing import Any

from src.config.project_context import bind_project
from src.config.projects import (
    ARCHIVABLE_STATUSES,
    UNARCHIVABLE_STATUSES,
    ProjectRecord,
    is_valid_slug,
)
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.db.repos import projects as repo
from src.services.projects.capacity import capacity_status
from src.services.projects.lifecycle_checks import (
    _last_active_check,
    _require_revision,
    _require_transition,
    busy_message,
    running_jobs,
)
from src.services.projects.records import new_project_for, record_from_row
from src.services.projects.registry import get_project_registry
from src.services.projects.store import create_with_default_spaces, get_row, run_repo
from src.services.rejections import rejection


logger = get_logger(__name__)


async def _capacity_error_or_warning(
    client: Any,
) -> list[dict[str, Any]]:
    """Runs the §2.3 capacity check for one more project's shards.
    Raises 409 ``shard_budget_exceeded`` when blocked; otherwise returns
    ``[{"code": "shard_budget_high", ...}]`` on warn, else ``[]``."""
    capacity = await capacity_status(client)
    if capacity is None:
        return []
    if capacity.status == 'blocked':
        raise rejection(
            409,
            'shard_budget_exceeded',
            capacity.message,
            active_shards=capacity.active_shards,
            needed=capacity.per_project_shards,
            soft_limit=capacity.soft_limit,
            hard_limit=capacity.hard_limit,
            heap_max_bytes=capacity.heap_max_bytes,
            capacity=capacity.to_wire(),
        )
    if capacity.status == 'warn':
        return [
            {
                'code': 'shard_budget_high',
                'message': capacity.message,
                'detail': capacity.to_wire(),
            }
        ]
    return []


async def _resolve_existing(slug: str) -> ProjectRecord | None:
    registry = get_project_registry()
    await registry.ensure_fresh()
    return registry.get(slug)


def _require_found(record: ProjectRecord | None, slug: str) -> ProjectRecord:
    if record is None:
        raise rejection(404, 'project_not_found', f"no project named '{slug}'", project=slug)
    return record


async def _get_mutable_record(slug: str) -> ProjectRecord:
    """The stored record, read fresh from the database (never the snapshot)."""
    return record_from_row(await get_row(slug))


async def _cleanup_orphaned_by_concurrent_delete(
    client: Any, record: ProjectRecord, exc: RequestRejectedError
) -> None:
    """A create's own ``building`` end (to ``active``, or ``failed`` after an error) can lose to a
    concurrent DELETE that already removed this project and retired its slug. Refusing is right (a
    retired slug must not come back), but this create may already have made its indexes, which no
    project-bound path can reach again. Best-effort: delete them here, and log the exact names at
    error level so an operator can find them if this cleanup fails too."""
    detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
    if detail.get('error') != 'project_not_found':
        return
    orphaned = sorted(set(record.resources.indexes.values()))
    logger.error('project_create_orphaned_after_delete', slug=record.slug, indexes=orphaned)
    for name in orphaned:
        try:
            with bind_project(record):
                await client.indices.delete(index=name, ignore=[404])
        except Exception as cleanup_exc:
            logger.warning(
                'project_create_orphan_cleanup_failed',
                slug=record.slug,
                index=name,
                error=str(cleanup_exc),
            )


async def create_project(
    client: Any,
    *,
    slug: str,
    display_name: str,
    description: str = '',
    clone_settings_from: str | None = None,
    clone_axes: list[str] | None = None,
    origin: dict | None = None,
    activate: bool = True,
) -> tuple[ProjectRecord, list[dict[str, Any]]]:
    """§4 ``POST /projects``. Steps: validate, capacity, row ``building`` (Postgres first), create
    indexes and dirs, optional clone, ``active``. A failure midway leaves the row ``failed``,
    never partially ``active``.

    ``origin`` is stored on the row (a combine's target names its job and sources).
    ``activate=False`` stops with the verified record still ``building`` for a caller that fills
    the project first and then calls :func:`finish_building`."""
    if not is_valid_slug(slug):
        raise rejection(422, 'slug_invalid', f"'{slug}' is not a valid project slug", project=slug)

    if clone_settings_from:
        # Every refusal a clone can raise -- unknown axis, clone into itself, a source that does
        # not exist or is not ready -- runs BEFORE the first write: a refused clone burns no slug,
        # holds no shards and leaves no dirs.
        from src.services.projects.clone_validate import _validate_clone_source

        await _validate_clone_source(
            client, target_slug=slug, from_slug=clone_settings_from, axes=clone_axes
        )

    warnings = await _capacity_error_or_warning(client)

    # The slug claim inside the create transaction is the one duplicate check: a second creator of
    # the same slug waits for the first and loses with 409 slug_taken (or slug_retired).
    row = await run_repo(
        'create a project',
        create_with_default_spaces,
        new_project_for(slug, display_name=display_name, description=description, origin=origin),
    )
    record = record_from_row(row)
    resources = record.resources
    registry = get_project_registry()

    try:
        # refresh_strict() raises on any failure (unlike ensure_fresh's swallow-and-log), so a
        # registry that cannot see this project yet aborts the create through the handler below
        # instead of creating indexes while the guard still does not recognize the new slug.
        await registry.refresh_strict()

        with bind_project(record):
            from src.services.curation.index_bootstrap import ensure_indexes

            resources.upload_root.mkdir(parents=True, exist_ok=True)
            resources.project_state_dir.mkdir(parents=True, exist_ok=True)
            await ensure_indexes(client)
            from src.services.projects.vlm_default import pin_new_project_vlm_off

            await pin_new_project_vlm_off(client)
        if clone_settings_from:
            clone_conflicts = await clone_settings(
                client,
                target_record=record,
                from_slug=clone_settings_from,
                axes=clone_axes,
            )
            if clone_conflicts:
                # `keymap_clone_conflicts` on the 201 response is always `[]` for a create (that
                # field belongs to the standalone clone_settings route), so a dropped keymap
                # conflict is surfaced as one ProjectWarning per conflict.
                logger.warning(
                    'project_create_keymap_clone_conflicts',
                    slug=slug,
                    from_slug=clone_settings_from,
                    conflicts=clone_conflicts,
                )
                for conflict in clone_conflicts:
                    warnings.append(
                        {
                            'code': 'keymap_clone_conflict',
                            'message': (
                                f"keymap action '{conflict['action_id']}' combo "
                                f"'{conflict['combo']}' collides with class "
                                f"'{conflict['class_name']}' (id {conflict['class_id']}) "
                                'so the keymap was not cloned'
                            ),
                        }
                    )
        # Idempotent: a clone that copied 'classes' (or _apply_clone's own
        # no-classes-copied branch) may already have seeded it, but every
        # new project gets one guaranteed call under its own binding.
        from src.services.classes.registry import get_class_registry
        from src.services.curation.region_class import ensure_region_class

        with bind_project(record):
            # The registry document exists from the first moment (new project, empty).
            await get_class_registry().snapshot()
            await ensure_region_class()

        # ensure_indexes is itself fail-open (every create/migration failure inside it is logged
        # and swallowed): verify every index this record claims to own exists before ever calling
        # the project 'active'.
        with bind_project(record, read_only=True):
            missing_indexes = [
                name
                for name in sorted(set(record.resources.indexes.values()))
                if not await client.indices.exists(index=name)
            ]
        if missing_indexes:
            raise RuntimeError(
                f"'{slug}' create verification found missing indexes: {missing_indexes}"
            )
    except Exception as exc:
        logger.error('project_create_failed', slug=slug, error=str(exc))
        # `from_states=('building',)` refuses to write 'failed' over a status this handler never
        # validated (e.g. a delete that raced in), so a slug is never resurrected.
        try:
            await run_repo(
                'fail a create',
                repo.set_status,
                slug,
                to='failed',
                from_states=('building',),
                bump_revision=False,
            )
        except RequestRejectedError as refetch_exc:
            await _cleanup_orphaned_by_concurrent_delete(client, record, refetch_exc)
            raise
        await registry.ensure_fresh()
        raise

    if not activate:
        return record, warnings

    return await finish_building(client, record), warnings


async def finish_building(client: Any, record: ProjectRecord, *, ok: bool = True) -> ProjectRecord:
    """The ``building`` -> ``active`` (or ``failed``) transition that ends a create, and a
    combine's fill of its target.

    ``from_states=('building',)`` is what stops a slow finish from resurrecting a project a
    concurrent DELETE already removed (the row is gone: 404). A create whose indexes outlive such a
    delete is cleaned by ``_cleanup_orphaned_by_concurrent_delete``.
    """
    try:
        row = await run_repo(
            'finish building a project',
            repo.set_status,
            record.slug,
            to='active' if ok else 'failed',
            from_states=('building',),
            bump_revision=False,
        )
    except RequestRejectedError as refetch_exc:
        await _cleanup_orphaned_by_concurrent_delete(client, record, refetch_exc)
        raise
    await get_project_registry().ensure_fresh()

    from src.services.projects.capacity import invalidate_capacity_cache

    invalidate_capacity_cache()  # a create just changed this cluster's shard count
    return record_from_row(row)


async def patch_project(
    client: Any,  # noqa: ARG001 - the lifecycle calls share one signature
    *,
    slug: str,
    display_name: str | None,
    description: str | None,
    expected_revision: int,
) -> ProjectRecord:
    """§4 ``PATCH /projects/{project}``. The slug is immutable; OCC via ``expected_revision``
    against the row's own ``revision``."""
    row = await run_repo(
        'patch a project',
        repo.patch_project,
        slug,
        expected_revision=expected_revision,
        display_name=display_name,
        description=description,
    )
    await get_project_registry().ensure_fresh()
    return record_from_row(row)


async def archive_project(
    client: Any,  # noqa: ARG001 - the lifecycle calls share one signature
    *,
    slug: str,
    expected_revision: int,
) -> ProjectRecord:
    """``active`` -> ``archived`` only (409 ``invalid_transition`` otherwise), never the last
    active project."""
    record = await _get_mutable_record(slug)
    _require_transition(record, 'archive', ARCHIVABLE_STATUSES)
    _require_revision(record, expected_revision)
    jobs = await running_jobs(record)
    if jobs:
        raise rejection(
            409,
            'project_busy',
            busy_message(slug, jobs),
            project=slug,
            jobs=[j.to_wire() for j in jobs],
        )
    await _last_active_check(record)
    row = await run_repo(
        'archive a project',
        repo.set_status,
        slug,
        to='archived',
        from_states=ARCHIVABLE_STATUSES,
        expected_revision=expected_revision,
        action='archive',
    )
    await get_project_registry().ensure_fresh()
    return record_from_row(row)


async def unarchive_project(
    client: Any,  # noqa: ARG001 - the lifecycle calls share one signature
    *,
    slug: str,
    expected_revision: int,
) -> ProjectRecord:
    """``archived`` -> ``active`` only (409 ``invalid_transition`` otherwise)."""
    record = await _get_mutable_record(slug)
    _require_transition(record, 'unarchive', UNARCHIVABLE_STATUSES)
    _require_revision(record, expected_revision)
    row = await run_repo(
        'unarchive a project',
        repo.set_status,
        slug,
        to='active',
        from_states=UNARCHIVABLE_STATUSES,
        expected_revision=expected_revision,
        action='unarchive',
    )
    await get_project_registry().ensure_fresh()
    return record_from_row(row)


# clone_settings / clone_settings_into live in clone.py (700-LOC
# ratchet); re-exported here so `lifecycle.clone_settings(...)` keeps
# working for every existing caller.
from src.services.projects.clone import clone_settings, clone_settings_into  # noqa: E402,F401  # bottom-of-module import breaks a circular import; imported for its side effect

# dry_run_delete / delete_project / delete_project_finish live in
# delete.py (700-LOC ratchet); re-exported here so
# `lifecycle.delete_project(...)` keeps working for every existing caller.
from src.services.projects.delete import delete_project, delete_project_finish, dry_run_delete  # noqa: E402,F401  # bottom-of-module import breaks a circular import; imported for its side effect
