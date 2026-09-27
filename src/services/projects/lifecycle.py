"""Project lifecycle service: create / patch / archive / unarchive /
clone_settings / delete (dry-run and guarded).

See ``docs/design/openprocessor_internal/projects_plan.md`` §4, §7, the
owner decisions in §13 (D4 capacity, D5 ``default`` is undeletable), and
the Cropwright rev-3 deltas 2 (envelope + revision), 3 (list membership,
implemented in P1's ``_project_models.py``), 10 (delete fits a proxy
timeout), 11 (structured refusals) and 12 (full ``capacity`` on the
409). There is no unscoped alias and no back-compat shim anywhere in
this module (owner decision, 2026-09-26): every route this backs lives
only under the global router or a project's scoped prefix.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.config.projects import DEFAULT_SLUG, ProjectRecord, is_valid_slug, resources_for_new
from src.core.logging import get_logger
from src.routers.curation._config_common_models import api_error
from src.routers.curation._project_models import ARCHIVABLE_STATUSES, UNARCHIVABLE_STATUSES
from src.services.projects.capacity import capacity_status
from src.services.projects.registry import (
    RevisionConflictError,
    get_project_registry,
    get_record_with_seq,
    write_record as _raw_write_record,
)


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)

# How long delete waits for the detection worker's per-project inflight
# count to drain before giving up and rolling back (plan §4 step 3).
_DELETE_DRAIN_TIMEOUT_SECONDS = 60.0
_DELETE_DRAIN_POLL_SECONDS = 1.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def write_record(
    client: Any,
    record: ProjectRecord,
    *,
    if_seq_no: int | None = None,
    if_primary_term: int | None = None,
) -> None:
    """``registry.write_record``, with the storage-level OCC race
    (:class:`RevisionConflictError` -- another writer's bump landed between
    this caller's read and its write) translated into the API's 409
    ``revision_conflict``, exactly like a stale ``expected_revision``
    would be (:func:`_require_revision`). Every lifecycle mutation
    writes through here, never the raw registry function, so a losing
    concurrent writer never silently clobbers or 500s."""
    try:
        await _raw_write_record(
            client, record, if_seq_no=if_seq_no, if_primary_term=if_primary_term
        )
    except RevisionConflictError as exc:
        raise api_error(
            409,
            'revision_conflict',
            f"'{record.slug}' was modified by another request; refresh and retry",
            project=record.slug,
        ) from exc


@dataclass(frozen=True)
class JobRef:
    """A running job blocking a lifecycle action (delta 11)."""

    kind: str
    kind_label: str
    id: str
    label: str
    started_at: str

    def to_wire(self) -> dict[str, Any]:
        return {
            'kind': self.kind,
            'kind_label': self.kind_label,
            'id': self.id,
            'label': self.label,
            'started_at': self.started_at,
        }


async def _capacity_error_or_warning(
    client: Any,
) -> list[dict[str, str]]:
    """Runs the §2.3 capacity check for one more project's shards.
    Raises 409 ``shard_budget_exceeded`` when blocked; otherwise returns
    ``[{"code": "shard_budget_high", ...}]`` on warn, else ``[]``."""
    capacity = await capacity_status(client)
    if capacity is None:
        return []
    if capacity.status == 'blocked':
        raise api_error(
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
        return [{'code': 'shard_budget_high', 'message': capacity.message}]
    return []


async def _resolve_existing(slug: str) -> ProjectRecord | None:
    registry = get_project_registry()
    await registry.ensure_fresh()
    return registry.get(slug)


def _require_found(record: ProjectRecord | None, slug: str) -> ProjectRecord:
    if record is None or record.status == 'deleted':
        raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)
    return record


async def _get_mutable_record(
    client: Any, slug: str
) -> tuple[ProjectRecord, int | None, int | None]:
    """The stored record plus its OCC seq/term. ``default`` is now an
    ordinary project record (``bootstrap_default_project`` writes it at
    startup like any other project), so there is no synthesis fallback
    here -- a missing doc is a genuine 404."""
    stored, seq, term = await get_record_with_seq(client, slug)
    if stored is not None:
        return stored, seq, term
    raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)


async def create_project(
    client: Any,
    *,
    slug: str,
    display_name: str,
    description: str = '',
    clone_settings_from: str | None = None,
    clone_axes: list[str] | None = None,
) -> tuple[ProjectRecord, list[dict[str, str]]]:
    """§4 ``POST /projects``. Steps: validate → capacity → record
    ``building`` → create indexes/dirs → optional clone → ``active``. A
    failure midway leaves the record ``failed`` with an error, never
    partially ``active``."""
    from src.config.curation import base_curation_config

    if not is_valid_slug(slug):
        raise api_error(422, 'slug_invalid', f"'{slug}' is not a valid project slug", project=slug)

    existing = await _resolve_existing(slug)
    if existing is not None:
        if existing.status == 'deleted':
            raise api_error(
                409,
                'slug_retired',
                f"'{slug}' was used by a deleted project and its slug is retired",
                project=slug,
            )
        raise api_error(409, 'slug_taken', f"a project named '{slug}' already exists", project=slug)

    warnings = await _capacity_error_or_warning(client)

    now = _now()
    resources = resources_for_new(slug, base_curation_config())
    record = ProjectRecord(
        slug=slug,
        display_name=display_name,
        description=description,
        status='building',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )
    await write_record(client, record)
    registry = get_project_registry()
    await registry.ensure_fresh()

    try:
        with bind_project(record):
            from src.routers.curation._common import _ensure_indexes

            resources.upload_root.mkdir(parents=True, exist_ok=True)
            resources.project_state_dir.mkdir(parents=True, exist_ok=True)
            await _ensure_indexes(client)
        if clone_settings_from:
            await clone_settings(
                client,
                target_record=record,
                from_slug=clone_settings_from,
                axes=clone_axes,
            )
        # Idempotent: a clone that copied 'classes' (or _apply_clone's own
        # no-classes-copied branch) may already have seeded it, but every
        # new project gets one guaranteed call under its own binding.
        from src.services.curation.region_class import ensure_region_class

        with bind_project(record):
            ensure_region_class()
    except Exception as exc:
        logger.error('project_create_failed', slug=slug, error=str(exc))
        _, seq, term = await get_record_with_seq(client, slug)
        failed = replace(record, status='failed', updated_at=_now())
        await write_record(client, failed, if_seq_no=seq, if_primary_term=term)
        await registry.ensure_fresh()
        raise

    _, seq, term = await get_record_with_seq(client, slug)
    active = replace(record, status='active', updated_at=_now())
    await write_record(client, active, if_seq_no=seq, if_primary_term=term)
    await registry.ensure_fresh()
    return active, warnings


async def patch_project(
    client: Any,
    *,
    slug: str,
    display_name: str | None,
    description: str | None,
    expected_revision: int,
) -> ProjectRecord:
    """§4 ``PATCH /projects/{project}``. The slug is immutable; OCC via
    ``expected_revision`` against the record's own ``revision`` field
    (not the registry-wide counter)."""
    record, seq, term = await _get_mutable_record(client, slug)
    _require_revision(record, expected_revision)
    updated = replace(
        record,
        display_name=display_name if display_name is not None else record.display_name,
        description=description if description is not None else record.description,
        revision=record.revision + 1,
        updated_at=_now(),
    )
    await write_record(client, updated, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return updated


_KIND_LABELS = {
    'train': 'Training run',
    'bakeoff': 'Bake-off',
    'autolabel': 'Auto-label',
    'export': 'Export',
    'detection_worker': 'Detection worker',
}


async def running_jobs(record: ProjectRecord) -> list[JobRef]:
    """The project's real running-job inventory, via
    ``src.services.projects.busy.running_jobs`` -- the single source of
    truth for "is this project busy" across every job-producing
    subsystem (§5.4). Delete/archive never re-implement their own file
    scan; they only adapt ``busy.JobRef`` (``kind``/``job_id``) to this
    module's wire-shaped ``JobRef`` (delta 11)."""
    from src.services.projects import busy

    return [
        JobRef(
            kind=j.kind,
            kind_label=_KIND_LABELS.get(j.kind, j.kind),
            id=j.job_id,
            label=j.job_id,
            started_at='',
        )
        for j in busy.running_jobs(record)
    ]


def _other_active_slugs(snapshot: dict[str, ProjectRecord], slug: str) -> list[str]:
    """Every *other* project that is ``active``. Archived, building,
    failed, deleting and deleted projects do not count: none of them can
    take writes. The one rule both the delete dry run and the real
    archive/delete guards use, so the dry run never disagrees."""
    return [s for s, r in snapshot.items() if s != slug and r.status == 'active']


_LAST_ACTIVE_MESSAGE = 'this is the only active project; leave at least one'


async def _last_active_check(record: ProjectRecord) -> None:
    """Refuse an archive or delete that would leave no active project."""
    registry = get_project_registry()
    await registry.ensure_fresh()
    if not _other_active_slugs(dict(registry.snapshot()), record.slug):
        raise api_error(409, 'last_active_project', _LAST_ACTIVE_MESSAGE, project=record.slug)


def _require_transition(record: ProjectRecord, action: str, allowed: frozenset[str]) -> None:
    if record.status not in allowed:
        raise api_error(
            409,
            'invalid_transition',
            f"cannot {action} '{record.slug}' while it is {record.status}",
            project=record.slug,
            project_status=record.status,
            action=action,
        )


def _require_revision(record: ProjectRecord, expected_revision: int) -> None:
    if record.revision != expected_revision:
        raise api_error(
            409,
            'revision_conflict',
            f'expected revision {expected_revision}, current is {record.revision}',
            project=record.slug,
            current_revision=record.revision,
        )


async def archive_project(client: Any, *, slug: str, expected_revision: int) -> ProjectRecord:
    """``active`` -> ``archived`` only (409 ``invalid_transition``
    otherwise), never the last active project."""
    record, seq, term = await _get_mutable_record(client, slug)
    _require_transition(record, 'archive', ARCHIVABLE_STATUSES)
    _require_revision(record, expected_revision)
    jobs = await running_jobs(record)
    if jobs:
        raise api_error(
            409,
            'project_busy',
            f"'{slug}' has {len(jobs)} running job(s)",
            project=slug,
            jobs=[j.to_wire() for j in jobs],
        )
    await _last_active_check(record)
    updated = replace(record, status='archived', revision=record.revision + 1, updated_at=_now())
    await write_record(client, updated, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return updated


async def unarchive_project(client: Any, *, slug: str, expected_revision: int) -> ProjectRecord:
    """``archived`` -> ``active`` only (409 ``invalid_transition`` otherwise)."""
    record, seq, term = await _get_mutable_record(client, slug)
    _require_transition(record, 'unarchive', UNARCHIVABLE_STATUSES)
    _require_revision(record, expected_revision)
    updated = replace(record, status='active', revision=record.revision + 1, updated_at=_now())
    await write_record(client, updated, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return updated


# clone_settings / clone_settings_into live in clone.py (700-LOC
# ratchet); re-exported here so `lifecycle.clone_settings(...)` keeps
# working for every existing caller.
from src.services.projects.clone import clone_settings, clone_settings_into  # noqa: E402,F401


async def dry_run_delete(client: Any, *, slug: str) -> dict[str, Any]:
    """§4 ``DELETE ?dry_run=true``: report only, writes nothing."""
    record = _require_found(await _resolve_existing(slug), slug)
    blocking: list[dict[str, str]] = []
    if record.slug == DEFAULT_SLUG:
        blocking.append(
            {
                'code': 'project_protected',
                'message': 'The default project can be archived but not deleted.',
            }
        )
    jobs = await running_jobs(record)
    if jobs:
        blocking.append({'code': 'project_busy', 'message': f'{len(jobs)} job(s) still running'})
    registry = get_project_registry()
    await registry.ensure_fresh()
    if not _other_active_slugs(dict(registry.snapshot()), slug):
        blocking.append({'code': 'last_active_project', 'message': _LAST_ACTIVE_MESSAGE})

    indexes: list[dict[str, Any]] = []
    for name in sorted(set(record.resources.indexes.values())):
        try:
            with bind_project(record, read_only=True):
                count_resp = await client.count(index=name)
            docs = int(count_resp.get('count') or 0)
        except Exception:
            docs = 0
        indexes.append({'name': name, 'docs': docs, 'store_bytes': None})

    dirs: list[dict[str, Any]] = []
    for path in (
        record.resources.export_root,
        record.resources.upload_root,
        record.resources.project_state_dir,
        record.resources.bakeoff_jobs_dir,
        record.resources.train_jobs_dir,
        record.resources.autolabel_dir,
    ):
        size = 0
        try:
            if path.exists():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        except OSError:
            size = 0
        dirs.append({'path': str(path), 'bytes': size})

    return {
        'indexes': indexes,
        'dirs': dirs,
        'promoted_models': [],
        'mlflow_experiment': record.resources.mlflow_experiment,
        'running_jobs': [j.to_wire() for j in jobs],
        'referenced_by': [],
        'blocking': [b['code'] for b in blocking],
        'blocking_detail': blocking,
    }


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


async def _delete_indexes(client: Any, record: ProjectRecord) -> None:
    for name in sorted(set(record.resources.indexes.values())):
        try:
            with bind_project(record):
                await client.indices.delete(index=name, ignore=[404])
        except Exception as exc:
            logger.warning(
                'project_delete_index_failed', project=record.slug, index=name, error=str(exc)
            )


def _rm_dir_guarded(path: Path, expected_root: Path) -> None:
    if not _path_within(path, expected_root):
        logger.error('project_delete_path_escape', path=str(path), expected_root=str(expected_root))
        raise api_error(
            500, 'internal_isolation_error', f'refusing to delete outside {expected_root}'
        )
    shutil.rmtree(path, ignore_errors=True)


async def _delete_dirs(record: ProjectRecord) -> None:
    """Remove every per-project dir, each guarded to resolve inside its
    expected root (§4 step 6: "a path outside refuses"). A new
    project's dirs are siblings under ``OP_PROJECTS_DATA_ROOT``
    (exports, class registry) or under the deployment ``state_dir``
    (uploads, job dirs) -- never under ``default``'s own dirs, which
    :func:`resources_for_new` never nests anything into."""
    from src.config.curation import base_curation_config
    from src.config.projects import projects_data_root

    base = base_curation_config()
    data_root = projects_data_root()
    _rm_dir_guarded(record.resources.export_root, data_root)
    _rm_dir_guarded(record.resources.class_registry_path.parent, data_root)
    _rm_dir_guarded(record.resources.bakeoff_eval_root, data_root)
    _rm_dir_guarded(record.resources.project_state_dir, base.state_dir)
    _rm_dir_guarded(record.resources.upload_root, base.state_dir)
    _rm_dir_guarded(record.resources.bakeoff_jobs_dir, base.state_dir)
    _rm_dir_guarded(record.resources.train_jobs_dir, record.resources.train_jobs_dir.parent.parent)
    _rm_dir_guarded(record.resources.autolabel_dir, record.resources.autolabel_dir.parent.parent)


async def _wait_for_drain(record: ProjectRecord) -> bool:  # noqa: ARG001
    """Best-effort drain wait (plan §4 step 3). P2 owns the authoritative
    per-project ``runtime`` doc this reads; until it lands there is
    nothing to poll, so this is a no-op success (nothing known to be
    inflight)."""
    await asyncio.sleep(0)
    return True


async def delete_project(
    client: Any,
    *,
    slug: str,
    confirm: str | None,
    force: bool = False,
) -> ProjectRecord:
    """§4 guarded delete, background-completing (delta 10): the caller
    gets the ``deleting`` record back immediately (202), and this
    function runs the rest (drain wait, index/dir removal, tombstone) as
    a background task so a slow delete never blocks past a proxy
    timeout. Call :func:`delete_project_finish` to run steps 3-9; this
    function only validates and flips the status."""
    record = _require_found(await _resolve_existing(slug), slug)

    if record.slug == DEFAULT_SLUG:
        raise api_error(
            409,
            'project_protected',
            'The default project can be archived but not deleted.',
            project=slug,
        )
    if confirm is None:
        raise api_error(422, 'confirm_mismatch', 'confirm is required for a real delete')
    if confirm != slug:
        raise api_error(422, 'confirm_mismatch', f"confirm must equal the slug '{slug}'")

    jobs = await running_jobs(record)
    if jobs:
        raise api_error(
            409,
            'project_busy',
            f"'{slug}' has {len(jobs)} running job(s)",
            project=slug,
            jobs=[j.to_wire() for j in jobs],
        )
    await _last_active_check(record)

    if not force:
        shared_models = await _shared_model_users(record)
        if shared_models:
            raise api_error(
                409,
                'in_use',
                f"'{slug}' has {len(shared_models)} model(s) opted into cross-project "
                'sharing; deleting could break another project that depends on them',
                project=slug,
                projects=shared_models,
            )
    else:
        shared_models = await _shared_model_users(record)
        if shared_models:
            logger.warning(
                'project_delete_forced_past_shared_models',
                project=slug,
                shared_models=shared_models,
            )

    _, seq, term = await get_record_with_seq(client, slug)
    deleting = replace(record, status='deleting', updated_at=_now())
    await write_record(client, deleting, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return deleting


async def _shared_model_users(record: ProjectRecord) -> list[str]:
    """§5.5 in_use guard: which of this project's own promoted models
    have opted into cross-project sharing (``PUT /models/{name}/sharing``,
    ``promote.json.shared``)?

    KNOWN GAP (flagged, not faked): this returns the *shared model
    names*, not the *dependent project slugs* the plan asks for -- P2's
    model-sharing plumbing (``src.services.training.model_classes``,
    ``src.routers.curation._models_sharing``) has no reverse index of
    "which projects actually reference model X as their active
    detector". That scan needs each project's own bound
    ``DetectionProfile`` read, which is explicitly the not-yet-landed W4
    profile-CRUD wave's job (see the ``TODO(W4/profile_validation)`` in
    ``_models_sharing.py``, which even ``PUT .../sharing`` itself defers
    on). Until W4 lands there is no way to name which projects would
    actually break, so a project with any ``shared=True`` promoted model
    is still refused (``in_use``) unless ``force=True`` -- "opted into
    sharing" is itself evidence someone may depend on it, and silently
    allowing the delete would be the worse failure mode -- but the
    caller must read the returned names as "these models of mine are
    shared", not as consumer project slugs.
    """
    from src.services.training.model_classes import is_model_shared, model_owner_project
    from src.services.training.triton_promote import resolve_triton_models_dir

    models_dir = resolve_triton_models_dir()
    if not models_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in models_dir.iterdir()
        if entry.is_dir()
        and model_owner_project(entry.name) == record.slug
        and is_model_shared(entry.name)
    )


async def delete_project_finish(client: Any, *, slug: str) -> ProjectRecord:
    """Steps 3-9 of the guarded delete (§4): drain wait, unload models
    (P2 scope, not yet wired), delete the exact indexes, remove the
    dirs, soft-delete the MLflow experiment if reachable, tombstone.
    Idempotent: safe to re-run after a crash between any two steps,
    because every step here is itself idempotent (index delete with
    ``ignore=[404]``, ``rmtree(ignore_errors=True)``, and the final
    ``deleted`` write is OCC-guarded so a duplicate run is a no-op)."""
    stored, _, _ = await get_record_with_seq(client, slug)
    if stored is None:
        raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)
    record = stored
    if record.status == 'deleted':
        return record

    drained = await _wait_for_drain(record)
    if not drained:
        _, seq, term = await get_record_with_seq(client, slug)
        rolled_back = replace(record, status='failed', updated_at=_now())
        await write_record(client, rolled_back, if_seq_no=seq, if_primary_term=term)
        await get_project_registry().ensure_fresh()
        raise api_error(
            409, 'project_busy', f"'{slug}' did not drain within the timeout", project=slug
        )

    await _delete_indexes(client, record)
    await _delete_dirs(record)
    await _soft_delete_mlflow(record)

    _, seq, term = await get_record_with_seq(client, slug)
    tombstoned = replace(record, status='deleted', updated_at=_now())
    await write_record(client, tombstoned, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    logger.info('project_deleted', project=slug)
    return tombstoned


async def _soft_delete_mlflow(record: ProjectRecord) -> None:
    """Best-effort MLflow experiment soft-delete; unreachable server is
    reported, never fatal to the delete."""
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(record.resources.mlflow_experiment)
        if experiment is not None:
            client.delete_experiment(experiment.experiment_id)
    except Exception as exc:
        logger.info('project_delete_mlflow_skipped', project=record.slug, error=str(exc))
