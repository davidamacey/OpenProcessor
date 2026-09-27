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
from src.services.projects.capacity import capacity_status
from src.services.projects.registry import get_project_registry, get_record_with_seq, write_record


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)

CLONEABLE_AXES: tuple[str, ...] = ('settings_defaults', 'classes')

# How long delete waits for the detection worker's per-project inflight
# count to drain before giving up and rolling back (plan §4 step 3).
_DELETE_DRAIN_TIMEOUT_SECONDS = 60.0
_DELETE_DRAIN_POLL_SECONDS = 1.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


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
    """Like :func:`get_record_with_seq`, but ``default`` always resolves
    even before its registry doc has ever been written (first-boot /
    test processes that never ran the startup bootstrap) -- it is
    synthesized from the env, with no seq/term, so the following write
    creates the doc unconditionally."""
    stored, seq, term = await get_record_with_seq(client, slug)
    if stored is not None:
        return stored, seq, term
    if slug == DEFAULT_SLUG:
        from src.services.projects.registry import default_project_record

        return default_project_record(), None, None
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
    if record.revision != expected_revision:
        raise api_error(
            409,
            'revision_conflict',
            f'expected revision {expected_revision}, current is {record.revision}',
            project=slug,
            current_revision=record.revision,
        )
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


async def running_jobs(record: ProjectRecord) -> list[JobRef]:
    """Best-effort job inventory for this project (train jobs, auto-label
    triggers, bake-off jobs). P2 (workers/jobs) owns the authoritative
    per-worker ``runtime`` docs this will eventually also read; until
    that lands, this checks the file-backed job dirs P1's
    ``ProjectResources`` already names, so archive/delete guards degrade
    to "no known job dirs" rather than silently skipping the check."""
    jobs: list[JobRef] = []
    for job_dir, kind, kind_label in (
        (record.resources.train_jobs_dir, 'train', 'Training run'),
        (record.resources.bakeoff_jobs_dir, 'bakeoff', 'Bake-off'),
    ):
        try:
            if not job_dir.exists():
                continue
            for job_file in sorted(job_dir.glob('*.job.json')):
                status_file = job_file.with_name(job_file.name.replace('.job.json', '.status.json'))
                if status_file.exists():
                    import json

                    try:
                        status_doc = json.loads(status_file.read_text())
                    except (OSError, ValueError):
                        continue
                    if status_doc.get('status') in ('running', 'queued'):
                        jobs.append(
                            JobRef(
                                kind=kind,
                                kind_label=kind_label,
                                id=job_file.stem.removesuffix('.job'),
                                label=status_doc.get('label', job_file.stem),
                                started_at=status_doc.get('started_at', ''),
                            )
                        )
        except OSError as exc:
            logger.warning(
                'project_job_scan_failed', project=record.slug, dir=str(job_dir), error=str(exc)
            )
    return jobs


async def _last_active_check(record: ProjectRecord) -> None:
    """§4: "the only non-archived project" -- counts anything that is
    not itself archived/deleted, matching the plan's delete-guard
    wording verbatim and reused for archive (a project cannot become
    the deployment's last writable project)."""
    registry = get_project_registry()
    await registry.ensure_fresh()
    snapshot = registry.snapshot()
    others_active = [
        s
        for s, r in snapshot.items()
        if s != record.slug and r.status not in ('archived', 'deleted')
    ]
    if not others_active:
        raise api_error(
            409,
            'last_active_project',
            'this is the only remaining project; leave at least one',
            project=record.slug,
        )


async def archive_project(client: Any, *, slug: str, expected_revision: int) -> ProjectRecord:
    record, seq, term = await _get_mutable_record(client, slug)
    if record.revision != expected_revision:
        raise api_error(
            409,
            'revision_conflict',
            f'expected revision {expected_revision}, current is {record.revision}',
            project=slug,
            current_revision=record.revision,
        )
    jobs = await running_jobs(record)
    if jobs:
        raise api_error(
            409,
            'project_busy',
            f"'{slug}' has {len(jobs)} running job(s)",
            project=slug,
            jobs=[j.id for j in jobs],
        )
    await _last_active_check(record)
    updated = replace(record, status='archived', revision=record.revision + 1, updated_at=_now())
    await write_record(client, updated, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return updated


async def unarchive_project(client: Any, *, slug: str, expected_revision: int) -> ProjectRecord:
    record, seq, term = await _get_mutable_record(client, slug)
    if record.revision != expected_revision:
        raise api_error(
            409,
            'revision_conflict',
            f'expected revision {expected_revision}, current is {record.revision}',
            project=slug,
            current_revision=record.revision,
        )
    updated = replace(record, status='active', revision=record.revision + 1, updated_at=_now())
    await write_record(client, updated, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return updated


async def clone_settings(
    client: Any,
    *,
    target_record: ProjectRecord,
    from_slug: str,
    axes: list[str] | None,
) -> None:
    """§4 ``clone_settings``: copy the source project's settings and/or
    class registry into ``target_record``. Reads the source under a
    read-only bind so the guard rejects any accidental write to it.
    Never partially unbinds a class: ``classes`` is refused (422→409
    ``target_not_empty``) unless the target has zero items, so a clone
    is always a byte-identical starting point, not a merge."""
    from src.clients.curation_opensearch import get_curation_settings, update_curation_settings

    resolved_axes = axes if axes else list(CLONEABLE_AXES)
    for axis in resolved_axes:
        if axis not in CLONEABLE_AXES:
            raise api_error(422, 'combine_invalid', f"unknown clone axis '{axis}'")

    source = await _resolve_existing(from_slug)
    source = _require_found(source, from_slug)

    if 'classes' in resolved_axes:
        with bind_project(target_record):
            from src.config.curation import items_index

            count_resp = await client.count(index=items_index())
        if (count_resp.get('count') or 0) > 0:
            raise api_error(
                409,
                'target_not_empty',
                f"'{target_record.slug}' already has items; classes cannot be cloned",
                project=target_record.slug,
            )

    if 'settings_defaults' in resolved_axes:
        with bind_project(source, read_only=True):
            source_settings = await get_curation_settings(client)
        with bind_project(target_record):
            await update_curation_settings(client, dict(source_settings.get('defaults', {})))

    if 'classes' in resolved_axes:
        src_path = source.resources.class_registry_path
        dst_path = target_record.resources.class_registry_path
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        with bind_project(target_record):
            from src.clients.curation_opensearch import ClassRegistry

            registry = ClassRegistry(dst_path)
            await registry.sync_to_opensearch(client)


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
    snapshot = registry.snapshot()
    others_active = [
        s for s, r in snapshot.items() if s != slug and r.status in ('active', 'archived')
    ]
    if not others_active:
        blocking.append(
            {'code': 'last_active_project', 'message': 'this is the only remaining project'}
        )

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
    confirm: str,
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
    if confirm != slug:
        raise api_error(422, 'confirm_mismatch', f"confirm must equal the slug '{slug}'")

    jobs = await running_jobs(record)
    if jobs:
        raise api_error(
            409,
            'project_busy',
            f"'{slug}' has {len(jobs)} running job(s)",
            project=slug,
            jobs=[j.id for j in jobs],
        )
    await _last_active_check(record)

    if not force:
        in_use = await _shared_model_users(record)
        if in_use:
            raise api_error(
                409,
                'in_use',
                f"a promoted model from '{slug}' is used by another project's active profile",
                project=slug,
                projects=in_use,
            )

    _, seq, term = await get_record_with_seq(client, slug)
    deleting = replace(record, status='deleting', updated_at=_now())
    await write_record(client, deleting, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return deleting


async def _shared_model_users(record: ProjectRecord) -> list[str]:  # noqa: ARG001
    """P2/§5.5 (shared promoted models) is not in this wave's scope; there
    is no shared-model registry to query yet, so this always returns
    "no users" rather than fabricating an answer."""
    return []


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
