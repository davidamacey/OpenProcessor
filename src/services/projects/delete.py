"""``dry_run_delete`` / ``delete_project`` / ``delete_project_finish`` (§4):
guarded project delete, background-completing.

Split out of ``lifecycle.py`` to stay under the repo's 700-LOC
pre-commit ratchet; ``lifecycle.py`` re-exports every public name here so
every existing caller (``lifecycle.delete_project(...)``, etc.) keeps
working unchanged.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from src.config.project_context import bind_project
from src.config.projects import DEFAULT_SLUG, ProjectStatus
from src.core.logging import get_logger
from src.routers.curation._config_common_models import api_error
from src.services.projects.registry import get_project_registry, get_record_with_seq


if TYPE_CHECKING:
    from pathlib import Path

    from src.config.projects import ProjectRecord


logger = get_logger(__name__)

# How long delete waits for the detection worker's per-project inflight
# count to drain before giving up and rolling back (plan §4 step 3).
_DELETE_DRAIN_TIMEOUT_SECONDS = 60.0
_DELETE_DRAIN_POLL_SECONDS = 1.0

_DELETABLE_STATUSES = frozenset({'active', 'archived', 'failed'})


async def dry_run_delete(client: Any, *, slug: str) -> dict[str, Any]:
    """§4 ``DELETE ?dry_run=true``: report only, writes nothing."""
    from src.services.projects.lifecycle import (
        _LAST_ACTIVE_MESSAGE,
        _other_active_slugs,
        _require_found,
        _resolve_existing,
        running_jobs,
    )

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


async def _delete_indexes(client: Any, record: ProjectRecord) -> list[str]:
    """Delete every one of the project's own indexes. M4: collects (never
    swallows) each failure and returns the names that did not delete, so
    the caller can leave the record retryable instead of tombstoning a
    project some of whose indexes are still there -- orphaned, owned by
    a slug nothing can ever reach again."""
    failed: list[str] = []
    for name in sorted(set(record.resources.indexes.values())):
        try:
            with bind_project(record):
                await client.indices.delete(index=name, ignore=[404])
        except Exception as exc:
            logger.warning(
                'project_delete_index_failed', project=record.slug, index=name, error=str(exc)
            )
            failed.append(name)
    return failed


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


async def _wait_for_drain(record: ProjectRecord) -> bool:
    """Drain wait (plan §4 step 3, M3): poll the detection worker's
    per-project inflight liveness (``busy._detection_worker_inflight``,
    P2's real file-based liveness docs -- landed since the stale
    docstring this replaces was written) until no host reports inflight
    writes, or ``_DELETE_DRAIN_TIMEOUT_SECONDS`` elapses. The project was
    already flipped to ``deleting`` before this runs, so the binder
    refuses any *new* write for the duration; this only waits out
    writers that were already inflight."""
    from src.services.projects import busy

    deadline = asyncio.get_event_loop().time() + _DELETE_DRAIN_TIMEOUT_SECONDS
    while True:
        if not busy._detection_worker_inflight(record):
            return True
        if asyncio.get_event_loop().time() >= deadline:
            return False
        await asyncio.sleep(_DELETE_DRAIN_POLL_SECONDS)


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
    from src.services.projects.lifecycle import (
        _last_active_check,
        _now,
        _require_found,
        _require_transition,
        _resolve_existing,
        running_jobs,
        write_record,
    )

    record = _require_found(await _resolve_existing(slug), slug)

    if record.slug == DEFAULT_SLUG:
        raise api_error(
            409,
            'project_protected',
            'The default project can be archived but not deleted.',
            project=slug,
        )
    # m3: active|archived|failed -> deleting only; never re-flip a
    # 'building' or already-'deleting' record.
    _require_transition(record, 'delete', _DELETABLE_STATUSES)
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
    deleting = replace(
        record, status='deleting', pre_delete_status=record.status, updated_at=_now()
    )
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
    from src.services.projects.lifecycle import _now, write_record

    stored, _, _ = await get_record_with_seq(client, slug)
    if stored is None:
        raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)
    record = stored
    if record.status == 'deleted':
        return record

    drained = await _wait_for_drain(record)
    if not drained:
        # M3: roll back to the status delete found it in, not always 'failed'.
        _, seq, term = await get_record_with_seq(client, slug)
        fallback_status: ProjectStatus = 'failed'
        rolled_back = replace(
            record,
            status=cast('ProjectStatus', record.pre_delete_status) or fallback_status,
            pre_delete_status=None,
            updated_at=_now(),
        )
        await write_record(client, rolled_back, if_seq_no=seq, if_primary_term=term)
        await get_project_registry().ensure_fresh()
        raise api_error(
            409, 'project_busy', f"'{slug}' did not drain within the timeout", project=slug
        )

    failed_indexes = await _delete_indexes(client, record)
    if failed_indexes:
        # M4: never tombstone past a failed index delete -- that
        # permanently orphans the leftover indexes (the retired slug can
        # never re-reach them). Leave the record 'deleting' so a
        # re-issued DELETE retries; every step here is itself idempotent.
        raise api_error(
            409,
            'project_busy',
            f"'{slug}' delete failed to remove {len(failed_indexes)} index(es); retry the delete",
            project=slug,
        )

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


__all__ = ['delete_project', 'delete_project_finish', 'dry_run_delete']
