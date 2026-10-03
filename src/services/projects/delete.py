"""``dry_run_delete`` / ``delete_project`` / ``delete_project_finish`` (§4):
guarded project delete, background-completing.

Split out of ``lifecycle.py`` to stay under the repo's 700-LOC
pre-commit ratchet; ``lifecycle.py`` re-exports every public name here so
every existing caller (``lifecycle.delete_project(...)``, etc.) keeps
working unchanged.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from src.config.project_context import bind_project
from src.config.projects import DEFAULT_SLUG, ProjectStatus
from src.core.logging import get_logger
from src.routers.curation._config_common_models import ModelSharingUser, api_error
from src.services.config_store.project_usage import model_dependents
from src.services.projects.registry import get_project_registry, get_record_with_seq


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


logger = get_logger(__name__)

# How long delete waits for the detection worker's per-project inflight
# count to drain before giving up and rolling back (plan §4 step 3).
_DELETE_DRAIN_TIMEOUT_SECONDS = 60.0
_DELETE_DRAIN_POLL_SECONDS = 1.0

_DELETABLE_STATUSES = frozenset({'active', 'archived', 'failed'})

# N1: a 'building' record whose own create crashed (or hung) before ever
# reaching a 'failed'/'active' write is wedged forever under m3's normal
# transition rule -- no in-process exception handler could ever run for
# a real process crash. A 'building' record older than this is treated
# as dead and becomes deletable; a fresh one might still be a live
# create in progress, so it stays refused.
_BUILDING_STALE_SECONDS = 120.0

# P3F pass-3 MA1 probe 2: a re-DELETE issued while a first finish's
# drain wait is still running must never let a SECOND finish run to
# completion concurrently -- both would independently decide the
# record's fate (rollback vs. tombstone) with no coordination between
# them, and whichever writes last wins, including a finish that deletes
# a project its own drain-timed-out sibling just rolled back to
# 'active'.
#
# P3F pass-4 F1 correction: this guard is PER-WORKER-PROCESS ONLY --
# yolo-api runs `--workers=32` (docker-compose.yml), and each worker has
# its own, separate, empty copy of this set. It stops the race only
# within the one worker process that happens to handle both the
# original DELETE and its retry. A retried DELETE that a proxy/client
# lands on a DIFFERENT worker (31 times out of 32 in production) sees an
# empty guard here and would previously have run its own finish to
# completion regardless of what a sibling finish on another worker had
# already done to the same record -- the review's cross-worker probe
# showed exactly this: finish A timed out and rolled the record back to
# 'active', and finish B (on a simulated second worker, unaware of A)
# went on to unload the models and delete all 7 indexes of what was, by
# then, an 'active' project.
#
# The actual cross-process protection is the claim write in
# `delete_project_finish`, right after the drain succeeds and before the
# first irreversible step: it re-reads the record with
# `_refetch_for_write(expect_status='deleting')` and writes it back
# through the OCC-guarded `write_record`. That is a real cross-process
# mutual-exclusion primitive built on OpenSearch's own document
# versioning (i.e. the same machinery `expect_status` already uses
# elsewhere in this module), so it works across all 32 workers -- not
# just this in-memory set, which remains useful only as a fast,
# zero-round-trip guard against a duplicate finish racing itself within
# one worker.
_FINISH_IN_PROGRESS: set[str] = set()


def _is_stale_building(record: ProjectRecord) -> bool:
    """Whether ``record`` (assumed ``status == 'building'``) has been
    sitting long enough that its create almost certainly crashed rather
    than still running. An unparsable ``updated_at`` is treated
    conservatively as NOT stale (fail-closed: refuse the delete rather
    than risk racing a live create whose timestamp we can't read)."""
    try:
        updated = datetime.fromisoformat(record.updated_at)
    except ValueError:
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return (datetime.now(UTC) - updated).total_seconds() > _BUILDING_STALE_SECONDS


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
            docs: int | None = int(count_resp.get('count') or 0)
        except Exception:
            # m10: a genuinely uncountable index reports null, the same
            # rule ProjectCounts.validated already follows -- never a
            # made-up 0 that looks like "confirmed empty".
            docs = None
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

    # P3F pass-3 MA2: report EVERY model this project owns (private and
    # shared alike), not just the shared subset -- the real delete's
    # unload step (_unload_owned_models) now unloads this same full set
    # unconditionally. `_shared_model_users` narrows to the
    # cross-project-sharing subset the `in_use` refusal cares about;
    # that is a strict SUBSET of ownership, not the ownership report
    # itself. Previously this was always `[]` for a private-only
    # project, even though those models were never unloaded either.
    promoted_models = await _owned_models(record)
    referenced_by: list[dict[str, str]] = []
    try:
        referenced_by = await model_dependents(
            client, record.slug, await _shared_model_users(record)
        )
    except Exception:
        blocking.append(
            {
                'code': 'config_store_unavailable',
                'message': "could not read every project to see who uses this project's models",
            }
        )

    return {
        'indexes': indexes,
        'dirs': dirs,
        'promoted_models': promoted_models,
        'mlflow_experiment': record.resources.mlflow_experiment,
        'running_jobs': [j.to_wire() for j in jobs],
        'referenced_by': referenced_by,
        'blocking': [b['code'] for b in blocking],
        'blocking_detail': blocking,
    }


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


def _require_project_scoped_path(path: Path, shared_root: Path, slug: str) -> None:
    """m1 / P3F pass-3 m-a: require ``path`` to resolve to
    ``<shared_root>/<slug>`` itself, or a proper descendant of it --
    never ``shared_root`` itself (a corrupted/hand-edited registry
    ``resources`` record pointing straight at the multi-project root
    would otherwise let a delete wipe every sibling project's dir), and
    never a sibling project's own ``<shared_root>/<other_slug>`` tree.

    The prior guard (``_rm_dir_guarded``, ``_path_within``) accepted
    ``path == shared_root`` for 6 of the project's 8 dirs -- only
    ``train_jobs_dir``/``autolabel_dir`` got a real per-slug check. This
    is the one guard every dir now goes through, each with its own
    independently-computed ``shared_root`` (never derived from the path
    being checked), and it always raises the plan's ``path_escape``
    code, never ``internal_isolation_error``."""
    project_root = (shared_root / slug).resolve()
    try:
        path.resolve().relative_to(project_root)
    except ValueError:
        logger.error('project_delete_path_escape', path=str(path), expected_root=str(project_root))
        raise api_error(500, 'path_escape', f'refusing to delete outside {project_root}') from None


def _project_scoped_dirs(record: ProjectRecord) -> list[tuple[Path, Path]]:
    """Every one of the project's 8 own dirs, paired with the shared
    root each must resolve strictly inside (§4 step 6: "a path outside
    refuses"). A new project's dirs are siblings under
    ``OP_PROJECTS_DATA_ROOT`` (exports, class registry, bakeoff eval) or
    under the deployment ``state_dir``/jobs roots (uploads, job dirs,
    autolabel state) -- never under ``default``'s own dirs, which
    :func:`resources_for_new` never nests anything into."""
    from src.config.curation import base_curation_config
    from src.config.projects import projects_data_root, trainer_jobs_root

    base = base_curation_config()
    data_root = projects_data_root()
    state_projects_root = base.state_dir / 'projects'
    autolabel_projects_root = (
        Path(os.environ.get('OP_AUTO_LABEL_STATE_DIR', '/jobs/auto_label')) / 'projects'
    )
    return [
        (record.resources.export_root, data_root),
        (record.resources.class_registry_path.parent, data_root),
        (record.resources.bakeoff_eval_root, data_root),
        (record.resources.project_state_dir, state_projects_root),
        (record.resources.upload_root, state_projects_root),
        (record.resources.bakeoff_jobs_dir, state_projects_root),
        (record.resources.train_jobs_dir, trainer_jobs_root() / 'projects'),
        (record.resources.autolabel_dir, autolabel_projects_root),
    ]


def _validate_delete_paths(record: ProjectRecord) -> None:
    """P3F pass-3 m-a: validate every one of the project's 8 dirs BEFORE
    any irreversible step runs. The prior ordering ran this check (for
    the 2 dirs it covered) only from inside ``_delete_dirs``, itself
    called AFTER ``_delete_indexes`` had already irreversibly deleted
    the project's OpenSearch indexes -- a ``path_escape`` raised there
    left the record wedged ``deleting`` forever (every retry hits the
    same escape again) with the indexes already gone. Called as a
    preflight in :func:`delete_project_finish`, before the drain wait,
    model unload or index delete."""
    for path, shared_root in _project_scoped_dirs(record):
        _require_project_scoped_path(path, shared_root, record.slug)


async def _delete_dirs(record: ProjectRecord) -> None:
    """Remove every per-project dir. Paths are already validated by
    :func:`_validate_delete_paths` earlier in
    :func:`delete_project_finish`; re-validate here too (cheap, and this
    function has its own unit-test callers) rather than trusting that
    nothing mutated ``record.resources`` in between."""
    _validate_delete_paths(record)
    for path, _shared_root in _project_scoped_dirs(record):
        shutil.rmtree(path, ignore_errors=True)


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
        _get_mutable_record,
        _last_active_check,
        _now,
        _require_transition,
        running_jobs,
        write_record,
    )

    # MA1: read the record ONCE, fresh (bypassing the registry's
    # in-process cache -- ``_get_mutable_record`` is a direct
    # get-by-id), and run every precondition check plus the write
    # itself against that SAME read's seq/term. The prior version
    # checked preconditions against a possibly-stale registry snapshot
    # (``_resolve_existing``) and then re-read fresh immediately before
    # the write -- a re-read taken right before a write always has a
    # trivially-current seq/term, so OCC could never catch a doc that
    # changed between the check and the write. Using ONE read for both
    # means a real race (another writer landing in between) now
    # genuinely conflicts (409 revision_conflict) instead of silently
    # racing through.
    record, seq, term = await _get_mutable_record(client, slug)
    if record.status == 'deleted':
        raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)

    if record.slug == DEFAULT_SLUG:
        raise api_error(
            409,
            'project_protected',
            'The default project can be archived but not deleted.',
            project=slug,
        )

    if record.status == 'deleting':
        # M4 retry: a failed finish (a failed index or model-unload step)
        # leaves the record 'deleting', retryable -- but m3's normal
        # transition rule below refuses re-flipping a 'deleting' record,
        # so a retry needs its own path. Every upfront check (busy,
        # last-active, shared-model) already passed the first time a
        # real delete flipped this record to 'deleting'; re-running them
        # here would only block a legitimate retry. confirm is still
        # required, so a bare accidental DELETE can't silently kick off a
        # re-finish. The caller (the router) unconditionally re-schedules
        # delete_project_finish after this returns, so returning the
        # unchanged record is sufficient to retry.
        if confirm is None:
            raise api_error(422, 'confirm_mismatch', 'confirm is required for a real delete')
        if confirm != slug:
            raise api_error(422, 'confirm_mismatch', f"confirm must equal the slug '{slug}'")
        return record

    # N1 escape hatch: a 'building' record stale enough that its create
    # almost certainly crashed (never a live one -- see
    # _is_stale_building) is deletable too. m3's normal rule
    # (active|archived|failed -> deleting) would otherwise wedge it
    # forever: nothing else can ever flip a 'building' record out of
    # 'building'.
    started_from_stale_building = record.status == 'building' and _is_stale_building(record)
    allowed = (
        _DELETABLE_STATUSES | frozenset({'building'})
        if started_from_stale_building
        else _DELETABLE_STATUSES
    )
    _require_transition(record, 'delete', allowed)
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

    shared_models = await _shared_model_users(record)
    if shared_models and not force:
        try:
            dependents = await model_dependents(client, record.slug, shared_models)
        except Exception as exc:
            raise api_error(
                503,
                'config_store_unavailable',
                "could not read every project to see who uses this project's shared models; "
                'retry, or pass force',
            ) from exc
        raise api_error(
            409,
            'in_use',
            f"'{slug}' has {len(shared_models)} model(s) opted into cross-project "
            f'sharing ({", ".join(shared_models)}); deleting could break another project '
            'that depends on them',
            project=slug,
            projects=sorted({d['project'] for d in dependents}),
            used_by=[
                ModelSharingUser(project=d['project'], profile=d['profile']) for d in dependents
            ],
        )
    if shared_models:
        logger.warning(
            'project_delete_forced_past_shared_models',
            project=slug,
            shared_models=shared_models,
        )

    # m2: pre_delete_status is what M3's rollback (below, in
    # delete_project_finish) restores on a drain timeout -- if this
    # delete started from a stale 'building' record, restoring
    # 'building' would recreate the exact N1 wedge, so land any rollback
    # in 'failed' (recoverable through the ordinary path) instead of the
    # record's real prior status.
    pre_delete_status = 'failed' if started_from_stale_building else record.status
    deleting = replace(
        record, status='deleting', pre_delete_status=pre_delete_status, updated_at=_now()
    )
    await write_record(client, deleting, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return deleting


async def _owned_models(record: ProjectRecord) -> list[str]:
    """Every Triton model this project owns (``promote.json.project ==
    record.slug``), private and shared alike -- plan §4 step 4's "the
    owned Triton models". This is the FULL ownership enumeration: a
    project's own models are always cleaned up on a real delete
    regardless of whether they opted into cross-project sharing.
    :func:`_shared_model_users` narrows this to the ``shared=True``
    subset for the ``in_use`` refusal only -- ``force`` bypasses THAT
    refusal (a shared model another project might depend on), never
    whether unload runs at all."""
    from src.services.training.model_classes import model_owner_project
    from src.services.training.triton_promote import resolve_triton_models_dir

    models_dir = resolve_triton_models_dir()
    if not models_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in models_dir.iterdir()
        if entry.is_dir() and model_owner_project(entry.name) == record.slug
    )


async def _shared_model_users(record: ProjectRecord) -> list[str]:
    """§5.5 in_use guard: which of this project's own promoted models
    have opted into cross-project sharing (``PUT /models/{name}/sharing``,
    ``promote.json.shared``)? A strict subset of :func:`_owned_models` --
    used ONLY to decide the ``in_use`` refusal (bypassable with
    ``force=True``), never to decide whether unload runs (see
    :func:`_unload_owned_models`, which unloads every owned model,
    shared or not).

    A project with any shared model is refused unless ``force=True``:
    "opted into sharing" is itself evidence someone may depend on it.
    :func:`~src.services.config_store.project_usage.model_dependents` names the projects that actually depend on
    one today (their ACTIVE detection profile names the model).
    """
    from src.services.training.model_classes import is_model_shared

    return [name for name in await _owned_models(record) if is_model_shared(name)]


async def _unload_owned_models(record: ProjectRecord) -> list[str]:
    """M5 step 4 / P3F pass-3 MA2: unload every one of this project's
    own promoted models -- private and shared alike (:func:`_owned_models`,
    the full ownership enumeration; NOT :func:`_shared_model_users`,
    which only narrows the ``in_use`` refusal) -- from Triton and remove
    their model repo directories, reusing P2's own unload primitive
    (``src.services.training.triton_promote.unload_triton_model``, the
    same one ``DELETE /models/{name}`` calls) rather than reimplementing
    Triton model removal here. A project's own models are ALWAYS cleaned
    up on delete; ``force`` only bypasses the upfront ``in_use`` 409 for
    the shared subset, never whether this step runs.

    Mirrors :func:`_delete_indexes`: collects (never swallows) failures
    so the caller can leave the record retryable instead of tombstoning
    a project some of whose models are still live in Triton."""
    from src.services.training.triton_promote import ModelNotPromotedError, unload_triton_model

    failed: list[str] = []
    for name in await _owned_models(record):
        try:
            await unload_triton_model(name)
        except ModelNotPromotedError:
            # Already gone (e.g. a prior finish attempt's unload
            # succeeded before a later step in that same run failed) --
            # idempotent no-op, not a failure to retry.
            continue
        except Exception as exc:
            logger.warning(
                'project_delete_model_unload_failed',
                project=record.slug,
                model=name,
                error=str(exc),
            )
            failed.append(name)
    return failed


async def delete_project_finish(client: Any, *, slug: str) -> ProjectRecord:
    """Steps 3-9 of the guarded delete (§4): validate every dir path,
    drain wait, unload the project's own promoted models, delete the
    exact indexes, remove the dirs, soft-delete the MLflow experiment if
    reachable, tombstone.

    Path validation (m-a) runs FIRST, before any irreversible step --
    previously it ran only inside the dir-removal step, itself after
    index deletion had already run, so a ``path_escape`` left the
    record wedged ``deleting`` with the indexes already gone and no way
    back (every retry hits the same escape again).

    Immediately after a successful drain wait, and still before any
    irreversible step, this claims exclusive ownership of the finish
    with an OCC-guarded re-read-and-write (P3F pass-4 F1) -- the one
    protection here that actually holds across ``--workers=32``
    processes, not just within this one (``_FINISH_IN_PROGRESS`` above
    is per-process only).

    Model unload runs before index deletion, not alongside or after:
    once the indexes are gone there is no cheap step back if unload then
    fails, where retrying an unload against an already-unloaded model is
    a clean, idempotent no-op (``ModelNotPromotedError``, handled in
    :func:`_unload_owned_models`).

    Idempotent: safe to re-run after a crash between any two steps,
    because every step here is itself idempotent (model unload tolerates
    "already gone", index delete uses ``ignore=[404]``,
    ``rmtree(ignore_errors=True)``, and the final ``deleted`` write is
    OCC-guarded so a duplicate run is a no-op)."""
    from src.services.projects.lifecycle import _now, _refetch_for_write, write_record

    stored, _, _ = await get_record_with_seq(client, slug)
    if stored is None:
        raise api_error(404, 'project_not_found', f"no project named '{slug}'", project=slug)
    record = stored
    if record.status == 'deleted':
        return record

    if slug in _FINISH_IN_PROGRESS:
        raise api_error(
            409,
            'finish_in_progress',
            f"a delete finish is already running for '{slug}'",
            project=slug,
        )
    _FINISH_IN_PROGRESS.add(slug)
    try:
        # m-a: validate every dir BEFORE any irreversible step (index
        # deletion, below) runs.
        _validate_delete_paths(record)

        drained = await _wait_for_drain(record)
        if not drained:
            # M3: roll back to the status delete found it in, not always
            # 'failed'. m2: base the rollback doc on a fresh read (the
            # drain wait can run up to _DELETE_DRAIN_TIMEOUT_SECONDS), not
            # the possibly-stale `record` read at the top of this
            # function. MA1: `expect_status='deleting'` refuses the
            # rollback if the fresh read is no longer 'deleting' (e.g.
            # some other writer already resolved this record) instead of
            # blindly overwriting a status this call never validated.
            rolled_back, seq, term = await _refetch_for_write(
                client,
                slug,
                expect_status='deleting',
                status=cast('ProjectStatus', record.pre_delete_status) or 'failed',
                pre_delete_status=None,
                updated_at=_now(),
            )
            await write_record(client, rolled_back, if_seq_no=seq, if_primary_term=term)
            await get_project_registry().ensure_fresh()
            raise api_error(
                409, 'project_busy', f"'{slug}' did not drain within the timeout", project=slug
            )

        # F1: claim exclusive ownership of this finish, right here --
        # after the drain succeeded, before the first irreversible step
        # (model unload, below). `_FINISH_IN_PROGRESS` above is per-
        # worker-process only, so a finish running on a DIFFERENT worker
        # (the common case under `--workers=32`) has its own, empty copy
        # and would otherwise have no way to know that a sibling finish
        # for this same slug already rolled the record back to 'active'
        # (M3 drain timeout) or otherwise moved it on. Re-reading fresh
        # here with `expect_status='deleting'` makes THIS read the one
        # thing that decides who still owns the finish: if some other
        # finish already resolved the record, the fresh status is no
        # longer 'deleting' and this raises 409 `invalid_transition`
        # before anything destructive runs. If two finishes' claim reads
        # both see 'deleting' and race each other's writes, only one
        # write's seq/term is still current by the time it executes --
        # the other gets `RevisionConflictError`, translated by
        # `write_record` into 409 `revision_conflict`. Either way this is
        # a genuine cross-process mutual-exclusion primitive built on
        # OpenSearch's own document versioning, not an in-memory guard
        # only one process can see. `record` is reassigned to the freshly
        # claimed doc so every step below works from the same read that
        # won this race, not the possibly-stale read from the top of this
        # function.
        claimed, seq, term = await _refetch_for_write(
            client, slug, expect_status='deleting', updated_at=_now()
        )
        await write_record(client, claimed, if_seq_no=seq, if_primary_term=term)
        record = claimed

        failed_models = await _unload_owned_models(record)
        if failed_models:
            # Same M4 rule as the index-delete failure below: leave the
            # record 'deleting' (retryable via the M4 retry path in
            # delete_project) rather than tombstoning past a model still
            # live in Triton.
            raise api_error(
                409,
                'project_busy',
                f"'{slug}' delete failed to unload {len(failed_models)} model(s); retry the delete",
                project=slug,
            )

        failed_indexes = await _delete_indexes(client, record)
        if failed_indexes:
            # M4: never tombstone past a failed index delete -- that
            # permanently orphans the leftover indexes (the retired slug
            # can never re-reach them). Leave the record 'deleting' so a
            # re-issued DELETE retries; every step here is itself
            # idempotent.
            raise api_error(
                409,
                'project_busy',
                f"'{slug}' delete failed to remove {len(failed_indexes)} index(es); "
                'retry the delete',
                project=slug,
            )

        await _delete_dirs(record)
        await _soft_delete_mlflow(record)

        # MA1: expect_status='deleting' -- the same defense as the
        # rollback above, on the write that makes the tombstone
        # permanent.
        tombstoned, seq, term = await _refetch_for_write(
            client, slug, expect_status='deleting', status='deleted', updated_at=_now()
        )
        await write_record(client, tombstoned, if_seq_no=seq, if_primary_term=term)
        await get_project_registry().ensure_fresh()

        from src.services.projects.capacity import invalidate_capacity_cache

        invalidate_capacity_cache()  # m7: the delete just freed this project's shards

        logger.info('project_deleted', project=slug)
        return tombstoned
    finally:
        _FINISH_IN_PROGRESS.discard(slug)


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
