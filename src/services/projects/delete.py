"""``dry_run_delete`` / ``delete_project`` / ``delete_project_finish`` (§4):
guarded project delete, completed by the ``project_delete`` job.

``delete_project`` validates and flips the project to ``deleting``; the route then starts the job
(:func:`start_project_delete`), whose handler runs :func:`delete_project_finish`. The project row's
status is the claim, and the job's exclusive scope (the slug) keeps two finishes apart.

Split out of ``lifecycle.py`` to stay under the repo's 700-LOC
pre-commit ratchet; ``lifecycle.py`` re-exports every public name here.
"""

from __future__ import annotations

import asyncio
import itertools
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException
from sqlalchemy import text

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import DEFAULT_SLUG, new_project_record
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.db.repos import projects as repo
from src.db.session import async_session_factory, project_tx
from src.schemas.curation.config_common import ModelSharingUser
from src.services.config_store.project_usage import model_dependents
from src.services.jobs.service import cancel_job, start_job
from src.services.projects.delete_models import _shared_model_users, _unload_owned_models
from src.services.projects.delete_paths import (
    _delete_dirs,
    _project_scoped_dirs,
    _validate_delete_paths,
)
from src.services.projects.mlflow_cleanup import MlflowCleanup, delete_experiment
from src.services.projects.records import record_from_row
from src.services.projects.registry import get_project_registry
from src.services.projects.store import get_row, run_repo
from src.services.rejections import rejection
from src.services.training.model_classes import owned_model_names


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.config.projects import ProjectRecord
    from src.services.jobs.service import StartResult

RowsDeleted = Callable[[str, int], Awaitable[None]]
"""Told after each committed batch: the table and the rows deleted from it so far."""


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
    from src.services.projects.lifecycle import _require_found, _resolve_existing
    from src.services.projects.lifecycle_checks import (
        _LAST_ACTIVE_MESSAGE,
        _other_active_slugs,
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
        except OpenSearchException:
            # m10: a genuinely uncountable index reports null, the same
            # rule ProjectCounts.validated already follows -- never a
            # made-up 0 that looks like "confirmed empty".
            docs = None
        indexes.append({'name': name, 'docs': docs, 'store_bytes': None})

    dirs: list[dict[str, Any]] = []
    for path, _shared_root in _project_scoped_dirs(record):
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
    promoted_models = await owned_model_names(record.slug)
    referenced_by: list[dict[str, str]] = []
    try:
        referenced_by = await model_dependents(record.slug, await _shared_model_users(record))
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


async def _wait_for_drain(record: ProjectRecord) -> bool:
    """Cancel the project's active jobs and wait until none is left (plan §4 step 3, trap T-27).

    The project was already flipped to ``deleting``, so the binder refuses any *new* write; this
    cancels the jobs that were already running (a ``cancelling`` job stops at its next boundary)
    and waits out the detection worker's last batch. Gives up after
    ``_DELETE_DRAIN_TIMEOUT_SECONDS``; the caller then rolls the delete back."""
    from src.services.projects.lifecycle_checks import running_jobs

    deadline = asyncio.get_running_loop().time() + _DELETE_DRAIN_TIMEOUT_SECONDS
    while True:
        busy_now = await running_jobs(record)
        if not busy_now:
            return True
        await _cancel_jobs(record, [j.id for j in busy_now if j.kind != 'detection_worker'])
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(_DELETE_DRAIN_POLL_SECONDS)


async def _cancel_jobs(record: ProjectRecord, job_ids: list[str]) -> None:
    """``cancel_job`` for each id; a job that already finished is not an error."""
    for job_id in job_ids:
        async with async_session_factory()() as session:
            try:
                await cancel_job(
                    session, uuid.UUID(job_id), record.id, requested_by='project_delete'
                )
            except RequestRejectedError as exc:
                if exc.detail.get('error') != 'job_finished':
                    raise


async def delete_project(
    *,
    slug: str,
    confirm: str | None,
    force: bool = False,
) -> ProjectRecord:
    """§4 guarded delete, background-completing (delta 10): the caller gets the ``deleting`` record
    back immediately (202), and :func:`delete_project_finish` runs the rest (drain wait, index and
    dir removal, retiring the slug) as a background task so a slow delete never blocks past a proxy
    timeout. This function only validates and flips the status."""
    from src.services.projects.lifecycle_checks import (
        _last_active_check,
        _require_transition,
        busy_message,
        running_jobs,
    )

    # Read the row ONCE, fresh from the database, and run every precondition check plus the write
    # against that same read's revision: a writer landing in between makes the write a 409
    # revision_conflict instead of racing through.
    record = record_from_row(await get_row(slug))

    if record.slug == DEFAULT_SLUG:
        raise rejection(
            409,
            'project_protected',
            'The default project can be archived but not deleted.',
            project=slug,
        )

    if record.status == 'deleting':
        # Retry of a failed finish (a failed index or model-unload step leaves the row
        # 'deleting'): the normal transition rule below refuses re-flipping it, so a retry needs
        # its own path. Every upfront check already passed the first time; re-running them here
        # would only block a legitimate retry. confirm is still required, so a bare accidental
        # DELETE cannot silently kick off a re-finish. The caller unconditionally re-schedules
        # delete_project_finish after this returns.
        if confirm is None:
            raise rejection(422, 'confirm_mismatch', 'confirm is required for a real delete')
        if confirm != slug:
            raise rejection(422, 'confirm_mismatch', f"confirm must equal the slug '{slug}'")
        return record

    # A 'building' row stale enough that its create almost certainly crashed (never a live one)
    # is deletable too; nothing else can ever flip a 'building' row out of 'building'.
    started_from_stale_building = record.status == 'building' and _is_stale_building(record)
    allowed = (
        _DELETABLE_STATUSES | frozenset({'building'})
        if started_from_stale_building
        else _DELETABLE_STATUSES
    )
    _require_transition(record, 'delete', allowed)
    if confirm is None:
        raise rejection(422, 'confirm_mismatch', 'confirm is required for a real delete')
    if confirm != slug:
        raise rejection(422, 'confirm_mismatch', f"confirm must equal the slug '{slug}'")

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

    shared_models = await _shared_model_users(record)
    if shared_models and not force:
        try:
            dependents = await model_dependents(record.slug, shared_models)
        except Exception as exc:
            raise rejection(
                503,
                'config_store_unavailable',
                "could not read every project to see who uses this project's shared models; "
                'retry, or pass force',
            ) from exc
        raise rejection(
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

    # pre_delete_status is what the rollback on a drain timeout restores. A delete that started
    # from a stale 'building' row rolls back to 'failed' (recoverable through the ordinary path),
    # never to 'building', which would recreate the wedge.
    row = await run_repo(
        'mark a project deleting',
        repo.mark_deleting,
        slug,
        from_states=allowed,
        pre_delete_status='failed' if started_from_stale_building else None,
        expected_revision=record.revision,
        action='delete',
    )
    await get_project_registry().ensure_fresh()
    return record_from_row(row)


@dataclass(frozen=True)
class DeleteFinished:
    """Result of a finished delete: the tombstone plus what became of the
    project's MLflow experiment (never silently skipped)."""

    record: ProjectRecord
    mlflow_cleanup: MlflowCleanup


# --- the batched row purge (review RR-7) ------------------------------------------------------

PURGE_BATCH_ROWS = 10_000
# The only place the immutability triggers' flag is set (tests/test_project_delete_flag_writers.py).
_DELETE_FLAG = text("SELECT set_config('op.project_delete', :pid, true)")
_ON_DELETE_OWNED = frozenset('acr')  # NO ACTION, CASCADE, RESTRICT: the row dies with its owner
_ON_DELETE_KEPT = frozenset('nd')  # SET NULL / SET DEFAULT: the row outlives its owner
_FOREIGN_KEYS = text(
    """
    SELECT cl.relname AS child, pl.relname AS parent, c.confdeltype AS on_delete,
      ARRAY(SELECT a.attname FROM unnest(c.conkey) WITH ORDINALITY k(num, ord)
            JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.num
            ORDER BY k.ord) AS child_cols,
      ARRAY(SELECT a.attname FROM unnest(c.confkey) WITH ORDINALITY k(num, ord)
            JOIN pg_attribute a ON a.attrelid = c.confrelid AND a.attnum = k.num
            ORDER BY k.ord) AS parent_cols
    FROM pg_constraint c
    JOIN pg_class cl ON cl.oid = c.conrelid
    JOIN pg_class pl ON pl.oid = c.confrelid
    WHERE c.contype = 'f' AND cl.relnamespace = 'public'::regnamespace
    """
)
_PROJECT_ID_TABLES = text(
    """
    SELECT c.relname FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
    WHERE a.attname = 'project_id' AND NOT a.attisdropped AND c.relkind IN ('r', 'p')
      AND c.relnamespace = 'public'::regnamespace
    """
)


@dataclass(frozen=True)
class ForeignKey:
    child: str
    parent: str
    on_delete: str
    child_cols: tuple[str, ...]
    parent_cols: tuple[str, ...]


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _owner_predicate(
    table: str,
    alias: str,
    by_child: Mapping[str, Sequence[ForeignKey]],
    direct: frozenset[str],
    aliases: Callable[[], int],
    seen: frozenset[str],
) -> str | None:
    """SQL true for the rows of ``table`` (as ``alias``) that belong to project ``:pid``.

    A table with a ``project_id`` column answers directly; any other reaches the project through
    a foreign key to an owning table (``job_items`` through ``jobs``), or to ``projects`` itself.
    ``None`` when the table is not owned by a project.
    """
    if table in direct:
        return f'{alias}.project_id = :pid'
    for fk in by_child.get(table, ()):
        if fk.on_delete not in _ON_DELETE_OWNED or fk.parent == table or fk.parent in seen:
            continue
        if fk.parent == 'projects':
            return f'{alias}.{_ident(fk.child_cols[0])} = :pid'
        parent_alias = f'p{aliases()}'
        inner = _owner_predicate(fk.parent, parent_alias, by_child, direct, aliases, seen | {table})
        if inner is None:
            continue
        join = ' AND '.join(
            f'{parent_alias}.{_ident(pc)} = {alias}.{_ident(cc)}'
            for cc, pc in zip(fk.child_cols, fk.parent_cols, strict=True)
        )
        return f'EXISTS (SELECT 1 FROM {_ident(fk.parent)} {parent_alias} WHERE {join} AND {inner})'  # nosec B608  names quoted from the catalog
    return None


def plan_purge(
    foreign_keys: Sequence[ForeignKey], project_id_tables: Sequence[str]
) -> list[tuple[str, str]]:
    """``(table, predicate)`` per owned table, children before parents (reverse foreign-key order).

    Read from the catalog rows, so a table a later revision adds is covered without an edit here.
    A table whose ``project_id`` is ``ON DELETE SET NULL`` outlives the project and is skipped.
    """
    kept = {
        fk.child
        for fk in foreign_keys
        if fk.parent == 'projects'
        and fk.child_cols == ('project_id',)
        and fk.on_delete in _ON_DELETE_KEPT
    }
    direct = frozenset(project_id_tables) - kept
    by_child: dict[str, list[ForeignKey]] = {}
    for fk in foreign_keys:
        by_child.setdefault(fk.child, []).append(fk)
    counter = itertools.count(1)
    predicates: dict[str, str] = {}
    candidates = {fk.child for fk in foreign_keys} | direct
    for table in sorted(candidates - kept):
        predicate = _owner_predicate(
            table, 't', by_child, direct, lambda: next(counter), frozenset()
        )
        if predicate is not None:
            predicates[table] = predicate
    # children first: a table is ready once no other owned table still references it
    waiting = {
        t: {
            fk.child
            for fk in foreign_keys
            if fk.parent == t and fk.child != t and fk.child in predicates
        }
        for t in predicates
    }
    ordered: list[str] = []
    while waiting:
        ready = sorted(t for t, refs in waiting.items() if not refs)
        if not ready:  # a reference cycle: the cascade is the safety net for what remains
            ready = sorted(waiting)
        for t in ready:
            ordered.append(t)
            del waiting[t]
        for refs in waiting.values():
            refs.difference_update(ready)
    return [(t, predicates[t]) for t in ordered]


async def purge_project_rows(
    project_id: uuid.UUID,
    *,
    on_rows_deleted: RowsDeleted | None = None,
    batch_rows: int = PURGE_BATCH_ROWS,
) -> dict[str, int]:
    """Delete the project's rows per table in reverse foreign-key order, ``batch_rows`` per
    transaction, and return the rows deleted per table.

    Every batch sets ``op.project_delete`` first: the immutability triggers honour it only while
    the project is ``deleting``. Re-running after a kill finds the finished tables empty and
    carries on at the table that was mid-way, so the cursor of the job is only progress.
    """
    async with async_session_factory()() as session:
        foreign_keys = [
            ForeignKey(r.child, r.parent, r.on_delete, tuple(r.child_cols), tuple(r.parent_cols))
            for r in (await session.execute(_FOREIGN_KEYS)).all()
        ]
        id_tables = [r.relname for r in (await session.execute(_PROJECT_ID_TABLES)).all()]
        await session.rollback()
    totals: dict[str, int] = {}
    for table, predicate in plan_purge(foreign_keys, id_tables):
        batch = text(
            f'WITH gone AS (DELETE FROM {_ident(table)} WHERE ctid = ANY(ARRAY('  # nosec B608  names quoted from the catalog, values bound
            f'SELECT t.ctid FROM {_ident(table)} t WHERE {predicate} LIMIT :n)) RETURNING 1) '
            'SELECT count(*) FROM gone'
        )
        while True:
            async with async_session_factory()() as session, project_tx(session, project_id):
                await session.execute(_DELETE_FLAG, {'pid': str(project_id)})
                deleted = (
                    await session.execute(batch, {'pid': project_id, 'n': batch_rows})
                ).scalar_one()
            if deleted:
                totals[table] = totals.get(table, 0) + deleted
                if on_rows_deleted is not None:
                    await on_rows_deleted(table, totals[table])
            if deleted < batch_rows:
                break
    return totals


async def delete_project_finish(
    client: Any, *, slug: str, on_rows_deleted: RowsDeleted | None = None
) -> DeleteFinished:
    """Steps 3-9 of the guarded delete (§4): validate every dir path, drain the project's jobs,
    unload its own promoted models, delete the exact indexes, remove the dirs, soft-delete the
    MLflow experiment over REST (outcome reported in :class:`DeleteFinished`), delete the control
    rows table by table (:func:`purge_project_rows`), then in one transaction retire the slug and
    delete the row.

    Path validation runs FIRST, before any irreversible step. After a successful drain, and still
    before any irreversible step, this claims the finish with a revision-guarded UPDATE of the
    ``deleting`` row: of two finishers reading one revision, exactly one wins, in every process.

    Model unload runs before index deletion: once the indexes are gone there is no cheap step back
    if unload then fails, where retrying an unload against an already-unloaded model is a clean
    no-op (``ModelNotPromotedError``, handled in :func:`_unload_owned_models`).

    Idempotent: safe to re-run after a crash between any two steps, because every step is itself
    idempotent (model unload tolerates "already gone", index delete uses ``ignore=[404]``,
    ``rmtree(ignore_errors=True)``, a purge finds its finished tables empty, and a finished delete
    is a retired slug)."""
    try:
        record = record_from_row(await get_row(slug))
    except RequestRejectedError:
        if slug in get_project_registry().retired_slugs():
            return DeleteFinished(
                replace(new_project_record(slug, base_curation_config()), status='deleted'),
                MlflowCleanup('done', 'already deleted; cleaned up by the first finish'),
            )
        raise

    # Validate every dir BEFORE any irreversible step (index deletion, below) runs.
    _validate_delete_paths(record)

    drained = await _wait_for_drain(record)
    if not drained:
        # Roll back to the status delete found. `from_states=('deleting',)` refuses the
        # rollback if some other writer already resolved this row.
        await run_repo('roll back a delete', repo.rollback_delete, slug)
        await get_project_registry().ensure_fresh()
        raise rejection(
            409, 'project_busy', f"'{slug}' did not drain within the timeout", project=slug
        )

    # Claim the finish, after the drain succeeded and before the first irreversible step. A fresh
    # read says what the row is now; if another finisher already resolved it the status is no
    # longer 'deleting' and this raises 409 invalid_transition before anything destructive runs;
    # two claims reading one revision cannot both win.
    fresh = await get_row(slug)
    claimed = record_from_row(
        await run_repo(
            'claim a project delete',
            repo.claim_delete,
            slug,
            expected_revision=fresh.revision,
        )
    )
    record = claimed

    failed_models = await _unload_owned_models(record)
    if failed_models:
        # Same rule as the index-delete failure below: leave the row 'deleting' (retryable via
        # the retry path in delete_project) rather than retiring past a model still live in
        # Triton.
        raise rejection(
            409,
            'project_busy',
            f"'{slug}' delete failed to unload {len(failed_models)} model(s); retry the delete",
            project=slug,
        )

    failed_indexes = await _delete_indexes(client, record)
    if failed_indexes:
        # Never retire past a failed index delete -- that permanently orphans the leftover
        # indexes (the retired slug can never re-reach them). Leave the row 'deleting' so a
        # re-issued DELETE retries; every step here is itself idempotent.
        raise rejection(
            409,
            'project_busy',
            f"'{slug}' delete failed to remove {len(failed_indexes)} index(es); retry the delete",
            project=slug,
        )

    await _delete_dirs(record)
    mlflow_cleanup = await delete_experiment(record.resources.mlflow_experiment)
    if record.id is not None:
        await purge_project_rows(record.id, on_rows_deleted=on_rows_deleted)

    # One transaction: retire the slug, delete the row (the cascade stays as the safety net for
    # any row the purge did not reach), bump the registry counter. `from_states=('deleting',)`
    # makes a duplicate finish fail instead of deleting twice.
    await run_repo('finish a project delete', repo.finish_delete, slug)
    await get_project_registry().ensure_fresh()

    from src.services.projects.capacity import invalidate_capacity_cache

    invalidate_capacity_cache()  # the delete just freed this project's shards

    logger.info(
        'project_deleted',
        project=slug,
        mlflow_cleanup=mlflow_cleanup.outcome,
        mlflow_cleanup_reason=mlflow_cleanup.reason,
    )
    return DeleteFinished(replace(record, status='deleted'), mlflow_cleanup)


async def start_project_delete(session: AsyncSession, slug: str) -> StartResult:
    """Queue the ``project_delete`` job of ``slug`` (deployment-level, exclusive per slug).

    Raises the 409 ``job_busy`` refusal when a finish is already queued or running."""
    return await start_job(session, 'project_delete', None, params={'slug': slug}, scope=slug)


__all__ = [
    'DeleteFinished',
    'delete_project',
    'delete_project_finish',
    'dry_run_delete',
    'purge_project_rows',
    'start_project_delete',
]
