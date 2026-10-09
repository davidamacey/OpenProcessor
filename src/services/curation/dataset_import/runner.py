"""The import job: claim, run in chunks, resume, cancel (W10.11).

Runs in-process as an ``asyncio.Task`` in one ``api`` worker; every
fact another process needs lives on disk (:mod:`.store`), so status, cancel
and resume work from whichever worker a request lands on.

A resume consumes what the START persisted: the resolved mapping with its
real class ids and the pinned region profile. It never re-resolves them
against the registry or config store of the resuming process.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.project_context import ensure_marked_dir
from src.core.logging import get_logger
from src.services.curation.dataset_import import limits
from src.services.curation.dataset_import.actions import resume_blocker, undo_blocker
from src.services.curation.dataset_import.chunk import import_chunk
from src.services.curation.dataset_import.holdout_record import mark_freeze_undone
from src.services.curation.dataset_import.prepare import (
    PinnedProfile,
    PreparedImport,
    mapping_from_dict,
    mapping_to_dict,
    materialize_created_classes,
)
from src.services.curation.dataset_import.report import ImportReport
from src.services.curation.dataset_import.scan import ScanEntry, scan_dataset, source_sha
from src.services.curation.dataset_import.store import (
    ACTIVE_STATUSES,
    COMPLETED_STATUSES,
    RESUMABLE_STATUSES,
    ImportStore,
    active_import,
    imports_root,
    latest_with_key,
    new_import_id,
    now_iso,
    open_store,
)
from src.services.curation.dataset_import.undo import UndoContext, undo_import
from src.services.curation.file_job import heartbeat_ticker
from src.services.curation.job_lock import exclusive_start_lock
from src.services.curation.next_steps import cluster_regions


if TYPE_CHECKING:
    from src.clients.curation_opensearch.registry import ClassRegistry
    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.options import DatasetImportRequest
    from src.services.curation.dataset_import.paths import PathGuard

logger = get_logger(__name__)

_SPLIT_ORDER = {'train': 0, 'val': 1, 'test': 2}
_tasks: dict[str, asyncio.Task[None]] = {}


class ImportBusyError(Exception):
    def __init__(self, import_id: str | None) -> None:
        super().__init__(f'an import is already running: {import_id}')
        self.import_id = import_id


class ImportResumableError(Exception):
    def __init__(self, import_id: str) -> None:
        super().__init__(f'import {import_id} can be resumed')
        self.import_id = import_id


class ImportNotResumableError(Exception):
    pass


class ImportNotUndoableError(Exception):
    pass


class DatasetChangedError(Exception):
    pass


def ordered_entries(entries: list[ScanEntry]) -> list[ScanEntry]:
    """Chunk order: split, then relative path. Stable across restarts."""
    return sorted(entries, key=lambda e: (_SPLIT_ORDER.get(e.split or '', 3), e.rel_path))


def chunked(entries: list[ScanEntry]) -> list[list[ScanEntry]]:
    size = limits.import_chunk_size()
    return [entries[i : i + size] for i in range(0, len(entries), size)]


# ------------------------------------------------------------------ claiming


def reusable_prior(prepared: PreparedImport) -> tuple[ImportStore, str] | None:
    """``(store, status)`` of an earlier import of this same request, by its
    key (or by ``reuse_key`` when its created classes now exist)."""
    for key in (prepared.import_key, prepared.reuse_key):
        if key is None:
            continue
        previous = latest_with_key(key)
        if previous is not None:
            return previous, previous.job.read().get('status', '')
    return None


def claim_import(
    request: DatasetImportRequest, prepared: PreparedImport
) -> tuple[ImportStore, bool]:
    """Atomically claim a new import (or return the completed one).

    Returns ``(store, reused)``. Raises :class:`ImportBusyError` while any
    import of the project is live and :class:`ImportResumableError` when the
    same key has an interrupted/failed/cancelled run.
    """
    root = ensure_marked_dir(imports_root())
    with exclusive_start_lock(root / 'start.lock') as acquired:
        if not acquired:
            raise ImportBusyError(None)
        live = active_import()
        if live is not None:
            raise ImportBusyError(live.import_id)
        previous = latest_with_key(prepared.import_key)
        if previous is not None:
            status = previous.job.read().get('status')
            if status in COMPLETED_STATUSES:
                return previous, True
            if status in RESUMABLE_STATUSES:
                raise ImportResumableError(previous.import_id)
        store = ImportStore(root / new_import_id(prepared.import_key))
        store.write_request(request.model_dump(mode='json'))
        store.job.write(
            {
                'import_id': store.import_id,
                'import_key': prepared.import_key,
                'project': prepared.view.project,
                'name': request.options.name,
                'status': 'queued',
                'mode': 'import',
                'source_sha': prepared.source_sha,
                'source_format': prepared.scan.format,
                'source_root': str(prepared.scan.root),
                'started_at': now_iso(),
                'updated_at': now_iso(),
            }
        )
        store.job.touch_heartbeat()
    return store, False


# ----------------------------------------------------------------- execution


def persist_scan_summary(store: ImportStore, prepared: PreparedImport) -> None:
    entries = prepared.scan.entries
    store.write_scan(
        {
            'images_total': len(entries),
            'chunks_total': len(chunked(entries)),
            'source_sha': prepared.source_sha,
            'boxes': sum(len(e.boxes) for e in entries),
            'issues': [
                {'code': i.code, 'count': i.count, 'severity': i.severity} for i in prepared.issues
            ],
        }
    )


def persist_pinned(store: ImportStore, prepared: PreparedImport) -> None:
    store.write_mapping(
        {
            **mapping_to_dict(prepared.resolved),
            'profile': prepared.view.profile.to_dict() if prepared.view.profile else None,
            'parents': prepared.parents,
        }
    )
    op = prepared.scan.op_export
    if op is not None:
        store.job.update(
            test_split={
                'test_label_sha': op.test_frozen.test_label_sha,
                'source_frozen_test_sha': op.frozen_test_sha,
            }
        )


def freeze_default(prepared: PreparedImport, request: DatasetImportRequest) -> bool:
    explicit = request.options.freeze_test_split
    if explicit is not None:
        return explicit
    op = prepared.scan.op_export
    return bool(getattr(op, 'freeze_test_split_default', False))


async def _backpressure(ctx: ImportContext, store: ImportStore) -> None:
    """Pause while the region worker is behind (propose / detected parents)."""
    if not ctx.uses_detector or ctx.profile is None:
        return
    from src.config.region_state import RegionStatus
    from src.services.curation.region_drain import region_drain_poll_interval_s

    fields = ctx.region_fields
    while True:
        resp = await ctx.opensearch.count(
            index=ctx.items_index,
            body={
                'query': {
                    'terms': {
                        fields.status: [
                            RegionStatus.PENDING_DETECTION.value,
                            RegionStatus.PENDING_VERIFICATION.value,
                        ]
                    }
                }
            },
        )
        if int(resp.get('count', 0)) <= limits.import_max_pending():
            break
        store.job.update(status='paused_backpressure', waiting_for='region_worker')
        if store.job.cancel_requested():
            return
        await asyncio.sleep(region_drain_poll_interval_s())
    store.job.update(status='running', waiting_for=None)


async def run_import_job(ctx: ImportContext, store: ImportStore, entries: list[ScanEntry]) -> None:
    """Run every chunk not yet recorded in ``chunks_done.jsonl``."""
    job = store.job
    chunks = chunked(entries)
    done = store.chunks_done()
    report = ImportReport()
    for r in done.values():
        report.add(ImportReport.from_dict(r))
    job.update(
        status='running',
        images_total=len(entries),
        chunks_total=len(chunks),
        chunks_done=len(done),
        report=report.to_dict(),
        waiting_for=None,
    )
    job.touch_heartbeat()
    ticker = asyncio.create_task(heartbeat_ticker(job))
    consecutive_failures = 0
    failed_chunks: list[int] = []
    started = asyncio.get_running_loop().time()
    try:
        for idx, chunk_entries in enumerate(chunks):
            if idx in done:
                continue
            if job.cancel_requested():
                job.update(status='cancelled', finished_at=now_iso(), poll_after_s=None)
                return
            await _backpressure(ctx, store)
            if job.cancel_requested():
                job.update(status='cancelled', finished_at=now_iso(), poll_after_s=None)
                return
            try:
                chunk_report = await import_chunk(ctx, store, idx, chunk_entries)
            except Exception as exc:
                logger.error('dataset_import_chunk_failed', chunk=idx, error=str(exc))
                store.append_issues(
                    [{'code': 'chunk_failed', 'chunk': idx, 'error': str(exc)[:300]}]
                )
                failed_chunks.append(idx)
                consecutive_failures += 1
                if consecutive_failures > limits.import_max_failed_chunks():
                    job.update(
                        status='failed',
                        error=f'{consecutive_failures} consecutive chunks failed: {exc}'[:300],
                        finished_at=now_iso(),
                        poll_after_s=None,
                    )
                    return
                continue
            consecutive_failures = 0
            report.add(chunk_report)
            store.mark_chunk_done(idx, chunk_report.to_dict())
            done[idx] = chunk_report.to_dict()
            elapsed = max(asyncio.get_running_loop().time() - started, 1e-6)
            images_done = sum(
                r.get('images_created', 0) + r.get('images_reused', 0) for r in done.values()
            )
            job.update(
                chunks_done=len(done),
                images_done=images_done,
                images_failed=report.images_failed,
                report=report.to_dict(),
                images_per_s=round(images_done / elapsed, 3),
                updated_at=now_iso(),
            )
            if ctx.after_chunk is not None:
                await ctx.after_chunk(idx)
        if failed_chunks:
            job.update(
                status='failed',
                error=f'chunks failed: {failed_chunks}',
                finished_at=now_iso(),
                poll_after_s=None,
            )
            return
        await finalize(ctx, store, report)
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker


async def finalize(ctx: ImportContext, store: ImportStore, report: ImportReport) -> None:
    """Persist the holdout freeze record and set the terminal status."""
    from src.services.curation.dataset_import.holdout_record import persist_import_freeze

    if ctx.freeze_test:
        persist_import_freeze(ctx, store)
    next_steps = []
    if report.boxes_written or report.standalone_regions:
        next_steps.append(cluster_regions())
    status = (
        'completed_with_errors' if report.images_failed or report.images_skipped else 'completed'
    )
    store.job.update(
        status=status,
        finished_at=now_iso(),
        poll_after_s=None,
        report=report.to_dict(),
        next_steps=next_steps,
    )


def check_undoable(store: ImportStore) -> None:
    """An import that finished (or was cut off) can be undone; one a live
    worker is still running cannot."""
    if undo_blocker(store.repaired_state()):
        raise ImportNotUndoableError(store.import_id)


def claim_undo(store: ImportStore) -> None:
    """Atomically move ``store`` to ``undoing``. Like a start or a resume it
    refuses while any other import of the project is live: an undo and an
    import writing the same items would race each other's decisions."""
    root = ensure_marked_dir(imports_root())
    with exclusive_start_lock(root / 'start.lock') as acquired:
        if not acquired:
            raise ImportBusyError(None)
        check_undoable(store)
        live = active_import()
        if live is not None:
            raise ImportBusyError(live.import_id)
        job = store.job
        job.clear_signals()
        job.update(status='undoing', mode='undo', error=None, finished_at=None, poll_after_s=2)
        job.touch_heartbeat()


def spawn_undo(
    ctx: UndoContext, store: ImportStore, body: Any, created_classes: dict[str, int]
) -> None:
    """Run a claimed undo (:func:`claim_undo`) as a background job:
    ``undoing`` -> ``undone``."""
    job = store.job

    async def run() -> None:
        ticker = asyncio.create_task(heartbeat_ticker(job))
        try:
            report = await undo_import(
                ctx,
                store,
                dry_run=False,
                remove_images=body.remove_images,
                deprecate_created_classes=body.deprecate_created_classes,
                created_classes=created_classes,
            )
            mark_freeze_undone(store)
            job.update(
                status='undone',
                undo=report.to_dict(),
                finished_at=now_iso(),
                poll_after_s=None,
            )
        except Exception as exc:
            logger.error('dataset_import_undo_failed', import_id=store.import_id, error=str(exc))
            job.update(
                status='failed', error=str(exc)[:300], finished_at=now_iso(), poll_after_s=None
            )
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker

    _tasks[store.import_id] = asyncio.create_task(run())


def spawn(ctx: ImportContext, store: ImportStore, entries: list[ScanEntry]) -> None:
    async def runner() -> None:
        try:
            await run_import_job(ctx, store, entries)
        except asyncio.CancelledError:
            store.job.update(status='cancelled', finished_at=now_iso(), poll_after_s=None)
            raise
        except Exception as exc:
            logger.error('dataset_import_failed', import_id=store.import_id, error=str(exc))
            store.job.update(
                status='failed', error=str(exc)[:300], finished_at=now_iso(), poll_after_s=None
            )

    _tasks[store.import_id] = asyncio.create_task(runner())


# -------------------------------------------------------------------- resume


def load_pinned(store: ImportStore) -> tuple[Any, PinnedProfile | None, str]:
    raw = store.read_mapping()
    return (
        mapping_from_dict(raw),
        PinnedProfile.from_dict(raw.get('profile')),
        raw.get('parents', 'labels'),
    )


def rescan_for_resume(
    store: ImportStore, request: DatasetImportRequest, *, path_guard: PathGuard
) -> list[ScanEntry]:
    """Re-scan the dataset and refuse to continue when it changed."""
    scan = scan_dataset(
        request.source, path_guard=path_guard, missing_label=request.options.missing_label
    )
    if source_sha(scan) != store.job.read().get('source_sha'):
        raise DatasetChangedError(store.import_id)
    return ordered_entries(scan.entries)


def check_resumable(store: ImportStore) -> None:
    state = store.repaired_state()
    if resume_blocker(state):
        raise ImportNotResumableError(store.import_id)
    live = active_import()
    if live is not None:
        raise ImportBusyError(live.import_id)


def claim_resume(store: ImportStore) -> dict[str, Any]:
    """Atomically move ``store`` to ``queued`` under the project start lock,
    the same lock a start and an undo take: two resumes, or a resume and an
    undo, cannot both pass the check. Returns the job state to hand back to
    :func:`release_resume` if the resume then fails before its worker runs."""
    root = ensure_marked_dir(imports_root())
    with exclusive_start_lock(root / 'start.lock') as acquired:
        if not acquired:
            raise ImportBusyError(None)
        check_resumable(store)
        prior = store.job.read()
        store.job.clear_signals()
        store.job.update(status='queued', error=None, finished_at=None)
        store.job.touch_heartbeat()
    return prior


def release_resume(store: ImportStore, prior: dict[str, Any]) -> None:
    """Put a claimed-but-not-started resume back to its pre-claim state."""
    store.job.update(
        status=prior.get('status'),
        error=prior.get('error'),
        finished_at=prior.get('finished_at'),
    )


def prepare_resume(store: ImportStore, registry: ClassRegistry) -> None:
    """Re-materialize classes an interrupted start created but never recorded."""
    resolved, _profile, _parents = load_pinned(store)
    materialize_created_classes(resolved, registry, adopt_existing=True)
    raw = store.read_mapping()
    raw.update(mapping_to_dict(resolved))
    store.write_mapping(raw)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cancel_import(import_id: str) -> ImportStore | None:
    store = open_store(import_id)
    if store is None:
        return None
    if store.job.read().get('status') in ACTIVE_STATUSES:
        store.job.request_cancel()
    return store


def reconcile_orphaned_jobs() -> bool:
    """Startup repair for the bound project: any import left active by a dead
    process becomes ``interrupted``, and expired uploads no import references
    are removed. True when an import state was rewritten."""
    from src.config import get_curation_config
    from src.services.curation.dataset_import.store import list_stores
    from src.services.curation.dataset_import.upload import sweep_uploads, upload_id_of

    upload_root = Path(get_curation_config().upload_root)
    repaired = False
    referenced: set[str] = set()
    for store in list_stores():
        repaired |= store.job.reconcile(
            active_statuses=ACTIVE_STATUSES, error_prefix='dataset import'
        )
        upload_id = upload_id_of(
            store.read_request().get('source', {}).get('path', ''), upload_root
        )
        if upload_id:
            referenced.add(upload_id)
    sweep_uploads(upload_root, referenced=referenced)
    return repaired


__all__ = [
    'DatasetChangedError',
    'ImportBusyError',
    'ImportNotResumableError',
    'ImportNotUndoableError',
    'ImportResumableError',
    'cancel_import',
    'check_resumable',
    'check_undoable',
    'chunked',
    'claim_import',
    'claim_undo',
    'freeze_default',
    'load_pinned',
    'ordered_entries',
    'persist_pinned',
    'persist_scan_summary',
    'prepare_resume',
    'reconcile_orphaned_jobs',
    'rescan_for_resume',
    'reusable_prior',
    'run_import_job',
    'spawn',
    'spawn_undo',
]
