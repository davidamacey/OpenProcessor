"""The import loop: seed the ledger, claim pending images in chunks, import them, report.

The loop runs inside the ``dataset_import`` job (:mod:`.job`). Its position is the ledger: a
chunk is the next pending images in scan order, so a run that was interrupted, failed or
cancelled continues with what is still ``pending`` (an image the dead run had begun carries its
write-ahead row, which keeps ``created`` vs ``updated`` truthful across the re-run).

A resume consumes what the START persisted: the resolved mapping with its real class ids and the
pinned region profile. It never re-resolves them against the registry or config store of the
process that resumes.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from src.core.logging import get_logger
from src.services.curation.dataset_import import limits
from src.services.curation.dataset_import.chunk import import_chunk
from src.services.curation.dataset_import.prepare import (
    PinnedProfile,
    PreparedImport,
    mapping_from_dict,
    mapping_to_dict,
)
from src.services.curation.dataset_import.scan import ScanEntry, scan_dataset, source_sha
from src.services.curation.dataset_import.status import BACKPRESSURE_STAGE


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.options import DatasetImportRequest
    from src.services.curation.dataset_import.paths import PathGuard
    from src.services.curation.dataset_import.report import ImportReport

logger = get_logger(__name__)

_SPLIT_ORDER = {'train': 0, 'val': 1, 'test': 2}


class ImportBusyError(Exception):
    def __init__(self, import_id: str | None) -> None:
        super().__init__(f'an import is already running: {import_id}')
        self.import_id = import_id


class ImportNotResumableError(Exception):
    pass


class ImportNotUndoableError(Exception):
    pass


class DatasetChangedError(Exception):
    pass


class RunControl(Protocol):
    """The slice of the job context the loop uses."""

    def cancelled(self) -> bool: ...

    async def progress(
        self,
        done: int,
        total: int | None = None,
        stage: str | None = None,
        counters: Mapping[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> None: ...


@dataclass
class ImportOutcome:
    report: ImportReport
    cancelled: bool = False
    error: str | None = None
    failed_chunks: list[int] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)


def ordered_entries(entries: list[ScanEntry]) -> list[ScanEntry]:
    """Chunk order: split, then relative path. Stable across restarts."""
    return sorted(entries, key=lambda e: (_SPLIT_ORDER.get(e.split or '', 3), e.rel_path))


def chunks_total(images: int) -> int:
    return math.ceil(images / limits.import_chunk_size())


def freeze_default(prepared: PreparedImport, request: DatasetImportRequest) -> bool:
    explicit = request.options.freeze_test_split
    if explicit is not None:
        return explicit
    op = prepared.scan.op_export
    return bool(getattr(op, 'freeze_test_split_default', False))


def scan_document(
    prepared: PreparedImport,
    request: DatasetImportRequest,
    issues: list[dict[str, Any]],
    *,
    freeze_test: bool,
) -> dict[str, Any]:
    """The scan summary and the run facts a worker reads back (``dataset_imports.scan``)."""
    entries = prepared.scan.entries
    document: dict[str, Any] = {
        'images_total': len(entries),
        'source_sha': prepared.source_sha,
        'boxes': sum(len(e.boxes) for e in entries),
        'issues': issues,
        'import_key': prepared.import_key,
        'reuse_key': prepared.reuse_key,
        'name': request.options.name,
        'source_format': prepared.scan.format,
        'source_root': str(prepared.scan.root),
        'freeze_test': freeze_test,
    }
    op = prepared.scan.op_export
    if op is not None:
        document['test_split'] = {
            'test_label_sha': op.test_frozen.test_label_sha,
            'source_frozen_test_sha': op.frozen_test_sha,
        }
    return document


def pinned_mapping(prepared: PreparedImport) -> dict[str, Any]:
    return {
        **mapping_to_dict(prepared.resolved),
        'profile': prepared.view.profile.to_dict() if prepared.view.profile else None,
        'parents': prepared.parents,
    }


def load_pinned(raw: dict[str, Any]) -> tuple[Any, PinnedProfile | None, str]:
    return (
        mapping_from_dict(raw),
        PinnedProfile.from_dict(raw.get('profile')),
        raw.get('parents', 'labels'),
    )


def rescan(
    request: DatasetImportRequest, expected_sha: str | None, *, path_guard: PathGuard
) -> list[ScanEntry]:
    """Scan the dataset again and refuse to go on when it changed since the start."""
    scan = scan_dataset(
        request.source, path_guard=path_guard, missing_label=request.options.missing_label
    )
    if source_sha(scan) != expected_sha:
        raise DatasetChangedError(expected_sha)
    return ordered_entries(scan.entries)


async def _backpressure(ctx: ImportContext, job: RunControl, done: int, total: int) -> None:
    """Pause while the region worker is behind (propose / detected parents)."""
    if not ctx.uses_detector or ctx.profile is None:
        return
    from src.config.region_state import RegionStatus
    from src.services.curation.region_drain import region_drain_poll_interval_s

    fields = ctx.region_fields
    waiting = False
    while not job.cancelled():
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
        waiting = True
        await job.progress(done, total, BACKPRESSURE_STAGE, force=True)
        await asyncio.sleep(region_drain_poll_interval_s())
    if waiting:
        await job.progress(done, total, 'running', force=True)


def _counters(report: ImportReport, *, chunks: int, total: int, rate: float) -> dict[str, Any]:
    return {
        'images_done': report.images_created + report.images_reused,
        'images_failed': report.images_failed,
        'images_skipped': report.images_skipped,
        'chunks_done': chunks,
        'chunks_total': chunks_total(total),
        'images_per_s': rate,
    }


async def run_import_job(
    job: RunControl, ctx: ImportContext, entries: list[ScanEntry]
) -> ImportOutcome:
    """Import every image still ``pending`` in the ledger, a chunk at a time."""
    ledger = ctx.ledger
    by_key = {e.rel_path: e for e in entries}
    await ledger.seed(entries)
    report, done = await ledger.finished_report()
    total = len(entries)
    size = limits.import_chunk_size()
    outcome = ImportOutcome(report)
    chunks_done = done // size
    consecutive_failures = 0
    last = -1
    started = asyncio.get_running_loop().time()
    start_done = done
    await job.progress(
        done,
        total,
        'running',
        _counters(report, chunks=chunks_done, total=total, rate=0.0),
        force=True,
    )
    while True:
        if job.cancelled():
            outcome.cancelled = True
            return outcome
        batch = await ledger.claim(size, after=last)
        if not batch:
            return outcome
        last = batch[-1].order
        await _backpressure(ctx, job, done, total)
        if job.cancelled():
            outcome.cancelled = True
            return outcome
        number = chunks_done + len(outcome.failed_chunks)
        try:
            chunk_report = await import_chunk(
                ctx, [by_key[c.key] for c in batch], {c.key: c.prior for c in batch}
            )
        except Exception as exc:
            logger.error('dataset_import_chunk_failed', chunk=number, error=str(exc))
            outcome.issues.append(
                {'code': 'chunk_failed', 'chunk': number, 'error': str(exc)[:300]}
            )
            outcome.failed_chunks.append(number)
            await ledger.save_report(report, issues=outcome.issues)
            consecutive_failures += 1
            if consecutive_failures > limits.import_max_failed_chunks():
                outcome.error = f'{consecutive_failures} consecutive chunks failed: {exc}'[:300]
                return outcome
            continue
        consecutive_failures = 0
        report.add(chunk_report)
        done += len(batch)
        chunks_done += 1
        elapsed = max(asyncio.get_running_loop().time() - started, 1e-6)
        rate = round((done - start_done) / elapsed, 3)
        await ledger.save_report(report, issues=outcome.issues)
        await job.progress(
            done,
            total,
            'running',
            _counters(report, chunks=chunks_done, total=total, rate=rate),
        )
        if ctx.after_chunk is not None:
            await ctx.after_chunk(number)


__all__ = [
    'DatasetChangedError',
    'ImportBusyError',
    'ImportNotResumableError',
    'ImportNotUndoableError',
    'ImportOutcome',
    'RunControl',
    'chunks_total',
    'freeze_default',
    'load_pinned',
    'ordered_entries',
    'pinned_mapping',
    'rescan',
    'run_import_job',
    'scan_document',
]
