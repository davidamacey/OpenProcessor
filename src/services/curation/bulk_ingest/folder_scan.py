"""Scan a local folder into a source's queue (the folder watcher's discovery).

The same contract as the S3 pass: new or changed files only, a watermark on the
modification time that moves forward after a completed pass, rows idempotent by
``(path, "<mtime_ns>:<size>")``. A file modified in the last :data:`SETTLE_S`
seconds is left for the next pass, so a copy still in progress is not ingested half
written.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.services.curation.bulk_ingest.s3_listing import (
    LAST_SCAN_META,
    WATERMARK_META,
    WATERMARK_OVERLAP_S,
    ScanResult,
)
from src.services.curation.bulk_ingest.source import iter_manifest_paths
from src.services.curation.bulk_ingest.source_queue import QueueEntry
from src.services.curation.bulk_ingest.walk import WalkStats, iter_image_paths


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from src.services.curation.bulk_ingest.source_queue import SourceQueue

SETTLE_S = 5.0
UNREADABLE_SAMPLE_LIMIT = 10
_CHUNK = 500


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def scan_folder(
    queue: SourceQueue,
    root: Path,
    *,
    extensions: Iterable[str],
    workers: int = 8,
    should_stop: Callable[[], bool] = lambda: False,
    now: float | None = None,
    full: bool = False,
) -> ScanResult:
    """One pass. ``full`` ignores the watermark and the settle delay and queues every file: a
    one-shot start names a folder a person has finished writing, and has no next pass."""
    started = now if now is not None else time.time()
    watermark: float | None = None if full else queue.meta_get(WATERMARK_META)
    result = ScanResult()
    chunk: list[QueueEntry] = []

    def flush() -> None:
        queued, duplicates = queue.enqueue(chunk)
        result.queued += queued
        result.duplicates += duplicates
        chunk.clear()

    stats = WalkStats()
    unreadable_files: list[str] = []
    for path in iter_image_paths(
        root, frozenset(e.lower() for e in extensions), workers, stats=stats
    ):
        if should_stop():
            break
        try:
            stat = path.stat()
        except OSError:
            unreadable_files.append(_relative(path, root))
            continue
        result.listed += 1
        if not full and stat.st_mtime > started - SETTLE_S:
            continue
        if watermark is not None and stat.st_mtime < watermark:
            continue
        chunk.append(
            QueueEntry(str(path), f'{stat.st_mtime_ns}:{stat.st_size}', None, stat.st_size)
        )
        if len(chunk) >= _CHUNK:
            flush()
    flush()
    result.unreadable = stats.unreadable_dirs + stats.symlinks + len(unreadable_files)
    result.unreadable_samples = [
        _relative(Path(p), root) for p in (*stats.unreadable_samples, *stats.symlink_samples)
    ][:UNREADABLE_SAMPLE_LIMIT]
    if len(result.unreadable_samples) < UNREADABLE_SAMPLE_LIMIT:
        result.unreadable_samples += unreadable_files[
            : UNREADABLE_SAMPLE_LIMIT - len(result.unreadable_samples)
        ]
    if should_stop():
        return result
    # A file deferred as unsettled is newer than this mark and so is seen next pass.
    queue.meta_set(WATERMARK_META, started - WATERMARK_OVERLAP_S)
    queue.meta_set(
        LAST_SCAN_META,
        {
            'finished_at': time.time(),
            'pages': 0,
            'listed': result.listed,
            'queued': result.queued,
            'unreadable': result.unreadable,
        },
    )
    return result


def scan_manifest(
    queue: SourceQueue, path: Path, *, should_stop: Callable[[], bool] = lambda: False
) -> ScanResult:
    """Queue every entry of a manifest (``source.py``); known entries queue nothing."""
    result = ScanResult()
    chunk: list[str] = []

    def flush() -> None:
        queued, duplicates = queue.enqueue_keys(chunk)
        result.queued += queued
        result.duplicates += duplicates
        chunk.clear()

    for entry in iter_manifest_paths(path):
        if should_stop():
            break
        result.listed += 1
        chunk.append(entry)
        if len(chunk) >= _CHUNK:
            flush()
    flush()
    return result


__all__ = ['scan_folder', 'scan_manifest']
