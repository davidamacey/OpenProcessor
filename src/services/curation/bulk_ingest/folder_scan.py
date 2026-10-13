"""Scan a local folder into a source's queue (the folder watcher's discovery).

The same contract as the S3 pass: new or changed files only, a watermark on the
modification time that moves forward after a completed pass, rows idempotent by
``(path, "<mtime_ns>:<size>")``. A file modified in the last :data:`SETTLE_S`
seconds is left for the next pass, so a copy still in progress is not ingested half
written.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.services.curation.bulk_ingest.object_queue import QueueEntry
from src.services.curation.bulk_ingest.s3_listing import (
    LAST_SCAN_META,
    WATERMARK_META,
    WATERMARK_OVERLAP_S,
    ScanResult,
)
from src.services.curation.bulk_ingest.walk import iter_image_paths


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from src.services.curation.bulk_ingest.object_queue import ObjectQueue

SETTLE_S = 5.0
_CHUNK = 500


def scan_folder(
    queue: ObjectQueue,
    root: Path,
    *,
    extensions: Iterable[str],
    workers: int = 8,
    should_stop: Callable[[], bool] = lambda: False,
    now: float | None = None,
) -> ScanResult:
    started = now if now is not None else time.time()
    watermark: float | None = queue.meta_get(WATERMARK_META)
    result = ScanResult()
    chunk: list[QueueEntry] = []

    def flush() -> None:
        queued, duplicates = queue.enqueue(chunk)
        result.queued += queued
        result.duplicates += duplicates
        chunk.clear()

    for path in iter_image_paths(root, frozenset(e.lower() for e in extensions), workers):
        if should_stop():
            break
        try:
            stat = path.stat()
        except OSError:
            continue
        result.listed += 1
        if stat.st_mtime > started - SETTLE_S:
            continue
        if watermark is not None and stat.st_mtime < watermark:
            continue
        chunk.append(
            QueueEntry(str(path), f'{stat.st_mtime_ns}:{stat.st_size}', None, stat.st_size)
        )
        if len(chunk) >= _CHUNK:
            flush()
    flush()
    if should_stop():
        return result
    # A file deferred as unsettled is newer than this mark and so is seen next pass.
    queue.meta_set(WATERMARK_META, started - WATERMARK_OVERLAP_S)
    queue.meta_set(
        LAST_SCAN_META,
        {'finished_at': time.time(), 'pages': 0, 'listed': result.listed, 'queued': result.queued},
    )
    return result


__all__ = ['scan_folder']
