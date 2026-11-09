"""Discovery for a named source: one scan pass into its queue, whichever backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.services.curation.bulk_ingest.folder_scan import scan_folder, scan_manifest
from src.services.curation.bulk_ingest.object_store import S3Config, get_object_store
from src.services.curation.bulk_ingest.s3_listing import ScanResult, scan_s3
from src.services.curation.bulk_ingest.sources import source_queue


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.services.curation.bulk_ingest.sources import IngestSource


def scan_source(
    source: IngestSource, *, should_stop: Callable[[], bool], full: bool = False
) -> ScanResult:
    """Run (or resume) one pass. Blocking. Raises the object store's classified errors and
    :class:`~src.services.curation.bulk_ingest.object_store.S3NotAllowedError` /
    ``S3NotConfiguredError`` (the allow-list is checked on every pass, so a source made
    before the configuration changed cannot read outside it).

    Two passes over one source never run side by side: a poll job is exclusive per source and a
    run is exclusive per project, and what a pass queues is idempotent anyway. ``full`` queues
    every file of a folder, not only those newer than the last pass."""
    queue = source_queue(source)
    if source.backend == 's3':
        config = S3Config.from_env()
        config.check(source.bucket, source.prefix)
        return scan_s3(
            get_object_store(),
            queue,
            bucket=source.bucket,
            prefix=source.prefix,
            extensions=source.extensions,
            workers=config.list_workers,
            should_stop=should_stop,
        )
    if source.backend == 'manifest':
        return scan_manifest(queue, Path(source.path), should_stop=should_stop)
    return scan_folder(
        queue, Path(source.path), extensions=source.extensions, should_stop=should_stop, full=full
    )
