"""Discovery for a named source: one scan pass into its queue, whichever backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.services.curation.bulk_ingest.folder_scan import scan_folder
from src.services.curation.bulk_ingest.object_store import S3Config, get_object_store
from src.services.curation.bulk_ingest.s3_listing import ScanResult, scan_s3
from src.services.curation.bulk_ingest.sources import source_dir, source_queue
from src.services.curation.job_lock import exclusive_start_lock


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.services.curation.bulk_ingest.sources import IngestSource


def scan_source(source: IngestSource, *, should_stop: Callable[[], bool]) -> ScanResult | None:
    """Run (or resume) one pass; ``None`` when another worker is already scanning this
    source. Blocking. Raises the object store's classified errors and
    :class:`~src.services.curation.bulk_ingest.object_store.S3NotAllowedError` /
    ``S3NotConfiguredError`` (the allow-list is checked on every pass, so a source made
    before the configuration changed cannot read outside it)."""
    queue = source_queue(source.source_id)
    with exclusive_start_lock(source_dir(source.source_id) / 'scan.lock') as acquired:
        if not acquired:
            return None
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
        return scan_folder(
            queue, Path(source.path), extensions=source.extensions, should_stop=should_stop
        )
