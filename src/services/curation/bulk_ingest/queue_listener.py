"""Settle queue rows from the outcome of a run's batches."""

from __future__ import annotations

from src.services.curation.bulk_ingest.object_store import ObjectRef
from src.services.curation.bulk_ingest.sources import IngestSource, source_queue


class QueueListener:
    """A :class:`~src.services.curation.bulk_ingest.runner.RunListener` that marks the
    rows of one source's queue done or failed as its run settles them."""

    def __init__(self, source: IngestSource) -> None:
        self._queue = source_queue(source.source_id)
        self._s3 = source.backend == 's3'

    def _identity(self, ref: str) -> tuple[str, str | None]:
        if not self._s3:
            return ref, None
        parsed = ObjectRef.parse(ref)
        return parsed.key, parsed.etag

    def on_done(self, refs: list[str]) -> None:
        self._queue.mark_done([self._identity(r) for r in refs])

    def on_failed(self, rows: list[tuple[str, str, str]]) -> None:
        self._queue.mark_failed([(*self._identity(ref), error, kind) for ref, error, kind in rows])
