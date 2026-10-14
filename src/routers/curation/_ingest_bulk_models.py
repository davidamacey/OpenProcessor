"""Wire models for ``/ingest/bulk*`` (bulk ingest of a folder or manifest)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.routers.curation._ingest_sources_models import (
    S3Location,  # noqa: TC001 - pydantic field type
)
from src.services.curation.bulk_ingest.limits import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    MAX_BATCH_SIZE,
    MAX_CONCURRENCY,
)


BulkIngestStatus = Literal[
    'queued', 'running', 'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted'
]
BulkSourceKind = Literal['folder', 'manifest', 'queue']


class BulkIngestRequest(BaseModel):
    """Exactly one of ``folder`` and ``manifest`` (a path inside the API container,
    under a configured source root), ``s3`` (a bucket and prefix inside
    ``OP_S3_ALLOWED``) or ``source_id`` (a registered ingest source, whose own options
    apply). Starting a source whose earlier run was interrupted, cancelled or finished
    with failures resumes that run."""

    model_config = ConfigDict(extra='forbid')

    folder: str | None = Field(default=None, min_length=1)
    manifest: str | None = Field(default=None, min_length=1)
    s3: S3Location | None = None
    source_id: str | None = Field(default=None, min_length=1, max_length=64)
    source: str = Field(default='bulk_ingest', min_length=1, max_length=120)
    extensions: list[str] | None = Field(default=None, max_length=32)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=MAX_BATCH_SIZE)
    concurrency: int = Field(default=DEFAULT_CONCURRENCY, ge=1, le=MAX_CONCURRENCY)

    @model_validator(mode='after')
    def _exactly_one_source(self) -> Self:
        chosen = [self.folder, self.manifest, self.s3, self.source_id]
        if sum(x is not None for x in chosen) != 1:
            raise ValueError('send exactly one of folder, manifest, s3 and source_id')
        return self


class BulkSourceWire(BaseModel):
    kind: BulkSourceKind
    # A folder or manifest path; for a queue source, the id of the registered source.
    path: str
    extensions: list[str]
    # What to show for the source: its name and location when it is a registered one.
    label: str | None = None


class BulkOptionsWire(BaseModel):
    source_tag: str
    batch_size: int
    concurrency: int


class BulkIngestProgress(BaseModel):
    # Paths seen so far; final once ``discovery_complete`` (a folder is walked while it runs).
    discovered: int
    discovery_complete: bool
    # Already stored by an earlier attempt of this run (skipped without a read).
    skipped: int
    # Cumulative over the run's attempts (as are crops_indexed and the n_* vector counts).
    ingested: int
    duplicates: int
    # This attempt's failures (a resume retries them); see GET .../failures.
    failed: int
    # skipped + stored this attempt + failed: what ``discovered`` is measured against.
    processed: int
    crops_indexed: int
    n_embedded: int
    n_not_embedded: int
    n_embed_failed: int
    in_flight: int
    images_per_s: float | None = None
    # Over the last 30 s; null until two samples exist.
    images_per_s_recent: float | None = None
    eta_s: float | None = None
    elapsed_s: float | None = None


class BulkIngestRun(BaseModel):
    project: str
    run_id: str
    status: BulkIngestStatus
    # True on the 202 that continued an earlier run of the same source.
    resumed: bool = False
    source: BulkSourceWire
    options: BulkOptionsWire
    progress: BulkIngestProgress
    # Wall seconds per stage. ``read`` and ``ingest_batch`` are this run's own
    # spans (summed over concurrent batches, so they can exceed elapsed time); the
    # rest are the pipeline's stage timers over the run (the whole process, so
    # other ingest traffic in the same API worker is included).
    stage_s: dict[str, float] = Field(default_factory=dict)
    attempts: int = 1
    # What the run allows now: cancel while it is live; resume (start the same
    # source again) once it stopped with work left.
    can_cancel: bool
    can_resume: bool
    # A cancel was asked for and the run has not stopped yet.
    cancel_requested: bool = False
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    # Seconds a poller should wait before the next read; null once the run is over.
    poll_after_s: float | None = None


class BulkIngestList(BaseModel):
    items: list[BulkIngestRun]
    total: int
    page: int
    page_size: int


class BulkIngestFailure(BaseModel):
    path: str
    error: str
    error_kind: str


class BulkIngestFailurePage(BaseModel):
    items: list[BulkIngestFailure]
    total: int
    offset: int
    limit: int


class BulkIngestConfig(BaseModel):
    source_roots: list[str]
    default_extensions: list[str]
    default_batch_size: int
    default_concurrency: int
    max_batch_size: int
    max_concurrency: int
