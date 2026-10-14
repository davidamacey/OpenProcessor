"""Wire models for ``/ingest/sources*``: registered ingest sources, their triggers and status."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.services.curation.bulk_ingest.limits import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    MAX_BATCH_SIZE,
    MAX_CONCURRENCY,
)
from src.services.curation.bulk_ingest.sources import DEFAULT_WATCH_INTERVAL_S


SourceBackend = Literal['s3', 'folder']
BlockedReason = Literal['s3_not_configured', 's3_not_allowed', 'invalid_source', 'project_run_live']


class S3Location(BaseModel):
    model_config = ConfigDict(extra='forbid')

    bucket: str = Field(min_length=1, max_length=255)
    prefix: str = Field(default='', max_length=1024)


class WatchTriggerWire(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    interval_s: int = Field(default=DEFAULT_WATCH_INTERVAL_S, ge=1, le=7 * 86400)


class ScheduleTriggerWire(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    # Five fields, UTC: minute hour day-of-month month day-of-week.
    cron: str = Field(default='', max_length=120)


class TriggersWire(BaseModel):
    model_config = ConfigDict(extra='forbid')

    watch: WatchTriggerWire = Field(default_factory=WatchTriggerWire)
    schedule: ScheduleTriggerWire = Field(default_factory=ScheduleTriggerWire)
    webhook_enabled: bool = False


class IngestSourceRequest(BaseModel):
    """Exactly one of ``s3`` and ``folder``, matching ``backend``."""

    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=1, max_length=120)
    backend: SourceBackend
    s3: S3Location | None = None
    folder: str | None = Field(default=None, min_length=1)
    extensions: list[str] | None = Field(default=None, max_length=32)
    source: str = Field(default='source', min_length=1, max_length=120)
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=MAX_BATCH_SIZE)
    concurrency: int = Field(default=DEFAULT_CONCURRENCY, ge=1, le=MAX_CONCURRENCY)
    triggers: TriggersWire = Field(default_factory=TriggersWire)

    @model_validator(mode='after')
    def _location_matches_backend(self) -> Self:
        if self.backend == 's3' and (self.s3 is None or self.folder is not None):
            raise ValueError('an s3 source takes s3 {bucket, prefix} and no folder')
        if self.backend == 'folder' and (self.folder is None or self.s3 is not None):
            raise ValueError('a folder source takes folder and no s3')
        return self


class IngestSourcePatch(BaseModel):
    """Changes to the options and triggers; the location of a source never changes."""

    model_config = ConfigDict(extra='forbid')

    name: str | None = Field(default=None, min_length=1, max_length=120)
    extensions: list[str] | None = Field(default=None, max_length=32)
    source: str | None = Field(default=None, min_length=1, max_length=120)
    batch_size: int | None = Field(default=None, ge=1, le=MAX_BATCH_SIZE)
    concurrency: int | None = Field(default=None, ge=1, le=MAX_CONCURRENCY)
    triggers: TriggersWire | None = None


class SourceCounts(BaseModel):
    pending: int
    done: int
    failed: int
    dead: int


class LastScan(BaseModel):
    finished_at: float
    pages: int
    listed: int
    queued: int


class SourceScanStatus(BaseModel):
    # A pass is under way (or was interrupted and will resume).
    active: bool
    pages: int
    listed: int
    queued: int
    last: LastScan | None = None


class SourceWatchStatus(BaseModel):
    last_poll_at: str | None = None
    next_poll_at: str | None = None
    error: str | None = None
    backoff_until: str | None = None


class SourceScheduleStatus(BaseModel):
    last_fire_at: str | None = None
    next_fire_at: str | None = None


class SourceWebhookStatus(BaseModel):
    # Path of the events route, relative to the API origin.
    path: str
    received: int
    queued: int
    duplicates: int
    ignored: int
    last_event_at: str | None = None


class SourceRunSummary(BaseModel):
    run_id: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    ingested: int
    duplicates: int
    failed: int
    error: str | None = None


class SourceObjectError(BaseModel):
    key: str
    etag: str
    state: Literal['failed', 'dead']
    attempts: int
    error: str


class IngestSourceStatus(BaseModel):
    project: str
    source_id: str
    name: str
    backend: SourceBackend
    # ``s3://bucket/prefix`` or the folder path.
    location: str
    s3: S3Location | None = None
    folder: str | None = None
    extensions: list[str]
    source: str
    batch_size: int
    concurrency: int
    triggers: TriggersWire
    created_at: str
    counts: SourceCounts
    scan: SourceScanStatus
    watch: SourceWatchStatus
    schedule: SourceScheduleStatus
    webhook: SourceWebhookStatus
    last_run: SourceRunSummary | None = None
    errors: list[SourceObjectError] = Field(default_factory=list)
    can_run: bool
    can_retry_dead: bool
    blocked_reason: BlockedReason | None = None
    # The reason in words, for display.
    blocked_message: str | None = None


class IngestSourceList(BaseModel):
    items: list[IngestSourceStatus]
    # Whether S3 sources can be created at all (OP_S3_ALLOWED set), and the entries.
    s3_configured: bool
    s3_allowed: list[str]
    default_batch_size: int
    default_concurrency: int
    default_watch_interval_s: int
    min_watch_interval_s: int
    # Seconds a poller should wait before the next read: short while a scan or run is live.
    poll_after_s: float


class RetryDeadResult(BaseModel):
    retried: int
    source: IngestSourceStatus


class EventsAccepted(BaseModel):
    """What one notification did. ``received`` = ``queued`` + ``duplicates`` + ``ignored``."""

    received: int
    queued: int
    duplicates: int
    ignored: int
