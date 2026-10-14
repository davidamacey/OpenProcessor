"""Named ingest sources: what is ingested from where, and which triggers feed it.

A source is a folder or an S3 bucket/prefix plus run options and triggers (a poll
interval, a cron schedule, a webhook). Definitions and runtime state live beside the
bulk-run directories of the bound project::

    <runs_root>/sources/<source_id>/source.json   the definition
                                    state.json    trigger bookkeeping (a FileJob)
                                    queue.sqlite  the durable queue (object_queue)

Nothing in a definition is a credential: S3 access comes from the deployment
(``object_store``), and a bucket/prefix outside ``OP_S3_ALLOWED`` is refused when the
source is created and again whenever it is used.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.config.project_context import ensure_marked_dir
from src.services.curation.bulk_ingest.limits import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    DEFAULT_EXTENSIONS,
)
from src.services.curation.bulk_ingest.object_queue import ObjectQueue
from src.services.curation.bulk_ingest.schedule import Cron, InvalidCronError
from src.services.curation.bulk_ingest.store import runs_root
from src.services.curation.file_job import FileJob
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from src.services.curation.bulk_ingest.object_store import S3Config

SOURCE_ID_RE = re.compile(r'^[a-z0-9][a-z0-9_]{2,47}$')
MIN_WATCH_INTERVAL_S = 10
DEFAULT_WATCH_INTERVAL_S = 60
Backend = Literal['s3', 'folder']


class InvalidSourceDefinitionError(ValueError):
    """The definition cannot be used; the message says why."""


@dataclass
class WatchTrigger:
    enabled: bool = False
    interval_s: int = DEFAULT_WATCH_INTERVAL_S


@dataclass
class ScheduleTrigger:
    enabled: bool = False
    cron: str = ''


@dataclass
class IngestSource:
    source_id: str
    name: str
    backend: Backend
    bucket: str = ''
    prefix: str = ''
    path: str = ''
    extensions: list[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    source_tag: str = 'source'
    batch_size: int = DEFAULT_BATCH_SIZE
    concurrency: int = DEFAULT_CONCURRENCY
    watch: WatchTrigger = field(default_factory=WatchTrigger)
    schedule: ScheduleTrigger = field(default_factory=ScheduleTrigger)
    webhook_enabled: bool = False
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> IngestSource:
        data = dict(raw)
        data['watch'] = WatchTrigger(**(data.get('watch') or {}))
        data['schedule'] = ScheduleTrigger(**(data.get('schedule') or {}))
        return cls(**data)

    @property
    def location(self) -> str:
        return f's3://{self.bucket}/{self.prefix}' if self.backend == 's3' else self.path


def s3_source_id(bucket: str, prefix: str) -> str:
    """The id of the trigger-less source a one-shot bucket ingest uses."""
    return 's3_' + hashlib.sha256(f'{bucket}/{prefix}'.encode()).hexdigest()[:12]


def new_source_id() -> str:
    return 'src_' + uuid.uuid4().hex[:10]


def validate_source_definition(source: IngestSource, s3: S3Config) -> None:
    """Raise for anything the system would refuse at use time, so a bad source never
    gets stored: :class:`InvalidSourceDefinitionError`, or for an S3 location the
    object store's ``S3NotConfiguredError`` / ``S3NotAllowedError``."""
    if not SOURCE_ID_RE.fullmatch(source.source_id):
        raise InvalidSourceDefinitionError(f'bad source id {source.source_id!r}')
    if source.backend == 's3':
        if not source.bucket:
            raise InvalidSourceDefinitionError('an S3 source needs a bucket')
        s3.check(source.bucket, source.prefix)
    else:
        from src.services.curation.image_serving import (
            UNSERVABLE_PATH_ERROR,
            is_servable_image_path,
        )

        if not source.path or not is_servable_image_path(source.path):
            raise InvalidSourceDefinitionError(f'{source.path}: {UNSERVABLE_PATH_ERROR}')
        if not Path(source.path).is_dir():
            raise InvalidSourceDefinitionError(f'{source.path}: not a directory')
    if source.watch.enabled and source.watch.interval_s < MIN_WATCH_INTERVAL_S:
        raise InvalidSourceDefinitionError(
            f'watch.interval_s must be at least {MIN_WATCH_INTERVAL_S}'
        )
    if source.schedule.enabled:
        try:
            Cron.parse(source.schedule.cron)
        except InvalidCronError as exc:
            raise InvalidSourceDefinitionError(f'schedule.cron: {exc}') from None
    if source.webhook_enabled and source.backend != 's3':
        raise InvalidSourceDefinitionError('only an S3 source can receive webhook events')


def sources_root() -> Path:
    return runs_root() / 'sources'


def source_dir(source_id: str) -> Path:
    if not SOURCE_ID_RE.fullmatch(source_id):
        raise InvalidSourceDefinitionError(f'bad source id {source_id!r}')
    return sources_root() / source_id


def list_sources() -> list[IngestSource]:
    root = sources_root()
    if not root.is_dir():
        return []
    found = [load_source(p.name) for p in sorted(root.iterdir()) if p.is_dir()]
    return [s for s in found if s is not None]


def load_source(source_id: str) -> IngestSource | None:
    try:
        raw = json.loads((source_dir(source_id) / 'source.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, InvalidSourceDefinitionError):
        return None
    try:
        return IngestSource.from_dict(raw)
    except TypeError:
        return None


def save_source(source: IngestSource) -> None:
    directory = source_dir(source.source_id)
    ensure_marked_dir(sources_root())
    directory.mkdir(parents=True, exist_ok=True)
    source.updated_at = now_iso()
    tmp = directory / 'source.json.tmp'
    tmp.write_text(json.dumps(source.to_dict()), encoding='utf-8')
    tmp.replace(directory / 'source.json')


def delete_source(source_id: str) -> None:
    shutil.rmtree(source_dir(source_id), ignore_errors=True)


def source_queue(source_id: str) -> ObjectQueue:
    return ObjectQueue(source_dir(source_id) / 'queue.sqlite')


def source_state(source_id: str) -> FileJob:
    """Trigger bookkeeping: ``last_poll_started_at`` (epoch s), ``last_poll_error``,
    ``failures`` and ``backoff_until`` (poll backoff), ``last_fire_at`` (ISO, schedule),
    ``last_run_id``, ``run_requested``."""
    return FileJob(source_dir(source_id))
