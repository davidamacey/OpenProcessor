"""Named ingest sources: what is ingested from where, and which triggers feed it.

A source is a folder or an S3 bucket/prefix plus run options and triggers (a poll interval, a
cron schedule, a webhook), stored as a row of ``ingest_sources`` with the files it has seen in
``ingest_source_files``. The trigger and scan bookkeeping is the row's ``scan_state``.

A one-shot start of a folder, manifest or bucket/prefix (no registered source) creates a
trigger-less source of its own the first time (named by its location, ``config.one_shot``), so a
repeat start of the same location finds the same queue and continues it.

Nothing in a definition is a credential: S3 access comes from the deployment
(``object_store``), and a bucket/prefix outside ``OP_S3_ALLOWED`` is refused when the source is
created and again whenever it is used.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.config.project_context import current_project_id
from src.db.repos import ingest_sources as repo
from src.db.session import async_session_factory, project_tx
from src.services.curation.bulk_ingest.limits import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    DEFAULT_EXTENSIONS,
)
from src.services.curation.bulk_ingest.schedule import Cron, InvalidCronError
from src.services.curation.bulk_ingest.source_queue import SourceQueue


if TYPE_CHECKING:
    from src.services.curation.bulk_ingest.object_store import S3Config

MIN_WATCH_INTERVAL_S = 10
DEFAULT_WATCH_INTERVAL_S = 60
Backend = Literal['s3', 'folder', 'manifest']
SourceNameTakenError = repo.SourceNameTakenError


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
    """A source as the routes and the runner see it. ``source_id`` is the row id (empty until
    the first :func:`save_source`)."""

    name: str
    backend: Backend
    source_id: str = ''
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
    one_shot: bool = False
    created_at: str = ''
    updated_at: str = ''

    @property
    def location(self) -> str:
        return f's3://{self.bucket}/{self.prefix}' if self.backend == 's3' else self.path

    @property
    def triggered(self) -> bool:
        return self.watch.enabled or self.schedule.enabled or self.webhook_enabled

    def config(self) -> dict[str, Any]:
        return {
            'bucket': self.bucket,
            'prefix': self.prefix,
            'path': self.path,
            'extensions': list(self.extensions),
            'source_tag': self.source_tag,
            'batch_size': self.batch_size,
            'concurrency': self.concurrency,
            'webhook_enabled': self.webhook_enabled,
            'one_shot': self.one_shot,
        }

    def triggers(self) -> dict[str, Any]:
        return {
            'watch': {'enabled': self.watch.enabled, 'interval_s': self.watch.interval_s},
            'schedule': {'enabled': self.schedule.enabled, 'cron': self.schedule.cron},
        }


def from_row(row: repo.SourceRow) -> IngestSource:
    config = row.config
    watch = row.schedule.get('watch') or {}
    schedule = row.schedule.get('schedule') or {}
    return IngestSource(
        source_id=str(row.id),
        name=row.name,
        backend=row.kind,  # type: ignore[arg-type]
        bucket=config.get('bucket') or '',
        prefix=config.get('prefix') or '',
        path=config.get('path') or '',
        extensions=list(config.get('extensions') or DEFAULT_EXTENSIONS),
        source_tag=config.get('source_tag') or 'source',
        batch_size=int(config.get('batch_size') or DEFAULT_BATCH_SIZE),
        concurrency=int(config.get('concurrency') or DEFAULT_CONCURRENCY),
        watch=WatchTrigger(
            bool(watch.get('enabled')), int(watch.get('interval_s') or DEFAULT_WATCH_INTERVAL_S)
        ),
        schedule=ScheduleTrigger(bool(schedule.get('enabled')), str(schedule.get('cron') or '')),
        webhook_enabled=bool(config.get('webhook_enabled')),
        one_shot=bool(config.get('one_shot')),
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
    )


def parse_source_id(value: str) -> uuid.UUID | None:
    """The id a path names, or ``None`` for anything that is not one (such a source never exists)."""
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def one_shot_name(backend: str, location: str) -> str:
    return f'{backend}:{location}'


def validate_source_definition(source: IngestSource, s3: S3Config) -> None:
    """Raise for anything the system would refuse at use time, so a bad source never
    gets stored: :class:`InvalidSourceDefinitionError`, or for an S3 location the
    object store's ``S3NotConfiguredError`` / ``S3NotAllowedError``."""
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
        if source.backend == 'folder' and not Path(source.path).is_dir():
            raise InvalidSourceDefinitionError(f'{source.path}: not a directory')
        if source.backend == 'manifest' and not Path(source.path).is_file():
            raise InvalidSourceDefinitionError(f'{source.path}: not a file')
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


# --- storage ----------------------------------------------------------------------------------


async def list_sources() -> list[IngestSource]:
    """The bound project's registered sources (one-shot sources are not listed)."""
    project = current_project_id()
    async with async_session_factory()() as session, project_tx(session, project):
        rows = await session.run_sync(lambda sync: repo.list_for_project(sync, project))
    return [from_row(r) for r in rows if not r.config.get('one_shot')]


async def load_source(source_id: str) -> IngestSource | None:
    parsed = parse_source_id(source_id)
    if parsed is None:
        return None
    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        row = await session.run_sync(lambda sync: repo.get(sync, parsed))
    return None if row is None else from_row(row)


async def save_source(source: IngestSource) -> IngestSource:
    """Insert a new source (empty ``source_id``) or replace an existing one's definition.

    Raises :class:`SourceNameTakenError` when another source of the project has the name."""
    project = current_project_id()
    enabled = source.triggered

    def write(sync: Any) -> repo.SourceRow | None:
        if not source.source_id:
            return repo.insert(
                sync,
                project,
                name=source.name,
                kind=source.backend,
                config=source.config(),
                schedule=source.triggers(),
                enabled=enabled,
            )
        return repo.update_definition(
            sync,
            uuid.UUID(source.source_id),
            name=source.name,
            config=source.config(),
            schedule=source.triggers(),
            enabled=enabled,
        )

    async with async_session_factory()() as session, project_tx(session, project):
        row = await session.run_sync(write)
    if row is None:
        raise LookupError(f'no ingest source {source.source_id!r}')
    return from_row(row)


async def delete_source(source_id: str) -> None:
    parsed = parse_source_id(source_id)
    if parsed is None:
        return
    async with async_session_factory()() as session, project_tx(session, current_project_id()):
        await session.run_sync(lambda sync: repo.delete_source(sync, parsed))


async def one_shot_source(candidate: IngestSource) -> IngestSource:
    """The trigger-less source of ``candidate``'s location: the stored one (its run options
    updated to the request's), or a new one."""
    candidate.one_shot = True
    candidate.name = one_shot_name(candidate.backend, candidate.location)
    project = current_project_id()
    async with async_session_factory()() as session, project_tx(session, project):
        existing = await session.run_sync(
            lambda sync: repo.get_by_name(sync, project, candidate.name)
        )
    if existing is not None:
        candidate.source_id = str(existing.id)
    return await save_source(candidate)


def source_queue(source: IngestSource) -> SourceQueue:
    """The queue of a loaded source (blocking calls: use from a thread, or ``to_thread``)."""
    return SourceQueue(
        current_project_id(),
        uuid.UUID(source.source_id),
        backend=source.backend,
        bucket=source.bucket,
    )
