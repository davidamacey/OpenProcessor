"""``/ingest/sources*``: registered ingest sources, their triggers and the run controls.

A thin HTTP adapter over :mod:`src.services.curation.bulk_ingest`. A source is a folder
or an S3 bucket/prefix with options and triggers (watch, schedule, webhook); every
trigger feeds the source's durable queue and the one bulk runner drains it. The
dispatcher (``bulk_ingest/dispatcher.py``) acts on due triggers; the routes here manage
definitions, start a run now, requeue dead objects and receive S3 event notifications.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from fastapi import Request  # noqa: TC002 - FastAPI resolves route annotations at runtime

from src.config import get_curation_config
from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves route annotations at runtime
from src.core.disk_space import DISK_GUARD
from src.routers.curation._api_error import api_error
from src.routers.curation._common import OpenSearchDep, RegistryDep, router
from src.routers.curation._ingest_bulk_models import BulkIngestRequest, BulkIngestRun
from src.routers.curation._ingest_bulk_wire import run_wire, slug
from src.routers.curation._ingest_sources_models import (
    EventsAccepted,
    IngestSourceList,
    IngestSourcePatch,
    IngestSourceRequest,
    IngestSourceStatus,
    RetryDeadResult,
    S3Location,
    ScheduleTriggerWire,
    SourceCounts,
    SourceObjectError,
    SourceRunSummary,
    SourceScanStatus,
    SourceScheduleStatus,
    SourceWatchStatus,
    SourceWebhookStatus,
    TriggersWire,
    WatchTriggerWire,
)
from src.services.curation.bulk_ingest import limits
from src.services.curation.bulk_ingest.object_store import (
    S3Config,
    S3NotAllowedError,
    S3NotConfiguredError,
)
from src.services.curation.bulk_ingest.runs import get_run, live_run
from src.services.curation.bulk_ingest.s3_events import (
    EventRecord,
    InvalidS3EventError,
    parse_s3_event,
)
from src.services.curation.bulk_ingest.s3_listing import wanted_extension
from src.services.curation.bulk_ingest.source import normalise_extensions
from src.services.curation.bulk_ingest.source_queue import QueueEntry
from src.services.curation.bulk_ingest.source_runs import (
    RESUMABLE_RUN_STATUSES,
    BulkBusyError,
    last_run,
    run_is_live,
    source_status,
    start_source_run,
)
from src.services.curation.bulk_ingest.sources import (
    DEFAULT_WATCH_INTERVAL_S,
    MIN_WATCH_INTERVAL_S,
    IngestSource,
    InvalidSourceDefinitionError,
    ScheduleTrigger,
    SourceNameTakenError,
    WatchTrigger,
    delete_source,
    list_sources,
    load_source,
    one_shot_source,
    save_source,
    source_queue,
    validate_source_definition,
)
from src.services.curation.ingest_service_factory import get_ingest_service


if TYPE_CHECKING:
    import uuid
    from typing import Any

    from fastapi import HTTPException
    from sqlalchemy.ext.asyncio import AsyncSession


LIVE_POLL_S = 2.0
IDLE_POLL_S = 10.0


def _not_found(source_id: str) -> HTTPException:
    return api_error(
        404, 'ingest_source_not_found', f'no ingest source {source_id!r}', project=slug()
    )


async def _load(source_id: str) -> IngestSource:
    source = await load_source(source_id)
    if source is None or source.one_shot:
        raise _not_found(source_id)
    return source


def check_definition(source: IngestSource) -> None:
    """Refuse a definition the system would refuse at use time, with the stable code."""
    try:
        validate_source_definition(source, S3Config.from_env())
    except S3NotConfiguredError as exc:
        raise api_error(503, 's3_not_configured', str(exc), project=slug()) from None
    except S3NotAllowedError as exc:
        raise api_error(422, 's3_not_allowed', str(exc), project=slug()) from None
    except InvalidSourceDefinitionError as exc:
        raise api_error(422, 'ingest_source_invalid', str(exc), project=slug()) from None


_BLOCKED_MESSAGES = {
    's3_not_configured': 'S3 is off on this deployment: OP_S3_ALLOWED is empty',
    's3_not_allowed': 'This bucket or prefix is outside OP_S3_ALLOWED',
    'invalid_source': 'The source can no longer be read: check its path',
    'project_run_live': 'Another ingest run of this project is live',
}


def _blocked_reason(source: IngestSource, *, project_run_live: bool) -> str | None:
    try:
        validate_source_definition(source, S3Config.from_env())
    except S3NotConfiguredError:
        return 's3_not_configured'
    except S3NotAllowedError:
        return 's3_not_allowed'
    except InvalidSourceDefinitionError:
        return 'invalid_source'
    return 'project_run_live' if project_run_live else None


async def status_wire(
    session: AsyncSession, project: uuid.UUID, source: IngestSource, *, project_run_live: bool
) -> IngestSourceStatus:
    data: dict[str, Any] = await source_status(session, project, source)
    reason = _blocked_reason(source, project_run_live=project_run_live)
    run = data['last_run']
    api = get_curation_config().api_prefix
    return IngestSourceStatus(
        project=slug(),
        source_id=source.source_id,
        name=source.name,
        backend=source.backend,
        location=source.location,
        s3=S3Location(bucket=source.bucket, prefix=source.prefix)
        if source.backend == 's3'
        else None,
        folder=source.path if source.backend == 'folder' else None,
        extensions=list(source.extensions),
        source=source.source_tag,
        batch_size=source.batch_size,
        concurrency=source.concurrency,
        triggers=TriggersWire(
            watch=WatchTriggerWire(
                enabled=source.watch.enabled, interval_s=source.watch.interval_s
            ),
            schedule=ScheduleTriggerWire(
                enabled=source.schedule.enabled, cron=source.schedule.cron
            ),
            webhook_enabled=source.webhook_enabled,
        ),
        created_at=source.created_at,
        counts=SourceCounts(**data['counts']),
        scan=SourceScanStatus(**data['scan']),
        watch=SourceWatchStatus(**data['watch']),
        schedule=SourceScheduleStatus(**data['schedule']),
        webhook=SourceWebhookStatus(
            path=f'{api}/projects/{slug()}/ingest/sources/{source.source_id}/events',
            **data['webhook'],
        ),
        last_run=SourceRunSummary(**run) if run else None,
        errors=[SourceObjectError(**e) for e in data['errors']],
        can_run=reason is None and not run_is_live(run),
        can_retry_dead=data['counts']['dead'] > 0,
        blocked_reason=reason,
        blocked_message=_BLOCKED_MESSAGES.get(reason) if reason else None,
    )


def _apply_triggers(source: IngestSource, triggers: TriggersWire) -> None:
    source.watch = WatchTrigger(triggers.watch.enabled, triggers.watch.interval_s)
    source.schedule = ScheduleTrigger(triggers.schedule.enabled, triggers.schedule.cron.strip())
    source.webhook_enabled = triggers.webhook_enabled


async def _save(source: IngestSource) -> IngestSource:
    try:
        return await save_source(source)
    except SourceNameTakenError:
        raise api_error(
            409, 'ingest_source_invalid', f'a source named {source.name!r} exists', project=slug()
        ) from None


@router.get('/ingest/sources', response_model=IngestSourceList)
async def list_ingest_sources(session: DbSessionDep) -> IngestSourceList:
    s3 = S3Config.from_env()
    project = current_project_id()
    live = await live_run(session, project) is not None
    items = [
        await status_wire(session, project, s, project_run_live=live) for s in await list_sources()
    ]
    busy = any(
        i.scan.active or run_is_live(i.last_run.model_dump() if i.last_run else None) for i in items
    )
    return IngestSourceList(
        items=items,
        s3_configured=s3.configured,
        s3_allowed=[f'{b}/{p}' if p else b for b, p in s3.allowed],
        default_batch_size=limits.DEFAULT_BATCH_SIZE,
        default_concurrency=limits.DEFAULT_CONCURRENCY,
        default_watch_interval_s=DEFAULT_WATCH_INTERVAL_S,
        min_watch_interval_s=MIN_WATCH_INTERVAL_S,
        poll_after_s=LIVE_POLL_S if busy else IDLE_POLL_S,
    )


async def _one(session: AsyncSession, source: IngestSource) -> IngestSourceStatus:
    project = current_project_id()
    live = await live_run(session, project) is not None
    return await status_wire(session, project, source, project_run_live=live)


@router.post('/ingest/sources', response_model=IngestSourceStatus, status_code=201)
async def create_ingest_source(
    body: IngestSourceRequest, session: DbSessionDep
) -> IngestSourceStatus:
    source = IngestSource(
        name=body.name,
        backend=body.backend,
        bucket=body.s3.bucket if body.s3 else '',
        prefix=body.s3.prefix.lstrip('/') if body.s3 else '',
        path=body.folder or '',
        extensions=list(normalise_extensions(body.extensions)),
        source_tag=body.source,
        batch_size=body.batch_size,
        concurrency=body.concurrency,
    )
    _apply_triggers(source, body.triggers)
    check_definition(source)
    return await _one(session, await _save(source))


@router.get('/ingest/sources/{source_id}', response_model=IngestSourceStatus)
async def get_ingest_source(source_id: str, session: DbSessionDep) -> IngestSourceStatus:
    return await _one(session, await _load(source_id))


@router.patch('/ingest/sources/{source_id}', response_model=IngestSourceStatus)
async def update_ingest_source(
    source_id: str, body: IngestSourcePatch, session: DbSessionDep
) -> IngestSourceStatus:
    source = await _load(source_id)
    if body.name is not None:
        source.name = body.name
    if body.extensions is not None:
        source.extensions = list(normalise_extensions(body.extensions))
    if body.source is not None:
        source.source_tag = body.source
    if body.batch_size is not None:
        source.batch_size = body.batch_size
    if body.concurrency is not None:
        source.concurrency = body.concurrency
    if body.triggers is not None:
        _apply_triggers(source, body.triggers)
    check_definition(source)
    return await _one(session, await _save(source))


@router.delete('/ingest/sources/{source_id}', status_code=204)
async def delete_ingest_source(source_id: str, session: DbSessionDep) -> None:
    """Remove the definition and its queue. Images already ingested stay."""
    source = await _load(source_id)
    live = await live_run(session, current_project_id())
    if live is not None and (live.params or {}).get('source_id') == source.source_id:
        raise api_error(
            409,
            'bulk_ingest_busy',
            'a run of this source is live',
            run_id=live.id,
            project=slug(),
        )
    await delete_source(source_id)


async def start_run_now(
    session: AsyncSession,
    source: IngestSource,
    opensearch: OpenSearchDep,
    registry: RegistryDep,
    *,
    checked: bool = False,
) -> BulkIngestRun:
    """Scan the source, then drain its queue, as a run (409 while the project has one live).

    A source that stopped with work left is continued (``resumed``); a finished one-shot source
    starts over."""
    if not checked:
        check_definition(source)
    # Refuse up front (503) what the first batch would refuse: no detector, no encoder.
    await get_ingest_service(opensearch, registry)
    project = current_project_id()
    state = await asyncio.to_thread(source_queue(source).meta_all)
    previous = await last_run(session, project, state)
    resumed = previous is not None and previous.status in RESUMABLE_RUN_STATUSES
    try:
        run_id = await start_source_run(
            session, project, source, scan=True, fresh=source.one_shot and not resumed
        )
    except BulkBusyError as exc:
        raise api_error(
            409, 'bulk_ingest_busy', str(exc), run_id=exc.run_id, project=slug()
        ) from None
    run = await get_run(session, project, str(run_id))
    assert run is not None  # the row was committed by the start
    return run_wire(run, resumed=resumed)


async def resolve_bulk_source(body: BulkIngestRequest) -> IngestSource:
    """The source a ``s3`` or ``source_id`` bulk start names: a registered one, or the
    trigger-less source of the bucket/prefix (created on first use, so a repeat start
    of the same bucket/prefix resumes its run)."""
    if body.source_id is not None:
        return await _load(body.source_id)
    assert body.s3 is not None
    source = IngestSource(
        name='',
        backend='s3',
        bucket=body.s3.bucket,
        prefix=body.s3.prefix.lstrip('/'),
        extensions=list(normalise_extensions(body.extensions)),
        source_tag=body.source,
        batch_size=body.batch_size,
        concurrency=body.concurrency,
    )
    check_definition(source)
    return await one_shot_source(source)


@router.post(
    '/ingest/sources/{source_id}/run',
    dependencies=[DISK_GUARD],
    response_model=BulkIngestRun,
    status_code=202,
)
async def run_ingest_source(
    source_id: str, session: DbSessionDep, opensearch: OpenSearchDep, registry: RegistryDep
) -> BulkIngestRun:
    """Run now: scan the source and ingest what is new, or resume its interrupted run."""
    return await start_run_now(session, await _load(source_id), opensearch, registry)


@router.post('/ingest/sources/{source_id}/retry-dead', response_model=RetryDeadResult)
async def retry_dead_objects(source_id: str, session: DbSessionDep) -> RetryDeadResult:
    """Put the objects that gave up (or can never succeed) back in the queue; the next
    run tries them again."""
    source = await _load(source_id)
    retried = await asyncio.to_thread(source_queue(source).retry_dead)
    return RetryDeadResult(retried=retried, source=await _one(session, source))


def _record_matches(source: IngestSource, record: EventRecord, s3: S3Config) -> bool:
    if not record.is_created or record.bucket != source.bucket:
        return False
    if not record.key.startswith(source.prefix) or not wanted_extension(
        record.key, frozenset(source.extensions)
    ):
        return False
    try:
        s3.check(record.bucket, record.key)
    except (S3NotAllowedError, S3NotConfiguredError):
        return False
    return True


def _queue_events(source: IngestSource, records: list[EventRecord]) -> EventsAccepted:
    s3 = S3Config.from_env()
    wanted = [r for r in records if _record_matches(source, r, s3)]
    queue = source_queue(source)
    queued, duplicates = queue.enqueue(
        QueueEntry(r.key, r.etag, r.version_id, r.size) for r in wanted
    )
    ignored = len(records) - len(wanted)
    queue.add_counters(
        {
            'events.received': len(records),
            'events.queued': queued,
            'events.duplicates': duplicates,
            'events.ignored': ignored,
        }
    )
    queue.meta_set('events.last_at', time.time())
    return EventsAccepted(
        received=len(records), queued=queued, duplicates=duplicates, ignored=ignored
    )


@router.post('/ingest/sources/{source_id}/events', response_model=EventsAccepted, status_code=202)
async def receive_s3_events(source_id: str, request: Request) -> EventsAccepted:
    """S3 event notification webhook of one source. Idempotent: a record is deduplicated by
    bucket, key and etag, so a redelivery queues nothing. Records for other buckets,
    prefixes or file types, and non-create events, are ignored (a bucket-wide
    notification can feed several sources). The request only writes queue rows; the
    dispatcher starts the run, so the answer is fast."""
    source = await _load(source_id)
    if not source.webhook_enabled:
        raise api_error(
            409, 'webhook_disabled', 'this source does not accept events', project=slug()
        )
    try:
        body = await request.json()
    except ValueError:
        raise api_error(
            422, 'invalid_s3_event', '$: the body is not JSON', project=slug()
        ) from None
    try:
        records = parse_s3_event(body)
    except InvalidS3EventError as exc:
        raise api_error(422, 'invalid_s3_event', str(exc), project=slug()) from None
    return await asyncio.to_thread(_queue_events, source, records)
