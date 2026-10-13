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
from src.routers.curation._common import OpenSearchDep, RegistryDep, _ensure_indexes, router
from src.routers.curation._config_common_models import api_error
from src.routers.curation._ingest_bulk_models import BulkIngestRequest, BulkIngestRun
from src.routers.curation._ingest_bulk_wire import own_address, run_wire, slug
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
from src.routers.curation.ingest import _get_ingest_service
from src.services.curation.bulk_ingest import limits, runner
from src.services.curation.bulk_ingest.object_queue import QueueEntry
from src.services.curation.bulk_ingest.object_store import (
    S3Config,
    S3NotAllowedError,
    S3NotConfiguredError,
)
from src.services.curation.bulk_ingest.s3_events import (
    EventRecord,
    InvalidS3EventError,
    parse_s3_event,
)
from src.services.curation.bulk_ingest.s3_listing import wanted_extension
from src.services.curation.bulk_ingest.source import normalise_extensions
from src.services.curation.bulk_ingest.source_runs import (
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
    WatchTrigger,
    delete_source,
    list_sources,
    load_source,
    new_source_id,
    s3_source_id,
    save_source,
    source_queue,
    validate_source_definition,
)
from src.services.curation.bulk_ingest.store import active_run


if TYPE_CHECKING:
    from typing import Any

    from fastapi import HTTPException


def _not_found(source_id: str) -> HTTPException:
    return api_error(
        404, 'ingest_source_not_found', f'no ingest source {source_id!r}', project=slug()
    )


def _load(source_id: str) -> IngestSource:
    source = load_source(source_id)
    if source is None:
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


def _blocked_reason(source: IngestSource) -> str | None:
    try:
        validate_source_definition(source, S3Config.from_env())
    except S3NotConfiguredError:
        return 's3_not_configured'
    except S3NotAllowedError:
        return 's3_not_allowed'
    except InvalidSourceDefinitionError:
        return 'invalid_source'
    return 'project_run_live' if active_run() is not None else None


def status_wire(source: IngestSource) -> IngestSourceStatus:
    data: dict[str, Any] = source_status(source)
    reason = _blocked_reason(source)
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
        blocked_reason=reason,  # type: ignore[arg-type]
    )


def _apply_triggers(source: IngestSource, triggers: TriggersWire) -> None:
    source.watch = WatchTrigger(triggers.watch.enabled, triggers.watch.interval_s)
    source.schedule = ScheduleTrigger(triggers.schedule.enabled, triggers.schedule.cron.strip())
    source.webhook_enabled = triggers.webhook_enabled


@router.get('/ingest/sources', response_model=IngestSourceList)
async def list_ingest_sources() -> IngestSourceList:
    s3 = S3Config.from_env()
    return IngestSourceList(
        items=[status_wire(s) for s in list_sources()],
        s3_configured=s3.configured,
        s3_allowed=[f'{b}/{p}' if p else b for b, p in s3.allowed],
        default_batch_size=limits.DEFAULT_BATCH_SIZE,
        default_concurrency=limits.DEFAULT_CONCURRENCY,
        default_watch_interval_s=DEFAULT_WATCH_INTERVAL_S,
        min_watch_interval_s=MIN_WATCH_INTERVAL_S,
    )


@router.post('/ingest/sources', response_model=IngestSourceStatus, status_code=201)
async def create_ingest_source(body: IngestSourceRequest) -> IngestSourceStatus:
    source = IngestSource(
        source_id=new_source_id(),
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
    save_source(source)
    return status_wire(source)


@router.get('/ingest/sources/{source_id}', response_model=IngestSourceStatus)
async def get_ingest_source(source_id: str) -> IngestSourceStatus:
    return status_wire(_load(source_id))


@router.patch('/ingest/sources/{source_id}', response_model=IngestSourceStatus)
async def update_ingest_source(source_id: str, body: IngestSourcePatch) -> IngestSourceStatus:
    source = _load(source_id)
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
    save_source(source)
    return status_wire(source)


@router.delete('/ingest/sources/{source_id}', status_code=204)
async def delete_ingest_source(source_id: str) -> None:
    """Remove the definition and its queue. Images already ingested stay."""
    _load(source_id)
    live = active_run()
    if live is not None and live.job.read().get('source_key') == f'queue:{source_id}':
        raise api_error(
            409,
            'bulk_ingest_busy',
            'a run of this source is live',
            run_id=live.run_id,
            project=slug(),
        )
    delete_source(source_id)


async def start_run_now(
    source: IngestSource, request: Request, opensearch: OpenSearchDep, registry: RegistryDep
) -> BulkIngestRun:
    """Scan the source, then drain its queue, as a run (409 while the project has one live)."""
    check_definition(source)
    await _ensure_indexes(opensearch)
    # Refuse up front (503) what the first batch would refuse: no detector, no encoder.
    await _get_ingest_service(opensearch, registry)
    host, port = own_address(request)
    netloc = f'[{host}]' if ':' in host else host
    try:
        store, resumed = start_source_run(
            source,
            base_url=f'http://{netloc}:{port}',
            api_prefix=get_curation_config().api_prefix,
            project=slug(),
            scan=True,
        )
    except runner.BulkBusyError as exc:
        raise api_error(
            409, 'bulk_ingest_busy', str(exc), run_id=exc.run_id, project=slug()
        ) from None
    return run_wire(store, resumed=resumed)


def resolve_bulk_source(body: BulkIngestRequest) -> IngestSource:
    """The source a ``s3`` or ``source_id`` bulk start names: a registered one, or the
    trigger-less source of the bucket/prefix (created on first use, so a repeat start
    of the same bucket/prefix resumes its run)."""
    if body.source_id is not None:
        return _load(body.source_id)
    assert body.s3 is not None
    prefix = body.s3.prefix.lstrip('/')
    source_id = s3_source_id(body.s3.bucket, prefix)
    existing = load_source(source_id)
    if existing is not None:
        return existing
    source = IngestSource(
        source_id=source_id,
        name=f's3://{body.s3.bucket}/{prefix}',
        backend='s3',
        bucket=body.s3.bucket,
        prefix=prefix,
        extensions=list(normalise_extensions(body.extensions)),
        source_tag=body.source,
        batch_size=body.batch_size,
        concurrency=body.concurrency,
    )
    check_definition(source)
    save_source(source)
    return source


@router.post('/ingest/sources/{source_id}/run', response_model=BulkIngestRun, status_code=202)
async def run_ingest_source(
    source_id: str, request: Request, opensearch: OpenSearchDep, registry: RegistryDep
) -> BulkIngestRun:
    """Run now: scan the source and ingest what is new, or resume its interrupted run."""
    return await start_run_now(_load(source_id), request, opensearch, registry)


@router.post('/ingest/sources/{source_id}/retry-dead', response_model=RetryDeadResult)
async def retry_dead_objects(source_id: str) -> RetryDeadResult:
    """Put the objects that gave up (or can never succeed) back in the queue; the next
    run tries them again."""
    source = _load(source_id)
    retried = source_queue(source_id).retry_dead()
    return RetryDeadResult(retried=retried, source=status_wire(source))


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
    queue = source_queue(source.source_id)
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
    source = _load(source_id)
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
