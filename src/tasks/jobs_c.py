"""Tasks of lane 4C: project delete, ingest and the open-vocabulary drain.

Each handler calls the family's existing coroutine with clients from the job context, never from
a request (stage 4 recipe step 3). The OpenSearch factory is injected here and not in
``src/tasks/runtime/`` (review R-2).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.db.session import async_session_factory
from src.tasks.runtime.context import JobResult
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from src.tasks.runtime.context import JobContext

OPENSEARCH_TIMEOUT_S = 60
logger = get_logger(__name__)


async def _opensearch() -> Any:
    """A project-guarded client of this task's own; the wrapper closes it with the task."""
    from src.config.settings import get_settings
    from src.services.projects.guard import make_script_opensearch

    return make_script_opensearch([get_settings().opensearch_url], timeout=OPENSEARCH_TIMEOUT_S)


async def bound_project(ctx: JobContext) -> AbstractContextManager[Any]:
    """Bind the job's project for the handler (a worker has no request to bind it)."""
    from src.config.project_context import bind_project
    from src.services.projects.registry import ProjectRegistry

    registry = ProjectRegistry()
    await registry.ensure_fresh()
    record = registry.get(str(ctx.project_slug))
    if record is None:
        raise LookupError(f'no project {ctx.project_slug!r} for job {ctx.job_id}')
    return bind_project(record)


async def project_delete(ctx: JobContext) -> dict[str, Any]:
    """Finish the delete of ``params['slug']``; the rows go in batches with progress in the job."""
    from src.services.events.publish import apublish_global_event
    from src.services.projects.delete import delete_project_finish

    slug = str(ctx.params['slug'])

    async def rows_deleted(table: str, deleted: int) -> None:
        await ctx.checkpoint({'table': table, 'deleted': deleted})
        await ctx.progress(0, None, 'rows', {'table': table, 'deleted': deleted})

    finished = await delete_project_finish(
        await ctx.client('opensearch'), slug=slug, on_rows_deleted=rows_deleted
    )
    cleanup = finished.mlflow_cleanup.to_wire()
    await apublish_global_event(
        'project.deleted',
        target=slug,
        status=finished.record.status,
        revision=finished.record.revision,
        mlflow_cleanup=cleanup,
    )
    return {'project': slug, 'mlflow_cleanup': cleanup}


async def open_vocab_drain(ctx: JobContext) -> dict[str, Any]:
    """Run the active open-vocabulary set over every ``pending`` image of the project."""
    from src.services.classes.registry import get_class_registry
    from src.services.curation.ingest_service_factory import get_ingest_service
    from src.services.curation.open_vocab_ingest import drain_pending
    from src.services.detection.segmenter_http import segment_image_http

    with await bound_project(ctx):
        opensearch = await ctx.client('opensearch')
        service = await get_ingest_service(opensearch, get_class_registry())

        async def batch_done(tried: int) -> None:
            await ctx.progress(tried, None, 'drain')

        return await drain_pending(
            opensearch, service, segment_image_http, should_stop=ctx.cancelled, on_batch=batch_done
        )


async def _search_service() -> Any:
    """The visual-search service over a client of this task's own (closed with it)."""
    from src.clients.opensearch.client import OpenSearchClient

    return OpenSearchClient()


async def directory_ingest(ctx: JobContext) -> JobResult | dict[str, Any]:
    """Ingest the directory in ``params``; a manual resume skips the batches already finished."""
    from src.services.visual_search import VisualSearchService
    from src.services.visual_search.directory_ingest import (
        DirectoryIngestOptions,
        run_directory_ingest,
    )

    resume = (ctx.resume_from or {}).get('batch', 0)

    async def batch_done(batch: int, done: int, total: int) -> None:
        await ctx.checkpoint({'batch': batch})
        await ctx.progress(done, total, 'ingest')

    service = VisualSearchService(await ctx.client('opensearch_client'))
    result = await run_directory_ingest(
        service,
        DirectoryIngestOptions.from_params(ctx.params),
        resume_batch=resume,
        should_stop=ctx.cancelled,
        on_batch=batch_done,
        servers=[ctx.server] if ctx.server else None,
    )
    return JobResult(result, with_errors=result['status'] != 'success')


async def bulk_ingest_run(ctx: JobContext) -> JobResult:
    """Drain the source's queue through the ingest route; a resume continues the counts."""
    from src.config import get_curation_config
    from src.services.curation.bulk_ingest.ledger import DbLedger
    from src.services.curation.bulk_ingest.limits import loopback_url
    from src.services.curation.bulk_ingest.runner import BulkIngestRun, BulkOptions
    from src.services.curation.bulk_ingest.s3_listing import ScanStoppedError
    from src.services.curation.bulk_ingest.sender import url_sender
    from src.services.curation.bulk_ingest.source_scan import scan_source
    from src.services.curation.bulk_ingest.sources import load_source, source_queue

    with await bound_project(ctx):
        source = await load_source(str(ctx.params['source_id']))
        if source is None:
            raise LookupError(f'no ingest source {ctx.params["source_id"]!r}')
        queue = source_queue(source)
        if ctx.resume_from is None and ctx.params.get('fresh'):
            await asyncio.to_thread(queue.requeue_all)
        await asyncio.to_thread(queue.release_claims)  # the claims of a run that died
        if ctx.params.get('scan'):
            # a stop request ends the scan early: the files queued so far are still drained
            with contextlib.suppress(ScanStoppedError):
                await asyncio.to_thread(
                    scan_source, source, should_stop=ctx.cancelled, full=source.one_shot
                )
        sender = url_sender(
            loopback_url(),
            api_prefix=get_curation_config().api_prefix,
            project=str(ctx.project_slug),
            objects=source.backend == 's3',
        )
        try:
            run = BulkIngestRun(
                sender=sender,
                ledger=DbLedger(ctx.session, uuid.UUID(source.source_id), ctx.job_id),
                reporter=ctx,
                options=BulkOptions(**ctx.params['options']),
                run_id=str(ctx.job_id),
                prior=ctx.resume_from,
                servers=[ctx.server] if ctx.server else [],
            )
            result = await run.run()
        finally:
            await sender.aclose()
    return JobResult(result, with_errors=result['failed'] > 0)


async def source_poll(ctx: JobContext) -> dict[str, Any]:
    """One watcher pass over a source: scan into its queue, then start a run if rows are pending."""
    from src.services.curation.bulk_ingest.dispatcher import backoff_s
    from src.services.curation.bulk_ingest.object_store import (
        ObjectStoreError,
        S3NotAllowedError,
        S3NotConfiguredError,
    )
    from src.services.curation.bulk_ingest.s3_listing import ScanStoppedError
    from src.services.curation.bulk_ingest.source_runs import BulkBusyError, start_source_run
    from src.services.curation.bulk_ingest.source_scan import scan_source
    from src.services.curation.bulk_ingest.sources import load_source, source_queue

    with await bound_project(ctx):
        source = await load_source(str(ctx.params['source_id']))
        if source is None:
            return {'source': 'removed'}
        queue = source_queue(source)
        try:
            await asyncio.to_thread(scan_source, source, should_stop=ctx.cancelled)
        except ScanStoppedError:
            return {'stopped': True}
        except (ObjectStoreError, S3NotAllowedError, S3NotConfiguredError, OSError) as exc:
            failures = int(await asyncio.to_thread(queue.meta_get, 'failures', 0) or 0) + 1
            await asyncio.to_thread(
                queue.meta_update,
                {
                    'failures': failures,
                    'last_poll_error': f'{type(exc).__name__}: {exc}'[:300],
                    'backoff_until': time.time() + backoff_s(failures),
                },
            )
            logger.warning('ingest_source_poll_failed', source=source.source_id, error=str(exc))
            return {'failed': True, 'failures': failures}
        await asyncio.to_thread(
            queue.meta_update, {'failures': 0, 'last_poll_error': None, 'backoff_until': None}
        )
        counts = await asyncio.to_thread(queue.counts)
        assert ctx.project_id is not None  # a project-scoped kind
        started = False
        if counts['pending'] > 0:
            async with async_session_factory()() as session:
                try:
                    await start_source_run(session, ctx.project_id, source, scan=False)
                    started = True
                except BulkBusyError:
                    pass  # a run holds the project; it drains what the scan queued
        return {'pending': counts['pending'], 'run_started': started}


register_job('project_delete', project_delete, client_factories={'opensearch': _opensearch})
register_job('open_vocab_drain', open_vocab_drain, client_factories={'opensearch': _opensearch})
register_job(
    'directory_ingest', directory_ingest, client_factories={'opensearch_client': _search_service}
)
register_job('bulk_ingest_run', bulk_ingest_run)
register_job('source_poll', source_poll)
