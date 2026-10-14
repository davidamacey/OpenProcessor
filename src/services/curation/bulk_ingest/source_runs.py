"""Start runs of a registered source and describe where it stands.

``start_source_run`` is the one way a source's queue becomes a bulk run: the routes
(run now, one-shot bucket ingest) and the dispatcher (poll, schedule, webhook) all call
it, so there is one runner, one claim and one set of per-project rules.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from src.services.curation.bulk_ingest import runner
from src.services.curation.bulk_ingest.queue_listener import QueueListener
from src.services.curation.bulk_ingest.s3_listing import LAST_SCAN_META, SCAN_META
from src.services.curation.bulk_ingest.schedule import Cron
from src.services.curation.bulk_ingest.sender import url_sender
from src.services.curation.bulk_ingest.source import BulkSource
from src.services.curation.bulk_ingest.sources import IngestSource, source_queue, source_state
from src.services.curation.bulk_ingest.store import ACTIVE_STATUSES, BulkStore, open_store


def bulk_source(source: IngestSource, *, scan: bool) -> BulkSource:
    return BulkSource('queue', source.source_id, tuple(source.extensions), scan)


def bulk_options(source: IngestSource) -> runner.BulkOptions:
    return runner.BulkOptions(source.source_tag, source.batch_size, source.concurrency)


def start_source_run(
    source: IngestSource, *, base_url: str, api_prefix: str, project: str, scan: bool
) -> tuple[BulkStore, bool]:
    """Claim and spawn a run draining ``source``'s queue (after a scan when ``scan``).
    Raises :class:`~src.services.curation.bulk_ingest.runner.BulkBusyError` while any run
    of the project is live."""
    spec = bulk_source(source, scan=scan)
    options = bulk_options(source)
    store, resumed = runner.claim_run(spec, options)
    source_state(source.source_id).update(last_run_id=store.run_id, run_requested=False)
    runner.spawn(
        sender=url_sender(
            base_url, api_prefix=api_prefix, project=project, objects=source.backend == 's3'
        ),
        store=store,
        source=spec,
        options=options,
        listener=QueueListener(source),
    )
    return store, resumed


def _iso(epoch: float | None) -> str | None:
    return datetime.fromtimestamp(epoch, UTC).isoformat() if epoch else None


def next_poll_at(source: IngestSource, state: dict[str, Any]) -> float | None:
    if not source.watch.enabled:
        return None
    last = state.get('last_poll_started_at')
    due = (last + source.watch.interval_s) if last else time.time()
    return max(due, state.get('backoff_until') or 0.0)


def next_fire_at(source: IngestSource, state: dict[str, Any]) -> str | None:
    if not source.schedule.enabled:
        return None
    last = state.get('last_fire_at')
    after = datetime.fromisoformat(last) if last else datetime.now(UTC)
    return Cron.parse(source.schedule.cron).next_after(after).isoformat()


def last_run_summary(state: dict[str, Any]) -> dict[str, Any] | None:
    run_id = state.get('last_run_id')
    store = open_store(run_id) if run_id else None
    if store is None:
        return None
    job = store.job.read()
    return {
        'run_id': store.run_id,
        'status': job.get('status', 'queued'),
        'started_at': job.get('started_at'),
        'finished_at': job.get('finished_at'),
        'ingested': job.get('ingested', 0),
        'duplicates': job.get('duplicates', 0),
        'failed': job.get('failed', 0),
        'error': job.get('error'),
    }


def run_is_live(summary: dict[str, Any] | None) -> bool:
    return summary is not None and summary['status'] in ACTIVE_STATUSES


def source_status(source: IngestSource) -> dict[str, Any]:
    """Everything the UI shows for a source, computed here so the frontend only renders."""
    queue = source_queue(source.source_id)
    state = source_state(source.source_id).read()
    scan_progress = queue.meta_get(SCAN_META)
    return {
        'counts': queue.counts(),
        'scan': {
            'active': scan_progress is not None,
            'pages': (scan_progress or {}).get('pages', 0),
            'listed': (scan_progress or {}).get('listed', 0),
            'queued': (scan_progress or {}).get('queued', 0),
            'last': queue.meta_get(LAST_SCAN_META),
        },
        'watch': {
            'last_poll_at': _iso(state.get('last_poll_started_at')),
            'next_poll_at': _iso(next_poll_at(source, state)),
            'error': state.get('last_poll_error'),
            'backoff_until': _iso(state.get('backoff_until')),
        },
        'schedule': {
            'last_fire_at': state.get('last_fire_at'),
            'next_fire_at': next_fire_at(source, state),
        },
        'webhook': {
            'received': queue.meta_get('events.received', 0),
            'queued': queue.meta_get('events.queued', 0),
            'duplicates': queue.meta_get('events.duplicates', 0),
            'ignored': queue.meta_get('events.ignored', 0),
            'last_event_at': _iso(queue.meta_get('events.last_at')),
        },
        'last_run': last_run_summary(state),
        'errors': queue.errors(limit=20),
    }
