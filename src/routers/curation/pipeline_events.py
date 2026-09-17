"""Curation router sub-module — pipeline SSE event stream.

Split out of :mod:`pipeline` so the parent module stays under the
700-LOC pre-commit ceiling. Defines exactly one endpoint:

* ``GET /curation/pipeline/events`` — server-sent events for the dashboard.

See the endpoint docstring for the event-type contract. The wakeup
mechanism is an asyncio ``Event`` signalled by the inotify watcher
registered in :mod:`src.main` (lifespan task ``watch_state_file``).
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

from src.routers.curation._common import OpenSearchDep, router


@router.get('/pipeline/events')
async def pipeline_events(opensearch: OpenSearchDep) -> StreamingResponse:
    """Server-sent events stream for the curation dashboard.

    Replaces a ``GET /curation/stats/dataset`` polling loop. One open
    HTTP connection per dashboard tab; the server pushes:

    * ``snapshot`` (on connect) — current ``state.json`` plus a one-shot
      dataset aggregation, so the client renders without any additional
      REST call.
    * ``state`` — whenever ``auto_label`` state changes (stage
      transitions, progress advances, terminal status). Driven by the
      inotify watcher in :func:`auto_label_job.watch_state_file`; zero
      CPU when idle.
    * ``stats`` — refreshed dataset aggregation. Emitted at auto_label
      stage boundaries AND on a periodic timer (``STATS_REFRESH_SECONDS``)
      so the dashboard counts (unlabeled / pending_detection /
      pending_verification) keep ticking while the detection worker
      drains the queue independent of auto_label. Frontends that only
      care about pipeline status can ignore these.
    * heartbeat comment frame every 15 s so reverse proxies don't reap
      the connection. This is the only fixed-rate traffic on the
      channel.

    The dashboard consumes this with ``new EventSource(url)`` and
    switches on ``ev.type`` (snapshot | state | stats).
    """
    from src.routers.curation.stats import stats_dataset
    from src.services.curation.autolabel import job as auto_label_job

    HEARTBEAT_SECONDS = 15.0
    # Periodic stats refresh — the detection worker writes back to the
    # items index continuously as it drains, but those writes don't touch
    # auto_label state.json so the inotify-driven path never wakes for
    # them. Without a fallback timer the dashboard would only see updated
    # counts at auto_label stage boundaries (i.e. minutes between
    # refreshes when auto_label isn't running at all). 15s matches the
    # heartbeat cadence so we ride the same wakeup cycle.
    STATS_REFRESH_SECONDS = 15.0

    async def _build_stats_payload() -> dict[str, Any]:
        """Single OpenSearch round-trip for the dashboard rollup. Same
        body the REST /curation/stats/dataset endpoint serves — reusing
        the handler keeps the two outputs identical."""
        try:
            return await stats_dataset(opensearch)
        except Exception as exc:
            return {'error': f'{type(exc).__name__}: {exc}'}

    def _sse(event: str, data: Any) -> str:
        body = json.dumps(data, default=str)
        return f'event: {event}\ndata: {body}\n\n'

    async def _gen() -> AsyncIterator[str]:
        import time as _time

        changed = auto_label_job.auto_label_changed_event()
        # Bootstrap. Send the current state + stats so the client renders
        # without any parallel REST call.
        state = auto_label_job.get_state()
        stats = await _build_stats_payload()
        yield _sse('snapshot', {'state': state, 'stats': stats})

        last_stage = state.get('stage')
        last_status = state.get('status')
        last_stats_at = _time.monotonic()
        while True:
            # Race the inotify-driven Event against the heartbeat clock.
            # Whichever wakes first determines what we emit next.
            wait_task = asyncio.create_task(changed.wait())
            try:
                done, pending = await asyncio.wait(
                    {wait_task},
                    timeout=HEARTBEAT_SECONDS,
                )
            except asyncio.CancelledError:
                wait_task.cancel()
                raise
            now = _time.monotonic()
            if wait_task in done:
                changed.clear()
                state = auto_label_job.get_state()
                yield _sse('state', state)
                stage = state.get('stage')
                status = state.get('status')
                # Only re-query stats when something the dashboard's
                # dataset rollup would care about flipped (stage
                # transition or terminal status). Skips repeated stats
                # frames during a single stage's progress ticks.
                if stage != last_stage or status != last_status:
                    last_stage = stage
                    last_status = status
                    stats = await _build_stats_payload()
                    last_stats_at = now
                    yield _sse('stats', stats)
            else:
                # Timeout fired — heartbeat. Cancel the pending wait so
                # we don't accumulate tasks.
                for t in pending:
                    t.cancel()
                yield ': keepalive\n\n'
            # Periodic stats refresh regardless of auto_label activity.
            # The detection worker keeps writing back to the items index
            # as it drains the queue; this lets the dashboard reflect
            # those counts in near-real-time without coupling to
            # auto_label.
            if (now - last_stats_at) >= STATS_REFRESH_SECONDS:
                stats = await _build_stats_payload()
                last_stats_at = now
                yield _sse('stats', stats)

    return StreamingResponse(
        _gen(),
        media_type='text/event-stream',
        headers={
            # Disable nginx/proxy buffering. Without this, events sit in
            # the proxy buffer until enough have accumulated to flush.
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache, no-transform',
        },
    )
