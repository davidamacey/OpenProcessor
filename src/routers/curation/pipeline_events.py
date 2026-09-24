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
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from opensearchpy import AsyncOpenSearch

from fastapi.responses import StreamingResponse

from src.routers.curation._common import OpenSearchDep, router


# F-21: the dataset-stats aggregation used to re-run once per SSE client
# every STATS_REFRESH_SECONDS (15s) -- N open dashboard tabs meant N
# identical `_search?size=0` round-trips every 15s. This module-level
# cache is shared by every SSE connection; the lock ensures that when
# several connections' timers fire in the same window, only the first
# actually queries OpenSearch -- the rest await the same in-flight
# refresh (or the fresh cache value it just wrote) instead of each
# issuing their own query.
_STATS_CACHE_TTL_SECONDS = 10.0
_stats_cache_lock = asyncio.Lock()
_stats_cache_payload: dict[str, Any] | None = None
_stats_cache_expires_at: float = 0.0


async def _cached_stats_payload(opensearch: AsyncOpenSearch) -> dict[str, Any]:
    """Return the dataset-stats payload, refreshing at most once per
    ``_STATS_CACHE_TTL_SECONDS`` across every concurrent SSE connection."""
    global _stats_cache_payload, _stats_cache_expires_at  # noqa: PLW0603

    from src.routers.curation.stats import stats_dataset

    async with _stats_cache_lock:
        # Re-check inside the lock: another connection may have just
        # refreshed it while we were waiting to acquire.
        if _stats_cache_payload is not None and time.monotonic() < _stats_cache_expires_at:
            return _stats_cache_payload
        try:
            payload = await stats_dataset(opensearch)
        except Exception as exc:
            payload = {'error': f'{type(exc).__name__}: {exc}'}
        _stats_cache_payload = payload
        _stats_cache_expires_at = time.monotonic() + _STATS_CACHE_TTL_SECONDS
        return payload


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
        """Dashboard rollup, shared across every open SSE connection via
        :func:`_cached_stats_payload`'s module-level TTL cache (F-21) —
        the REST ``/curation/stats/dataset`` endpoint's body, not a
        separate query."""
        return await _cached_stats_payload(opensearch)

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
