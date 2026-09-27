"""Curation SSE event endpoints — split out of pipeline.py for the file-size gate."""

from __future__ import annotations

import asyncio as _events_asyncio
import json as _events_json
from typing import Any

from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse

from src.config import get_curation_config
from src.routers.curation._common import _PublishEvent, router
from src.services.curation.event_hub import get_event_hub
from src.services.curation.wire import region_wire_key


_SSE_HEARTBEAT_SECONDS = 15.0


@router.get('/events')
async def curation_events(
    topic: str | None = Query(
        default=None,
        description='Optional topic filter (e.g. the region-status topic, "crop").',
    ),
    class_id: int | None = Query(
        default=None,
        description='Optional class_id filter — only emit crop.classified '
        'events whose class matches.',
    ),
) -> StreamingResponse:
    """SSE stream of advisory crop-state events for the bound project
    only (global ``project: null`` events go to ``{api}/events``).

    The connection stays open until the client disconnects. A heartbeat
    comment line is sent every 15s so reverse-proxies don't kill the
    socket. Subscribers each get their own bounded queue (capped at
    1000 events; oldest drops on overflow — events are advisory).
    """
    hub = get_event_hub()
    sub = await hub.subscribe(
        project=get_curation_config().project_slug, topic=topic, class_id=class_id
    )
    return sse_response(hub, sub)


def sse_response(hub: Any, sub: Any) -> StreamingResponse:
    """Stream ``sub``'s queue as SSE until the client disconnects, then
    unsubscribe. Shared by the scoped ``/events`` and the global one."""

    async def event_gen() -> Any:
        try:
            # Initial comment so the client EventSource transitions to
            # "open" immediately even before the first real event.
            yield ': connected\n\n'
            while True:
                try:
                    ev = await _events_asyncio.wait_for(
                        sub.queue.get(), timeout=_SSE_HEARTBEAT_SECONDS
                    )
                except TimeoutError:
                    yield ': hb\n\n'
                    continue
                payload = _events_json.dumps(ev, default=str)
                yield f'event: {ev.get("type", "message")}\ndata: {payload}\n\n'
        except _events_asyncio.CancelledError:
            # Client disconnected — clean up and re-raise so Starlette
            # finalises the response.
            raise
        finally:
            await hub.unsubscribe(sub)

    return StreamingResponse(
        event_gen(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',  # disable nginx buffering
            'Connection': 'keep-alive',
        },
    )


@router.post('/events/publish')
async def curation_events_publish(payload: _PublishEvent) -> dict[str, Any]:
    """Publish one event into the in-process hub.

    Used by external publishers like ``scripts/curation/region_worker_main.py``
    that don't share the API process. Events from in-process callers
    (ingest, VLM label_batch) skip this endpoint and call the hub directly.
    """
    status_key = region_wire_key('status')
    event: dict[str, Any] = {
        'type': payload.type,
        'topic': payload.topic
        or (status_key if payload.type == 'crop.region_verified' else 'crop'),
    }
    if payload.crop_id is not None:
        event['crop_id'] = payload.crop_id
    if payload.class_id is not None:
        event['class_id'] = payload.class_id
    if payload.class_name is not None:
        event['class_name'] = payload.class_name
    if payload.class_source is not None:
        event['class_source'] = payload.class_source
    if payload.region_status is not None:
        event[status_key] = payload.region_status
    if payload.region_text is not None:
        event[region_wire_key('text')] = payload.region_text
    if payload.image_path is not None:
        event['image_path'] = payload.image_path
    bound = get_curation_config().project_slug
    extra = dict(payload.extra or {})
    if 'project' in extra and extra.pop('project') != bound:
        # The bound project always wins; an event cannot be redirected to
        # another project's stream (or the global one) from here.
        raise HTTPException(
            status_code=422,
            detail=f"extra.project must be the bound project '{bound}' (or omitted)",
        )
    extra.pop('target', None)
    event.update(extra)
    event['project'] = bound
    get_event_hub().publish(event)
    return {'ok': True}


@router.get('/events/stats')
async def curation_events_stats() -> dict[str, Any]:
    """Ops counters for the bound project's stream: its subscribers and
    publishes, plus the hub's ``events_dropped``, active ``bus``
    (``file``/``process``) and ``log_path``."""
    return get_event_hub().stats(get_curation_config().project_slug)
