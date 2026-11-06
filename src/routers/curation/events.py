"""Curation SSE event endpoints and the worker publish route."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Header, HTTPException, Query
from fastapi.sse import EventSourceResponse

from src.config import get_curation_config
from src.routers.curation._common import router
from src.routers.curation._common_models import _PublishEvent
from src.routers.curation._event_streams import ProjectSubscriptionDep, to_sse
from src.services.curation.wire import region_wire_key
from src.services.events import publish as event_publish
from src.services.events.listener import get_listener
from src.services.events.sse import event_stream


@router.get('/events', response_class=EventSourceResponse)
async def curation_events(
    sub: ProjectSubscriptionDep,
    topic: str | None = Query(
        default=None,
        description='Optional topic filter (e.g. the region-status topic, "crop").',
    ),
    class_id: int | None = Query(
        default=None,
        description='Optional class_id filter — only emit crop.classified '
        'events whose class matches.',
    ),
    last_event_id: Annotated[
        str | None,
        Header(description='Resume point: the `id` of the last event the client received.'),
    ] = None,
) -> Any:
    """SSE stream of the bound project's events (global ``project: null`` events go to
    ``{api}/events``).

    Every frame carries ``id: <xid>-<id>``; a reconnect with ``Last-Event-ID`` replays what was
    missed. A cursor the server cannot honour (older than the 24 h retention, or from before a
    restore) gets one ``reset`` event first: reload state. A client that falls behind is dropped
    with a final ``reconnect`` event.
    """
    async for frame in event_stream(sub, last_event_id, topic=topic, class_id=class_id):
        yield to_sse(frame)


@router.post('/events/publish')
async def curation_events_publish(payload: _PublishEvent) -> dict[str, Any]:
    """Publish one event for the bound project.

    Used by external publishers like ``scripts/curation/worker/`` that have no database
    session of their own.
    """
    if payload.type.startswith(event_publish.GLOBAL_EVENT_PREFIXES):
        # Project lifecycle / combine events are the API's own, on the global stream; a client
        # may not spoof them on a project stream.
        raise HTTPException(
            status_code=422, detail=f'{payload.type!r} is a global event type; not publishable here'
        )
    status_key = region_wire_key('status')
    event: dict[str, Any] = {
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
    if payload.region_count is not None:
        event['region_count'] = payload.region_count
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
    extra.pop('type', None)
    event.update(extra)
    await event_publish.apublish(payload.type, event, project=bound)
    return {'ok': True}


@router.get('/events/stats')
async def curation_events_stats() -> dict[str, Any]:
    """Ops counters for the bound project's stream: its local subscribers, the clients this
    process dropped for falling behind, and whether the listener and its Valkey subscription run."""
    return get_listener().stats(get_curation_config().project_slug)
