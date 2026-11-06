"""The resumable event stream behind the SSE routes, free of any web framework.

The routes (``src/routers/curation/_event_streams.py``) turn each :class:`Frame` into a
``fastapi.sse.ServerSentEvent`` with ``id="<xid>-<id>"``; the browser re-sends the last id as
``Last-Event-ID`` on reconnect
(https://fastapi.tiangolo.com/tutorial/server-sent-events/#resuming-with-last-event-id).
A stream holds no database connection between queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.events.cursor import START, Cursor, InvalidCursorError
from src.services.events.listener import Overflow, Subscription, get_listener


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.db.repos.events import EventRow


@dataclass(frozen=True)
class Frame:
    event: str
    data: dict[str, Any]
    id: str | None = None


async def open_subscription(slug: str | None) -> AsyncIterator[Subscription]:
    """Join a project's stream (``None``: the global one) for the life of the request."""
    listener = get_listener()
    sub = await listener.subscribe(slug)
    try:
        yield sub
    finally:
        await listener.unsubscribe(sub)


def wire_event(slug: str | None, row: EventRow) -> dict[str, Any]:
    """The event as the browser has always seen it: payload fields plus ``type`` and ``project``."""
    return {**row.payload, 'type': row.type, 'project': slug}


def _wanted(event: dict[str, Any], topic: str | None, class_id: int | None) -> bool:
    if topic is not None and event.get('topic') not in (None, topic):
        return False
    if class_id is not None:
        cid = event.get('class_id')
        # Events without a class_id are not about one class; a class-filtered client skips them.
        if cid is None or int(cid) != class_id:
            return False
    return True


def _frame(
    sub: Subscription, row: EventRow, topic: str | None, class_id: int | None
) -> Frame | None:
    data = wire_event(sub.slug, row)
    if not _wanted(data, topic, class_id):
        return None
    return Frame(id=Cursor(row.xid, row.id).format(), event=row.type, data=data)


def _reset(reason: str, head: Cursor) -> Frame:
    return Frame(
        event='reset', data={'reason': reason}, id=head.format() if head != START else None
    )


async def event_stream(
    sub: Subscription,
    last_event_id: str | None,
    *,
    topic: str | None = None,
    class_id: int | None = None,
) -> AsyncIterator[Frame]:
    """Replay from ``last_event_id`` up to the cursor the subscription joined at, then go live."""
    listener = get_listener()
    position = sub.start
    if last_event_id:
        try:
            resume = Cursor.parse(last_event_id)
            reason = await listener.reset_reason(sub.project_id, resume)
        except InvalidCursorError:
            resume, reason = None, 'cursor_invalid'
        if reason is not None:
            yield _reset(reason, sub.start)
        elif resume is not None:
            while resume < sub.start:
                rows = [
                    r
                    for r in await listener.read_after(sub.project_id, resume)
                    if Cursor(r.xid, r.id) <= sub.start
                ]
                if not rows:
                    break
                for row in rows:
                    resume = Cursor(row.xid, row.id)
                    if (frame := _frame(sub, row, topic, class_id)) is not None:
                        yield frame
            position = max(resume, sub.start)
    while True:
        item = await sub.queue.get()
        if isinstance(item, Overflow):
            yield Frame(event='reconnect', data={'reason': 'slow_client'})
            return
        if Cursor(item.xid, item.id) <= position:
            continue
        if (frame := _frame(sub, item, topic, class_id)) is not None:
            yield frame
