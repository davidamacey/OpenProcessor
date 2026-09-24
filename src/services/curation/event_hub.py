"""In-process event hub for live labeler UI updates.

The curation `/review` and `/clusters` pages used to require a manual
refresh to pick up newly ingested or classified crops. This module
provides a small fan-out queue: producers (ingest, VLM label batch,
sam-worker via HTTP publish endpoint) call :func:`publish` to enqueue an
event; SSE subscribers in :mod:`src.routers.curation` pull from their
own per-subscriber queue.

Design constraints:

- No Redis / no broker. Pure :mod:`asyncio` queues, in-process.
- Subscribers each get their own bounded queue (cap 1000). On overflow
  the oldest event is dropped — these events are advisory, not a
  durable change feed.
- Optional ``topic`` filter (e.g. the region-status topic) and
  ``class_id`` filter applied at fan-out time so a subscriber on the
  ``/clusters/42`` page only sees crops labeled with class 42.
- Ops counters (``subscribers``, ``events_published``, ``events_dropped``)
  surfaced via ``/curation/events/stats``.

Event payload schema (advisory — frontend code keys on ``type``):

- ``crop.created``: ``{type, crop_id, image_path, ts}``
- ``crop.classified``: ``{type, crop_id, class_id, class_name,
  class_source, ts}``
- ``crop.region_verified``: ``{type, crop_id, region_status, region_text?,
  ts}``
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.curation.wire import region_event_payload


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = get_logger(__name__)


# Per-subscriber queue cap. Events are advisory — drop oldest on overflow
# so a slow client never causes back-pressure on the publisher.
_SUBSCRIBER_QUEUE_MAX = 1000


class _Subscriber:
    """One connected SSE client.

    Holds a bounded queue plus the topic / class_id filter the client
    asked for at subscription time. Filtering happens on the publish
    side so a subscriber's queue never accumulates events it would
    immediately discard.
    """

    __slots__ = ('class_id', 'queue', 'topic')

    def __init__(self, *, topic: str | None, class_id: int | None) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        self.topic = topic
        self.class_id = class_id

    def matches(self, event: dict[str, Any]) -> bool:
        """Return True if this subscriber wants ``event``."""
        if self.topic is not None:
            ev_topic = event.get('topic')
            if ev_topic is not None and ev_topic != self.topic:
                return False
        if self.class_id is not None:
            cid = event.get('class_id')
            # Events without a class_id are global — a class-filtered
            # subscriber doesn't want them. ``crop.created`` falls in
            # this bucket and is intentionally hidden from per-class
            # cluster pages.
            if cid is None or int(cid) != int(self.class_id):
                return False
        return True


class EventHub:
    """Singleton fan-out hub for advisory crop-state events."""

    def __init__(self) -> None:
        self._subscribers: set[_Subscriber] = set()
        self._lock = asyncio.Lock()
        self._published = 0
        self._dropped = 0

    # -- subscribe / unsubscribe -----------------------------------------

    async def subscribe(
        self,
        *,
        topic: str | None = None,
        class_id: int | None = None,
    ) -> _Subscriber:
        sub = _Subscriber(topic=topic, class_id=class_id)
        async with self._lock:
            self._subscribers.add(sub)
        return sub

    async def unsubscribe(self, sub: _Subscriber) -> None:
        async with self._lock:
            self._subscribers.discard(sub)

    # -- publish ---------------------------------------------------------

    def publish(self, event: dict[str, Any]) -> None:
        """Synchronous publisher — safe from any running coroutine.

        Adds a server timestamp if the caller didn't include one. Drops
        oldest event from any subscriber whose queue is full so back-
        pressure never propagates to the writer.
        """
        if 'ts' not in event:
            event['ts'] = time.time()
        self._published += 1
        # No lock needed for the iteration: ``set`` mutation during
        # iteration is the only risk and Python's GC + the small
        # subscriber count make a snapshot copy effectively free.
        for sub in list(self._subscribers):
            if not sub.matches(event):
                continue
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest — events are advisory.
                try:
                    sub.queue.get_nowait()
                    self._dropped += 1
                    sub.queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    self._dropped += 1

    # -- stream ----------------------------------------------------------

    async def stream(self, sub: _Subscriber) -> AsyncIterator[dict[str, Any]]:
        """Yield events for one subscriber until cancelled."""
        while True:
            ev = await sub.queue.get()
            yield ev

    # -- ops -------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            'subscribers': len(self._subscribers),
            'events_published': self._published,
            'events_dropped': self._dropped,
        }


# Module-level singleton. Importers call ``get_event_hub()``; tests may
# reset by re-creating the global. The hub is process-local — for the
# sam-worker (separate process) we expose ``POST /curation/events/publish``.
_HUB: EventHub | None = None


def get_event_hub() -> EventHub:
    global _HUB  # noqa: PLW0603 — module singleton
    if _HUB is None:
        _HUB = EventHub()
    return _HUB


def publish_crop_created(crop_id: str, image_path: str = '') -> None:
    """Convenience wrapper for the ingest write path."""
    get_event_hub().publish(
        {
            'type': 'crop.created',
            'topic': 'crop',
            'crop_id': crop_id,
            'image_path': image_path,
        }
    )


def publish_crop_classified(
    crop_id: str,
    *,
    class_id: int | None,
    class_name: str | None = None,
    class_source: str = '',
) -> None:
    """Convenience wrapper for the VLM / ensemble write path."""
    get_event_hub().publish(
        {
            'type': 'crop.classified',
            'topic': 'crop',
            'crop_id': crop_id,
            'class_id': class_id,
            'class_name': class_name,
            'class_source': class_source,
        }
    )


def publish_region_verified(
    crop_id: str,
    *,
    region_status: str,
    region_text: str | None = None,
) -> None:
    """Convenience wrapper for sam-worker region updates. Payload keys are
    the fixed wire names, never the storage field names."""
    get_event_hub().publish(
        region_event_payload(crop_id, region_status=region_status, region_text=region_text)
    )
