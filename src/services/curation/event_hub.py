"""Cross-process event hub for live labeler UI updates.

The curation `/review` and `/clusters` pages used to require a manual
refresh to pick up newly ingested or classified crops. This module
provides a small fan-out mechanism: producers (ingest, VLM label batch,
sam-worker via HTTP publish endpoint) call :func:`publish` to enqueue an
event; SSE subscribers in :mod:`src.routers.curation` pull from their
own per-subscriber queue.

Design (S-3, file-backed bus):

- ``yolo-api`` runs 8 uvicorn *worker processes*; an in-process-only hub
  (the original design) means an SSE client connected to worker 3 never
  sees an event published by worker 7. The bus fixes that with one
  shared append-only JSONL log per state dir
  (``{CurationConfig.state_dir}/events/events.jsonl``): every process —
  every uvicorn worker, every curation background worker — tails the
  same file and fans events out to its own local subscribers.
- ``publish()`` on the ``file`` bus (``OP_EVENT_BUS``, default) only
  appends to the log; it never dispatches directly. Local delivery
  happens exclusively in that same process's tail loop (started via
  :meth:`EventHub.start_tail` from the FastAPI lifespan), so the
  publishing process doesn't deliver an event twice.
- ``OP_EVENT_BUS=process`` restores the pre-S-3 in-process-only
  behavior (single-worker deployments, tests).
- The log is bounded: past ``OP_EVENT_LOG_MAX_BYTES`` (default 8 MiB) a
  write rotates the file to a single ``.1`` backup, so the log never
  exceeds roughly 2x that bound. Lines over 4096 bytes are refused and
  logged rather than written.
- A worker process that isn't the API (e.g. ``scripts/curation/worker``)
  has no FastAPI lifespan to run a tailer in — it publishes over HTTP
  via ``POST {api_prefix}/events/publish`` instead (see
  ``scripts/curation/worker/bulk_writer.py``); the API process that
  receives that POST appends to the shared log like any other
  publisher.
- Subscribers each get their own bounded queue (cap 1000). On overflow
  the oldest event is dropped — these events are advisory, not a
  durable change feed.
- Optional ``topic`` filter (e.g. the region-status topic) and
  ``class_id`` filter applied at fan-out time so a subscriber on the
  ``/clusters/42`` page only sees crops labeled with class 42.
- Ops counters (``subscribers``, ``events_published``, ``events_dropped``)
  plus ``bus``/``log_path`` surfaced via ``/curation/events/stats``.

Event payload schema (advisory — frontend code keys on ``type``):

- ``crop.created``: ``{type, crop_id, image_path, ts}``
- ``crop.classified``: ``{type, crop_id, class_id, class_name,
  class_source, ts}``
- ``crop.region_verified``: ``{type, crop_id, region_status, region_text?,
  ts}``
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import time
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.wire import region_event_payload


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path


logger = get_logger(__name__)


# Per-subscriber queue cap. Events are advisory — drop oldest on overflow
# so a slow client never causes back-pressure on the publisher.
_SUBSCRIBER_QUEUE_MAX = 1000

# A single JSONL line over this size is refused rather than written —
# guards against an accidental huge payload wedging the shared log.
_MAX_LINE_BYTES = 4096

_DEFAULT_MAX_LOG_BYTES = 8 * 1024 * 1024  # 8 MiB

# How often the tailer polls for new lines / rotation. Cheap: a stat
# plus a read on a small local file, a few syscalls per second per
# process.
_TAIL_POLL_S = 0.25

# Bounded retry count for the open/flock/verify-inode race against a
# concurrent rotation (see _EventLog._append_sync).
_APPEND_RETRIES = 5


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


class _EventLog:
    """Shared append-only JSONL log, tailed by every process sharing it.

    Not a message queue: there's no ack, no per-consumer offset. Every
    tailer starts at EOF when it attaches and sees everything appended
    from then on — exactly the semantics the old in-process queue had,
    just fanned out across processes instead of across
    ``asyncio.Queue`` instances in one.
    """

    def __init__(self, path: Path | None = None, max_bytes: int | None = None) -> None:
        cfg = get_curation_config()
        self.path = path or (cfg.state_dir / 'events' / 'events.jsonl')
        if max_bytes is not None:
            self.max_bytes = max_bytes
        else:
            self.max_bytes = int(
                os.environ.get('OP_EVENT_LOG_MAX_BYTES', str(_DEFAULT_MAX_LOG_BYTES))
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Touch now so a startup-time permissions problem surfaces as a
        # clear log line (caught by the caller) instead of a silent
        # first-publish failure.
        with self.path.open('a'):
            pass
        self._pending_writes: set[asyncio.Task[None]] = set()

    # -- write -------------------------------------------------------

    def append_nowait(self, event: dict[str, Any]) -> None:
        """Schedule ``event`` to be appended without blocking the caller.

        ``publish()`` is called from both async and (rarely) sync
        contexts and must never itself block on disk I/O — when a loop
        is running the actual write happens on a thread
        (:func:`asyncio.to_thread`); with no running loop (a script, a
        unit test) it runs inline, since there's no event loop to
        protect.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._append_sync(event)
            return
        task = loop.create_task(asyncio.to_thread(self._append_sync, event))
        self._pending_writes.add(task)
        task.add_done_callback(self._pending_writes.discard)

    def _append_sync(self, event: dict[str, Any]) -> None:
        try:
            line = json.dumps(event, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning('event_log_encode_failed', error=str(exc))
            return
        data = (line + '\n').encode('utf-8')
        if len(data) > _MAX_LINE_BYTES:
            logger.warning('event_log_line_dropped_too_large', size=len(data))
            return

        for _attempt in range(_APPEND_RETRIES):
            try:
                fd = os.open(str(self.path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            except OSError as exc:
                logger.warning('event_log_open_failed', error=str(exc))
                return
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    # A concurrent writer may have rotated ``self.path``
                    # to ``.1`` between our open() and this flock() —
                    # our fd would then be locked on the *old* (now
                    # renamed-away) inode. Detect that and retry against
                    # the fresh path rather than silently writing into
                    # the rotated-out backup.
                    current_ino: int | None
                    try:
                        current_ino = self.path.stat().st_ino
                        fd_ino = os.fstat(fd).st_ino
                    except OSError:
                        current_ino = None
                        fd_ino = None
                    if current_ino is None or current_ino != fd_ino:
                        continue
                    os.write(fd, data)
                    if os.fstat(fd).st_size > self.max_bytes:
                        rotated = self.path.with_name(self.path.name + '.1')
                        self.path.replace(rotated)
                    return
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        logger.warning('event_log_append_gave_up', path=str(self.path))

    # -- tail ----------------------------------------------------------

    async def tail(self, on_event: Callable[[dict[str, Any]], None]) -> None:
        """Long-lived task: deliver every line appended from now on.

        Starts at EOF (never replays history — matches the old
        in-process queue's semantics for a subscriber that just
        connected). Detects rotation (the path's inode changes under
        us) and truncation, finishing whatever the old file descriptor
        still has buffered before switching over.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a'):
            pass
        f = self.path.open('rb')
        f.seek(0, os.SEEK_END)
        try:
            ino = os.fstat(f.fileno()).st_ino
        except OSError:
            ino = None
        buf = b''
        try:
            while True:
                chunk = f.read()
                if chunk:
                    buf += chunk
                    *complete, buf = buf.split(b'\n')
                    for raw in complete:
                        if not raw:
                            continue
                        try:
                            on_event(json.loads(raw))
                        except json.JSONDecodeError as exc:
                            logger.warning('event_log_tail_bad_line', error=str(exc))
                    continue

                try:
                    st = self.path.stat()
                except FileNotFoundError:
                    st = None

                if st is not None and ino is not None and st.st_ino != ino:
                    # Rotated. Every byte written before the rename is
                    # already drained (the fd above still pointed at
                    # the old inode and `read()` returns everything
                    # available before we ever look at `stat()`), so
                    # it's safe to just reopen at the new inode.
                    f.close()
                    f = self.path.open('rb')
                    try:
                        ino = os.fstat(f.fileno()).st_ino
                    except OSError:
                        ino = None
                    buf = b''
                    await asyncio.sleep(0)
                    continue

                if st is not None and st.st_size < f.tell():
                    # Truncated in place — not our own rotation scheme,
                    # but handle it rather than looping on read errors.
                    f.seek(0)
                    buf = b''

                await asyncio.sleep(_TAIL_POLL_S)
        finally:
            f.close()


class EventHub:
    """Fan-out hub for advisory crop-state events.

    One instance per process (see :func:`get_event_hub`). On the
    ``file`` bus every instance shares the same on-disk log via its own
    :class:`_EventLog`.
    """

    def __init__(self) -> None:
        self._subscribers: set[_Subscriber] = set()
        self._lock = asyncio.Lock()
        self._published = 0
        self._dropped = 0
        self._bus = os.environ.get('OP_EVENT_BUS', 'file')
        self._log: _EventLog | None = None
        self._tail_task: asyncio.Task[None] | None = None
        if self._bus == 'file':
            try:
                self._log = _EventLog()
            except OSError as exc:
                logger.warning('event_log_unavailable_falling_back_to_process_bus', error=str(exc))
                self._bus = 'process'

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
        """Publisher entry point — safe from any running coroutine.

        Adds a server timestamp if the caller didn't include one. On
        the ``file`` bus this only appends to the shared log; local
        delivery happens in this same process's tail loop (started via
        :meth:`start_tail`), never here directly, so a process never
        delivers its own publish twice.
        """
        if 'ts' not in event:
            event['ts'] = time.time()
        self._published += 1
        if self._bus == 'file' and self._log is not None:
            self._log.append_nowait(event)
            return
        self._dispatch(event)

    def _dispatch(self, event: dict[str, Any]) -> None:
        """Fan out ``event`` to every matching local subscriber.

        No lock needed for the iteration: ``set`` mutation during
        iteration is the only risk and Python's GC + the small
        subscriber count make a snapshot copy effectively free.
        """
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

    # -- cross-process tail ----------------------------------------------

    async def start_tail(self) -> None:
        """Start this process's tail task on the shared event log.

        No-op on the ``process`` bus (nothing to tail) or if already
        running. Call once from the FastAPI lifespan; every uvicorn
        worker process gets its own tailer and its own subscriber set.
        """
        if self._bus != 'file' or self._log is None or self._tail_task is not None:
            return
        self._tail_task = asyncio.create_task(self._log.tail(self._dispatch))

    async def stop_tail(self) -> None:
        if self._tail_task is None:
            return
        self._tail_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._tail_task
        self._tail_task = None

    # -- stream ----------------------------------------------------------

    async def stream(self, sub: _Subscriber) -> AsyncIterator[dict[str, Any]]:
        """Yield events for one subscriber until cancelled."""
        while True:
            ev = await sub.queue.get()
            yield ev

    # -- ops -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            'subscribers': len(self._subscribers),
            'events_published': self._published,
            'events_dropped': self._dropped,
            'bus': self._bus,
            'log_path': str(self._log.path) if self._log is not None else None,
        }


# Module-level singleton. Importers call ``get_event_hub()``; tests may
# reset by re-creating the global. Each process gets its own instance —
# on the ``file`` bus they all share the same on-disk log, which is what
# makes an event published by one uvicorn worker visible to a subscriber
# connected to another. A process without a FastAPI lifespan to run
# ``start_tail()`` in (e.g. ``scripts/curation/worker``) publishes over
# HTTP via ``POST {api_prefix}/events/publish`` instead.
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
