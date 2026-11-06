"""The per-process event listener: one Valkey subscription, one catch-up query per wake-up.

Each API process holds one pub/sub connection (redis-py asyncio,
https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html) subscribed to
``op:events:global`` and to ``op:events:<project id>`` for every project that has a local SSE
client. A wake-up, or the 2 s safety tick, runs the committed-safe cursor query once per such
stream on a pooled connection that goes back at once, then fans the rows out to local
subscribers through bounded queues. A slow client is dropped with a final ``reconnect`` event
and never blocks the others. Valkey down: the tick keeps delivering (02b section 6.3).
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import redis.asyncio as aioredis
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from src.config.env import EnvError
from src.config.queue import QueueSettings
from src.core.logging import get_logger
from src.db.repos import events as repo
from src.db.session import async_session_factory, deployment_tx, project_tx
from src.services.events.cursor import Cursor
from src.tasks.runtime.event_wakeup import (
    CHANNEL_PREFIX,
    GLOBAL_CHANNEL,
    channel_for,
    client_kwargs,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio.client import PubSub

logger = get_logger(__name__)

SAFETY_TICK_S: Final = 2.0
QUEUE_MAX: Final = 1000
_BACKOFF_START_S: Final = 0.5
_BACKOFF_MAX_S: Final = 30.0
_POLL_S: Final = 1.0


class ListenerNotRunningError(RuntimeError):
    """A subscription was requested while the listener is stopped."""


class ProjectNotFoundError(LookupError):
    """A subscription named a project that does not exist."""


class Overflow:
    """Queue sentinel: the client fell behind and was dropped."""


OVERFLOW: Final = Overflow()


@dataclass(eq=False)
class _Stream:
    project_id: uuid.UUID | None
    slug: str | None
    cursor: Cursor
    subs: set[Subscription] = field(default_factory=set)

    @property
    def channel(self) -> str:
        return channel_for(self.project_id)


@dataclass(eq=False)
class Subscription:
    """One local SSE client. ``start`` is the stream cursor when it joined: every row after
    it arrives on ``queue``; rows up to it come from :meth:`EventListener.read_after`."""

    project_id: uuid.UUID | None
    slug: str | None
    start: Cursor
    queue: asyncio.Queue[repo.EventRow | Overflow]
    _stream: _Stream = field(repr=False)


def _default_valkey() -> aioredis.Redis:
    settings = QueueSettings.from_env()
    return aioredis.Redis.from_url(
        settings.url, health_check_interval=15, **client_kwargs(settings)
    )


class EventListener:
    def __init__(
        self,
        *,
        valkey_factory: Callable[[], Any] = _default_valkey,
        tick_s: float = SAFETY_TICK_S,
        queue_max: int = QUEUE_MAX,
    ) -> None:
        self._valkey_factory = valkey_factory
        self._tick_s = tick_s
        self._queue_max = queue_max
        self._streams: dict[uuid.UUID | None, _Stream] = {}
        self._dirty: set[uuid.UUID | None] = set()
        self._wake = asyncio.Event()
        self._create_lock = asyncio.Lock()
        self._tasks: list[asyncio.Task[None]] = []
        self.pubsub_connected = False
        self.dropped_clients = 0

    @property
    def running(self) -> bool:
        return bool(self._tasks)

    async def start(self) -> None:
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._catch_up_loop(), name='op-events-catch-up'),
            asyncio.create_task(self._pubsub_loop(), name='op-events-pubsub'),
        ]

    async def stop(self) -> None:
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for stream in self._streams.values():
            for sub in list(stream.subs):
                self._drop(stream, sub)
        self._streams.clear()
        self.pubsub_connected = False

    # -- database --------------------------------------------------------------------------

    def _scope(self, session: Any, project_id: uuid.UUID | None) -> Any:
        if project_id is None:
            return deployment_tx(session, 'global event stream')
        return project_tx(session, project_id)

    async def _project_id(self, slug: str) -> uuid.UUID:
        async with async_session_factory()() as session:
            found = await repo.project_id_for_slug(session, slug)
        if found is None:
            raise ProjectNotFoundError(slug)
        return found

    async def read_after(
        self, project_id: uuid.UUID | None, after: Cursor, *, limit: int = repo.PAGE_SIZE
    ) -> list[repo.EventRow]:
        async with async_session_factory()() as session, self._scope(session, project_id):
            return await repo.fetch_after(session, project_id, after.xid, after.id, limit=limit)

    async def _head(self, project_id: uuid.UUID | None) -> Cursor:
        async with async_session_factory()() as session, self._scope(session, project_id):
            return Cursor(*await repo.head_cursor(session, project_id))

    async def reset_reason(self, project_id: uuid.UUID | None, cursor: Cursor) -> str | None:
        """Why a resume from ``cursor`` cannot be trusted, or ``None`` when it can."""
        async with async_session_factory()() as session, self._scope(session, project_id):
            if await repo.cursor_from_future(session, cursor.xid):
                return 'cursor_unknown'
            oldest = await repo.oldest_cursor(session, project_id)
        if oldest is not None and cursor < Cursor(*oldest):
            return 'cursor_expired'
        return None

    # -- subscribe -------------------------------------------------------------------------

    async def subscribe(self, project: str | None) -> Subscription:
        """Join the stream of a project slug, or the global stream for ``None``."""
        if not self.running:
            raise ListenerNotRunningError
        project_id = None if project is None else await self._project_id(project)
        async with self._create_lock:
            stream = self._streams.get(project_id)
            if stream is None:
                stream = _Stream(project_id, project, await self._head(project_id))
                self._streams[project_id] = stream
            sub = Subscription(
                project_id, project, stream.cursor, asyncio.Queue(self._queue_max), stream
            )
            stream.subs.add(sub)
        return sub

    async def unsubscribe(self, sub: Subscription) -> None:
        stream = sub._stream
        stream.subs.discard(sub)
        if not stream.subs and self._streams.get(stream.project_id) is stream:
            del self._streams[stream.project_id]

    def _drop(self, stream: _Stream, sub: Subscription) -> None:
        stream.subs.discard(sub)
        while not sub.queue.empty():
            sub.queue.get_nowait()
        sub.queue.put_nowait(OVERFLOW)

    def _fan_out(self, stream: _Stream, row: repo.EventRow) -> None:
        for sub in list(stream.subs):
            try:
                sub.queue.put_nowait(row)
            except asyncio.QueueFull:
                self.dropped_clients += 1
                self._drop(stream, sub)

    def stats(self, project: str | None) -> dict[str, Any]:
        return {
            'subscribers': sum(len(s.subs) for s in self._streams.values() if s.slug == project),
            'clients_dropped': self.dropped_clients,
            'listener_running': self.running,
            'pubsub_connected': self.pubsub_connected,
        }

    # -- catch-up --------------------------------------------------------------------------

    async def _catch_up(self, stream: _Stream) -> None:
        while True:
            rows = await self.read_after(stream.project_id, stream.cursor)
            for row in rows:
                stream.cursor = Cursor(row.xid, row.id)
                self._fan_out(stream, row)
            if len(rows) < repo.PAGE_SIZE:
                return

    async def _catch_up_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), self._tick_s)
                keys = self._dirty & self._streams.keys()
            except TimeoutError:
                keys = set(self._streams)
            self._wake.clear()
            self._dirty.clear()
            for key in keys:
                stream = self._streams.get(key)
                if stream is None:
                    continue
                try:
                    await self._catch_up(stream)
                except (SQLAlchemyError, OSError) as exc:
                    logger.warning('event_catch_up_failed', error=str(exc))

    # -- Valkey ----------------------------------------------------------------------------

    def _on_message(self, channel: str) -> None:
        if channel == GLOBAL_CHANNEL:
            self._dirty.add(None)
        elif channel.startswith(CHANNEL_PREFIX):
            try:
                self._dirty.add(uuid.UUID(channel.removeprefix(CHANNEL_PREFIX)))
            except ValueError:
                return
        self._wake.set()

    async def _sync_channels(self, pubsub: PubSub, subscribed: set[str]) -> None:
        wanted = {GLOBAL_CHANNEL} | {s.channel for s in self._streams.values()}
        if added := wanted - subscribed:
            await pubsub.subscribe(*added)
        if removed := subscribed - wanted:
            await pubsub.unsubscribe(*removed)
        subscribed.clear()
        subscribed.update(wanted)

    async def _pubsub_loop(self) -> None:
        backoff = _BACKOFF_START_S
        while True:
            client = None
            try:
                client = self._valkey_factory()
                pubsub = client.pubsub()
                subscribed: set[str] = set()
                try:
                    await self._sync_channels(pubsub, subscribed)
                    self.pubsub_connected = True
                    backoff = _BACKOFF_START_S
                    # Rows committed while the connection was down have no wake-up.
                    self._dirty.update(self._streams)
                    self._wake.set()
                    while True:
                        await self._sync_channels(pubsub, subscribed)
                        message = await pubsub.get_message(
                            ignore_subscribe_messages=True, timeout=_POLL_S
                        )
                        if message is not None:
                            self._on_message(message['channel'])
                finally:
                    self.pubsub_connected = False
                    with contextlib.suppress(RedisError, OSError):
                        await pubsub.aclose()
            except (RedisError, EnvError, OSError) as exc:
                logger.warning('event_pubsub_down', error=str(exc), retry_in_s=backoff)
            finally:
                if client is not None:
                    with contextlib.suppress(RedisError, OSError):
                        await client.aclose()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _BACKOFF_MAX_S)


_listener: EventListener | None = None


def get_listener() -> EventListener:
    global _listener  # noqa: PLW0603  # one listener per process
    if _listener is None:
        _listener = EventListener()
    return _listener


async def start_listener() -> None:
    await get_listener().start()


async def stop_listener() -> None:
    global _listener  # noqa: PLW0603  # one listener per process
    if _listener is not None:
        await _listener.stop()
        _listener = None
