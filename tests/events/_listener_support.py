"""Helpers for tests that run a real EventListener against the test database and a fake Valkey."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from _events_support import FakeAsyncRedis

from src.services.events.listener import EventListener


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from _events_support import FakeBroker

    from src.db.repos.events import EventRow
    from src.services.events.listener import Overflow, Subscription


@asynccontextmanager
async def running_listener(broker: FakeBroker, **kwargs) -> AsyncIterator[EventListener]:
    listener = EventListener(valkey_factory=lambda: FakeAsyncRedis(broker), **kwargs)
    await listener.start()
    try:
        yield listener
    finally:
        await listener.stop()


async def take(sub: Subscription, count: int, *, timeout: float = 6.0) -> list[EventRow | Overflow]:
    items: list[EventRow | Overflow] = []
    async with asyncio.timeout(timeout):
        while len(items) < count:
            items.append(await sub.queue.get())
    return items


async def assert_quiet(sub: Subscription, *, wait: float = 0.5) -> None:
    await asyncio.sleep(wait)
    assert sub.queue.empty(), f'unexpected events: {list(sub.queue._queue)}'
