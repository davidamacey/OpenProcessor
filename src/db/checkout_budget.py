"""A request's budget of simultaneous Postgres connections.

The API pool has no overflow and a 5 s wait that becomes a 503. A request that fans out over
many tasks (one ingest batch runs up to 16 images at once, each with its own short
transactions) would otherwise ask for one connection per task and starve every other request
of the process. ``checkout_budget(n)`` gives the tasks started under it ``n`` connections to
share; ``db_slot()`` wraps one session's whole life (checkout to release) and waits for a
free slot instead of for the pool. Outside a budget ``db_slot()`` does nothing.

The budget is a semaphore of one request's own event loop, not shared state: it bounds one
request's use of the pool, and the pool itself stays the cross-process limit. SQLAlchemy
asyncio guidance: keep a session's scope short and give each task its own session
(https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#using-asyncsession-with-concurrent-tasks).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

_BUDGET: ContextVar[asyncio.Semaphore | None] = ContextVar('db_checkout_budget', default=None)


@contextmanager
def checkout_budget(connections: int) -> Iterator[None]:
    """Tasks created inside share ``connections`` simultaneous ``db_slot()`` holders."""
    if connections < 1:
        msg = 'a checkout budget needs at least one connection'
        raise ValueError(msg)
    token = _BUDGET.set(asyncio.Semaphore(connections))
    try:
        yield
    finally:
        _BUDGET.reset(token)


@asynccontextmanager
async def db_slot() -> AsyncIterator[None]:
    """Hold one slot of the surrounding budget for the body (a no-op without a budget)."""
    budget = _BUDGET.get()
    if budget is None:
        yield
        return
    async with budget:
        yield
