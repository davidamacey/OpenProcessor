"""Run sibling coroutines together so a failure leaves none of them running."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any


async def gather_all[T](coros: list[Coroutine[Any, Any, T]]) -> list[T]:
    """The results of ``coros`` in order. When one raises, the others are cancelled and
    awaited before that exception propagates (``asyncio.gather`` would leave them running,
    spending budget for a loop that has already failed). Several failures raise the first
    in argument order; the caller sees the exception itself, not a group."""
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(coro) for coro in coros]
    except BaseExceptionGroup as failures:
        raise failures.exceptions[0] from None
    return [task.result() for task in tasks]


__all__ = ['gather_all']
