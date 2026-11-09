"""Run a list of coroutines together in an ``asyncio.TaskGroup`` and get the results in order.

Python docs, "Task Groups": when a child fails the group cancels the remaining children and
awaits them, so nothing is left running; ``asyncio.gather`` leaves its siblings running. A
failure arrives as an ``ExceptionGroup``; the callers here catch specific exception types, so
:func:`run_all` re-raises the first child's exception unchanged at this boundary.

Request handlers do not fan out. The ingest data plane (the four modules named in
``docs/design/engineering_standards.md``) and worker job bodies are the only callers.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence


async def run_all[T](coros: Sequence[Coroutine[Any, Any, T]]) -> list[T]:
    """Results of ``coros`` in order; the first failure cancels the rest and is re-raised."""
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(c) for c in coros]
    except BaseExceptionGroup as failure:
        raise failure.exceptions[0] from failure
    return [t.result() for t in tasks]


async def run_settled[T](coros: Sequence[Coroutine[Any, Any, T]]) -> list[T | Exception]:
    """Results of ``coros`` in order, a failed one as its ``Exception``; every child runs to its end.

    For per-item steps whose failure is recorded, not raised: a ``TaskGroup`` would cancel the
    siblings on the first failure, which is what these steps must not do. Nothing here cancels a
    sibling, so ``gather(return_exceptions=True)`` leaves none running. A child that ends by
    cancellation (not an ``Exception``) is raised, as it would be from ``await`` itself.
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result
    return cast('list[T | Exception]', results)
