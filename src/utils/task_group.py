"""Run a list of coroutines together in an ``asyncio.TaskGroup`` and get the results in order.

Python docs, "Task Groups": when a child fails the group cancels the remaining children and
awaits them, so nothing is left running; ``asyncio.gather`` leaves its siblings running. A
failure arrives as an ``ExceptionGroup``; the callers here catch specific exception types, so
:func:`run_all` re-raises the first child's exception unchanged at this boundary.

Request handlers do not fan out. The ingest data plane (the four modules named in
``docs/design/engineering_standards.md``) and worker job bodies are the only callers.

:func:`finish_despite_cancel` is the one other primitive: a step that must end (a stored photo's
release and events) even when the task that awaits it is cancelled, once or again.
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


async def finish_despite_cancel[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to its end although the awaiting task is cancelled, once or repeatedly, then
    re-raise the cancellation (its result or exception otherwise).

    Python docs, ``asyncio.shield``: it protects the inner task, and "if the coroutine containing
    ``shield()`` is cancelled, the Task running in ``something()`` is not cancelled"; the docs also
    require keeping a reference to the task, which this frame does until it is done. One ``shield``
    only absorbs the first cancellation: a second one (a ``TaskGroup`` cancelling its parent
    again) would interrupt the await, so the wait is repeated until the task is done. A
    cancellation of the inner task itself (the loop shutting down) is not absorbed.
    """
    task = asyncio.ensure_future(coro)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return task.result()
