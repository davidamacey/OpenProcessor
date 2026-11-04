"""Make a worker pipeline task fail loudly.

A task that dies with an exception while nobody awaits it leaves the worker reporting a passing
healthcheck and doing nothing. Every stage task of a worker is spawned through
:func:`spawn_critical`: an unhandled exception is logged and the process exits non-zero, so the
container restart policy recovers it.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence


def crash_on_unhandled_exception(task: asyncio.Task[Any]) -> None:
    """Done-callback: any exception other than cancellation is fatal (log, then exit 1)."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    logging.getLogger('curation-worker').critical(
        'fatal: task %s exited with an unhandled exception', task.get_name(), exc_info=exc
    )
    os._exit(1)


def spawn_critical(coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task[Any]:
    """``asyncio.create_task`` plus :func:`crash_on_unhandled_exception`."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(crash_on_unhandled_exception)
    return task


def log_drain_results(stage: str, results: Sequence[object]) -> None:
    """Log the exceptions a ``gather(..., return_exceptions=True)`` drain collected."""
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            logging.getLogger('curation-worker').error(
                'worker stage %s ended with an exception', stage, exc_info=result
            )
