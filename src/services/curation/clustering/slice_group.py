"""Run the slice readers of a point-in-time scan together, and stop them all on a failure.

``asyncio.gather`` leaves the sibling tasks running when one raises, so they keep searching
against a point-in-time view that is about to be deleted. ``asyncio.TaskGroup`` cancels the
siblings on the first failure (Python docs, "Task Groups"). It raises an ``ExceptionGroup``;
callers of the scans catch specific exceptions (a cancelled job, an OpenSearch error), so the
first child exception is re-raised unchanged at this boundary.

This is clustering code on a worker, not a request handler: the fan-out is the point of the scan.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence


async def run_slices[T](coros: Sequence[Coroutine[Any, Any, T]]) -> list[T]:
    """Results of ``coros`` in order; the first failure cancels the rest and is re-raised."""
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(c) for c in coros]
    except BaseExceptionGroup as failure:
        raise failure.exceptions[0] from failure
    return [t.result() for t in tasks]
