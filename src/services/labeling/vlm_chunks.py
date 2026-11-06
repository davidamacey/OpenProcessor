"""Bounded fan-out of the chunks of one batched VLM call.

A batch of up to 64 crops is split into chunks of ``max_images_per_call``; with a cap of 1 that
is 64 upstream calls at once. :func:`run_chunks` runs them under a semaphore inside a
``TaskGroup`` (Python docs, "Task Groups": a failing chunk cancels its siblings, and no task is
left running), and returns the results in chunk order.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

VLM_CHUNK_CONCURRENCY = 8
"""Most chunk calls of one batch in flight at once; the labeler's token bucket still sets the rate."""


async def run_chunks[T](
    calls: Sequence[Awaitable[T]], *, limit: int = VLM_CHUNK_CONCURRENCY
) -> list[T]:
    """Await ``calls`` with at most ``limit`` running, keeping their order."""
    gate = asyncio.Semaphore(limit)

    async def one(call: Awaitable[T]) -> T:
        async with gate:
            return await call

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(c)) for c in calls]
    return [t.result() for t in tasks]
