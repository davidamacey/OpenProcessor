"""A GPU job holds its cards and waits for the arbiter before it touches them.

The train, bake-off and probe handlers reserve their cards (``gpu_reservations``) and then call
:func:`wait_for_arbiter`. The ``op.periodic.gpu_arbiter`` tick, the only process that holds the
Docker socket, stops the services the claim needs and records the claim as confirmed in
``deployment_settings('vlm_paused_for_training')``. The wait polls that record with a bounded
exponential backoff (the policy Celery documents for ``retry_backoff``,
https://docs.celeryq.dev/en/stable/userguide/tasks.html#Task.retry_backoff) and fails closed:
once the wait is spent it raises :class:`GpuBusyError` and the job ends ``failed`` with a
``gpu_busy`` message, never running next to a service it could not have stopped. A deployment
that names no GPU-resident service (``OP_GPU_ARBITER_CONTAINERS`` empty) has no arbiter and
nothing to wait for.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from src.config import get_gpu_arbiter_config
from src.config.env import env_float
from src.db.repos import gpu_arbiter as arbiter_repo, gpu_reservations as reservations
from src.domain import gpu_arbiter_state as state


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from sqlalchemy.orm import Session

    from src.db.repos.gpu_devices import Card
    from src.tasks.runtime.context import JobContext

WAIT_ENV = 'OP_GPU_CLAIM_WAIT_S'
DEFAULT_WAIT_S = 300.0
BACKOFF_START_S = 2.0
BACKOFF_CAP_S = 15.0


class GpuBusyError(RuntimeError):
    """The cards are held by another job or the arbiter did not confirm the claim in time."""

    code = 'gpu_busy'


async def _poll_until(
    ctx: JobContext,
    done: Callable[[], Awaitable[str | None]],
    *,
    sleep: Callable[[float], Awaitable[None]],
    clock: Callable[[], float],
) -> None:
    """Call ``done`` with a bounded backoff until it returns ``None`` (finished) or the wait is
    spent (:class:`GpuBusyError` with the reason it last returned). A cancel ends the wait."""
    deadline = clock() + env_float(WAIT_ENV, DEFAULT_WAIT_S, minimum=0.0)
    delay = BACKOFF_START_S
    while not ctx.cancelled():
        reason = await done()
        if reason is None:
            return
        remaining = deadline - clock()
        if remaining <= 0:
            msg = f'gpu_busy: {reason}'
            raise GpuBusyError(msg)
        await sleep(min(delay, remaining))
        delay = min(delay * 2, BACKOFF_CAP_S)


async def wait_for_arbiter(
    ctx: JobContext,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Return once the arbiter confirmed this job's claim, or at once on a cancel.

    ``sleep`` and ``clock`` are the seams the tests replace.
    """
    if not get_gpu_arbiter_config().containers:
        return
    job = str(ctx.job_id)

    async def confirmed() -> str | None:
        async with ctx.session() as session:
            record = state.arbiter_record(await session.run_sync(arbiter_repo.read_value))
        if job in record.acknowledged:
            return None
        detail = f' (last arbiter error: {record.error})' if record.error else ''
        return f'the GPU arbiter did not confirm the claim in time{detail}'

    await _poll_until(ctx, confirmed, sleep=sleep, clock=clock)


async def claim_cards(
    ctx: JobContext,
    find: Callable[[Session], Sequence[Card]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """Reserve the cards ``find`` returns, then wait for the arbiter; the UUIDs, ``[]`` when the
    deployment has no arbiter.

    A card another job holds is waited for with the same backoff and ends in
    :class:`GpuBusyError`; cards that are not recorded at all are a ``LookupError`` (the
    inventory has no row for them, so there is nothing to hold).
    """
    if not get_gpu_arbiter_config().containers:
        return []
    uuids: list[str] = []

    async def reserved() -> str | None:
        try:
            async with ctx.session() as session:
                cards = await session.run_sync(find)
                if not cards:
                    msg = 'no GPU card is recorded for this claim; start the GPU worker once first'
                    raise LookupError(msg)
                await session.run_sync(lambda s: reservations.reserve(s, ctx.job_id, cards))
        except reservations.CardHeldError as exc:  # the transaction rolled back: all or none
            return str(exc)
        uuids[:] = [c.gpu_uuid for c in cards]
        return None

    await _poll_until(ctx, reserved, sleep=sleep, clock=clock)
    await wait_for_arbiter(ctx, sleep=sleep, clock=clock)
    return uuids


async def release_cards(ctx: JobContext) -> None:
    """Free every card the job holds; the arbiter starts the services after the next pass."""
    if not get_gpu_arbiter_config().containers:
        return
    async with ctx.session() as session:
        await session.run_sync(lambda s: reservations.release(s, ctx.job_id))
