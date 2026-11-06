"""Infrastructure tasks: the no-op jobs the live checks run through the whole job path."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext


MIN_SLEEP_S = 1
MAX_SLEEP_S = 600


async def noop(ctx: JobContext) -> dict[str, Any]:
    """Does nothing: proves a started job is claimed, run and finished through the wrapper."""
    await ctx.progress(1, 1, 'noop')
    return {'noop': True}


async def sleep_noop(ctx: JobContext) -> dict[str, Any]:
    """Sleeps ``params['seconds']`` (1..600), one progress snapshot a second; stops on cancel.

    Live checks use it as a job that is still running when a broker, database or worker is hurt.
    """
    seconds = ctx.params.get('seconds')
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int)
        or not MIN_SLEEP_S <= seconds <= MAX_SLEEP_S
    ):
        raise ValueError(f'seconds must be an integer from {MIN_SLEEP_S} to {MAX_SLEEP_S}')
    slept = 0
    while slept < seconds and not ctx.cancelled():
        await asyncio.sleep(1)
        slept += 1
        await ctx.progress(slept, seconds, 'sleeping')
    return {'slept': slept, 'cancelled': slept < seconds}


register_job('noop', noop)
register_job('sleep_noop', sleep_noop)
