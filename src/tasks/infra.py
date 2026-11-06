"""Infrastructure tasks: the no-op job the live queue check runs through the whole job path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext


async def noop(ctx: JobContext) -> dict[str, Any]:
    """Does nothing: proves a started job is claimed, run and finished through the wrapper."""
    await ctx.progress(1, 1, 'noop')
    return {'noop': True}


register_job('noop', noop)
