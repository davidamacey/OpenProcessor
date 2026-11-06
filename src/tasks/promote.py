"""The `promote` job: build the Triton model of a validated plan, load it and record it.

The handler rebuilds the plan the route validated from the job's parameters and runs the existing
:func:`~src.services.training.promote_run.execute_promote`; the stored rows come from
:class:`~src.services.training.promote_job.DbPromotionRecorder`. The heavy imports (httpx, the
ONNX tools) stay inside the handler so importing this module loads none of them.

The phases the promoter reports (``exporting``, ``loading``, ``building``, ``warming``) become
``progress.stage`` values; cancel is checked before the work starts (the build itself, a Triton
call of minutes, is not interruptible without leaving a half-loaded model).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext

REQUESTED_BY = 'job'


async def promote(ctx: JobContext) -> dict[str, Any]:
    from src.services.projects.registry import get_project_registry
    from src.services.training import promote_job
    from src.services.training.promote_run import execute_promote

    if ctx.project_id is None or ctx.project_slug is None:
        raise ValueError('a promote job belongs to a project')
    record = await get_project_registry().lookup(ctx.project_slug)
    if record is None:
        raise LookupError(f'project {ctx.project_slug!r} no longer exists')
    plan, payload = promote_job.plan_from_params(ctx.params)
    recorder = promote_job.DbPromotionRecorder(
        ctx.session,
        project_id=ctx.project_id,
        job_id=ctx.job_id,
        plan=plan,
        requested_by=REQUESTED_BY,
    )
    steps = len(promote_job.RUNNING_PHASES)
    if ctx.cancelled():
        return {'cancelled': True}
    # The promoter's phase callback is synchronous; each phase becomes a progress write, and
    # the references are kept until they are awaited (a bare create_task can be collected).
    writes: set[asyncio.Task[None]] = set()

    def on_phase(phase: str) -> None:
        if phase in promote_job.RUNNING_PHASES:
            write = asyncio.create_task(
                ctx.progress(promote_job.RUNNING_PHASES.index(phase), steps, phase)
            )
            writes.add(write)
            write.add_done_callback(writes.discard)

    try:
        with bind_project(record):
            response = await execute_promote(plan, payload, recorder, on_phase)
    finally:
        await asyncio.gather(*writes, return_exceptions=True)
    return response.model_dump()


register_job('promote', promote)
