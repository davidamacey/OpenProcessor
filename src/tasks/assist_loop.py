"""Tasks of the assist loop: ``op.assist_loop.run``, one slice of a loop per delivery.

A delivery loads the run's tree from its tables, forfeits what a dead slice left reserved, and
works nodes until the queue is empty (the job completes), a cancel or a pause is seen at a
node boundary, or the kind's ``slice_s`` has passed with work left (the job yields and is
dispatched again behind other projects' slices). The loop's own pauses (a failing audit, a
reached budget, a VLM that does not answer) are requested through the job with their reason.
The OpenSearch client is injected here, never imported by ``src/tasks/runtime/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.db.repos import assist_loop_record as rows
from src.services.curation.assist_loop import driver, runtime
from src.services.curation.assist_loop.budget import LoopBudget
from src.services.curation.assist_loop.node_step import RunControl
from src.services.curation.assist_loop.record import LoopRecord
from src.services.curation.assist_loop.snapshots import SnapshotStore
from src.services.curation.assist_loop.vlm_failure import VlmFailedError
from src.services.curation.job_binding import bound_job_project, opensearch_client, project_record
from src.services.jobs.blockers import (
    PROJECT_PAUSED,
    VLM_UNAVAILABLE,
    project_is_paused,
    vlm_available,
)
from src.tasks.runtime.context import Paused, Yield
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from src.tasks.runtime.context import JobContext

logger = get_logger(__name__)
_FACTORIES = {'opensearch': opensearch_client}


def _system_pause(session: Session, ctx_project: Any) -> str | None:
    if ctx_project is not None and project_is_paused(session, ctx_project):
        return PROJECT_PAUSED
    if not vlm_available(session):
        return VLM_UNAVAILABLE
    return None


async def _blocked(ctx: JobContext) -> str | None:
    """Why this slice may not start: the project was paused, or the VLM was taken."""
    async with ctx.session() as session:
        return await session.run_sync(lambda s: _system_pause(s, ctx.project_id))


async def assist_loop(ctx: JobContext) -> dict[str, Any] | Paused | Yield:
    """One slice of the loop of the job's project."""
    project = await project_record(ctx)
    with bound_job_project(ctx, project):
        opensearch = await ctx.client('opensearch')
        async with ctx.session() as session:
            run = await session.run_sync(lambda s: rows.run_by_job(s, ctx.job_id))
        if run is None:
            raise RuntimeError(f'job {ctx.job_id} has no assist loop run')
        if (reason := await _blocked(ctx)) is not None:
            await ctx.request_pause(reason)
            return Paused(reason)
        record = LoopRecord(
            ctx.session,
            run,
            token=ctx.fencing_token,
            snapshots=SnapshotStore(project.slug, run.id),
        )
        budget = LoopBudget(record)
        forfeited = await budget.start_slice()
        if forfeited:
            logger.info('assist_loop_forfeited', loop=record.loop_id, reservations=forfeited)
        built = await runtime.build(opensearch, record, budget)
        control = RunControl(
            cancelled=ctx.cancelled,
            paused=ctx.paused,
            slice_due=ctx.slice_due,
            progress=ctx.progress,
        )
        loop_run = await driver.prepare(
            record, built.params, built.deps, budget, built.scope, control
        )
        budget.usage_missing = record.tree.usage_missing
        try:
            outcome = await driver.schedule(loop_run)
        except VlmFailedError as exc:
            logger.error('assist_loop_vlm_failed', loop=record.loop_id, error=str(exc))
            outcome = driver.LoopOutcome('paused', pause_reason=VLM_UNAVAILABLE)
        await driver.finish(loop_run, outcome)
    if outcome.status == 'cancelled':
        return {}
    if outcome.status == 'yield':
        return Yield()
    if outcome.status == 'paused':
        if outcome.pause_reason is None:
            return Paused()
        await ctx.request_pause(outcome.pause_reason)
        return Paused(outcome.pause_reason)
    return {
        'stop_reason': outcome.stop_reason,
        'loop_id': record.loop_id,
    }


register_job('assist_loop', assist_loop, client_factories=_FACTORIES)
