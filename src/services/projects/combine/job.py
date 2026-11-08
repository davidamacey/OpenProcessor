"""The ``combine`` job as a worker runs it: load the run, bind the projects, call the body.

The job's project is the combine target, so the handler is already in the target's scope; the
sources are looked up by the slugs the run recorded. The near-duplicate pending rows of every
copied image and item are written through the job's own session before the documents they
describe (:mod:`src.services.near_duplicates.pending`, signature fixed by the near-duplicate plan).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from src.db.models.datasets import CombineRun
from src.services.projects import lifecycle
from src.services.projects.combine.execute import load_plan, run_combine
from src.services.projects.registry import get_project_registry


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.config.projects import ProjectRecord
    from src.tasks.runtime.context import JobContext


class CombineJobError(RuntimeError):
    """The run cannot start: a project it names is gone (fail closed, nothing is written)."""


async def _lookup(slug: str) -> ProjectRecord:
    record = await get_project_registry().lookup(slug)
    if record is None:
        raise CombineJobError(f"project '{slug}' is not registered")
    return record


def _settle_for(client: Any, target: ProjectRecord) -> Callable[[bool], Awaitable[None]]:
    async def settle(ok: bool) -> None:
        await lifecycle.finish_building(client, target, ok=ok)
        if ok:
            from src.services.events.publish import apublish_global_event

            await apublish_global_event(
                'project.created', target=target.slug, status='active', revision=target.revision
            )

    return settle


def _enqueue_for(ctx: JobContext) -> Callable[[str, list[str]], Awaitable[None]]:
    async def enqueue(level: str, member_ids: list[str]) -> None:
        from src.services.near_duplicates.pending import enqueue_pending

        if ctx.project_id is None:
            raise CombineJobError(f'job {ctx.job_id} is not bound to its target project')
        async with ctx.session() as session:
            await enqueue_pending(session, ctx.project_id, level, member_ids, 'added')

    return enqueue


async def run_combine_job(ctx: JobContext, client: Any) -> dict[str, Any] | None:
    async with ctx.session() as session:
        document = (
            await session.execute(
                select(CombineRun.class_mapping).where(CombineRun.job_id == ctx.job_id)
            )
        ).scalar_one_or_none()
    if document is None:
        raise CombineJobError(f'job {ctx.job_id} has no combine run')
    plan = load_plan(ctx.params, document)
    target = await _lookup(str(ctx.project_slug))
    sources = [await _lookup(s.project) for s in plan.request.sources]
    return await run_combine(
        ctx,
        client,
        plan=plan,
        sources=sources,
        target=target,
        settle=_settle_for(client, target),
        enqueue_pending=_enqueue_for(ctx),
    )
