"""The job handlers of the near-duplicate kinds: bind the job's project, then call the service.

Each handler runs inside the Celery job wrapper (``src/tasks/runtime``); the OpenSearch client
comes from the image's registered factory, never from a request. The project is bound for the
body of the handler so the OpenSearch guard and the embedding-space resolver see the job's own
project and no other.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.services.near_duplicates import build, projection, rescore
from src.services.near_duplicates.plan import level_plan
from src.services.near_duplicates.profiles import active_profile
from src.services.near_duplicates.runs import Env
from src.services.projects.registry import get_project_registry


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.tasks.runtime.context import JobContext


class JobProjectError(RuntimeError):
    """The job names a project the registry does not know."""


@asynccontextmanager
async def job_env(ctx: JobContext, level: str) -> AsyncIterator[Env]:
    """The run environment of ``level`` for the job's project, with that project bound."""
    if ctx.project_id is None or ctx.project_slug is None:
        msg = f'job {ctx.job_id} has no project'
        raise JobProjectError(msg)
    registry = get_project_registry()
    await registry.ensure_fresh()
    record = registry.get(ctx.project_slug)
    if record is None:
        msg = f'project {ctx.project_slug!r} is not in the registry'
        raise JobProjectError(msg)
    with bind_project(record):
        async with ctx.session() as session:
            profile = await active_profile(session, ctx.project_id)
        plan = await level_plan(profile, level)
        client = await ctx.client('opensearch')
        yield Env(
            scope=ctx.session,
            client=client,
            project_id=ctx.project_id,
            profile=profile,
            plan=plan,
            job_id=ctx.job_id,
        )


def _level(ctx: JobContext) -> str:
    level = ctx.params.get('level')
    if level not in ('image', 'item'):
        msg = f'params.level must be image or item, got {level!r}'
        raise ValueError(msg)
    return str(level)


def _result(summary: Any) -> dict[str, Any]:
    return {
        'run_id': str(summary.run_id),
        'members_scored': summary.members_scored,
        'dedup_groups': summary.dedup_groups,
        'leak_groups': summary.leak_groups,
        'grouped_members': summary.grouped_members,
        'largest_group': summary.largest_group,
        'oversized_split': summary.oversized_split,
    }


async def score(ctx: JobContext) -> dict[str, Any]:
    """``near_dup_score``: a full build of one level (every vector, exact kNN, groups, flip)."""
    async with job_env(ctx, _level(ctx)) as env:
        await ctx.progress(0, None, 'score')
        summary = await build.full_build(env, cancelled=ctx.cancelled)
    return {'cancelled': True} if summary is None else _result(summary)


async def group(ctx: JobContext) -> dict[str, Any]:
    """``near_dup_group``: regroup the stored edges (after a decision or a profile change)."""
    async with job_env(ctx, _level(ctx)) as env:
        await ctx.progress(0, None, 'group')
        return _result(await build.group_stored(env, scored=0))


async def run_rescore(ctx: JobContext) -> dict[str, Any]:
    """``near_dup_rescore``: the incremental path over the pending rows of one level."""
    async with job_env(ctx, _level(ctx)) as env:
        await ctx.progress(0, None, 'rescore')
        outcome = await rescore.rescore(env)
        if outcome.summary is None:
            return {'claimed': 0}
        return {
            **_result(outcome.summary),
            'claimed': outcome.claimed,
            'missing_vectors': outcome.missing_vectors,
        }


async def projection_sync(ctx: JobContext) -> dict[str, Any]:
    """``near_dup_projection_sync``: write one run's fields to the index."""
    run_id = uuid.UUID(str(ctx.params['run_id']))
    raw_task = ctx.params.get('projection_task_id')
    async with job_env(ctx, _level(ctx)) as env:
        await ctx.progress(0, None, 'projection')
        done = await projection.apply_run(
            env, run_id, None if raw_task is None else uuid.UUID(str(raw_task))
        )
    return {'skipped': done.skipped, 'written': done.written, 'cleared': done.cleared}
