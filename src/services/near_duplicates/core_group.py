"""Group the core global index from its pending rows: the body of ``near_dup_core_group``.

Core ingest writes ``near_dup_pending`` rows (level ``core``, no project) and returns; this job is
the one writer (exclusive group ``near_dup_core``), so two concurrent ingest batches that hold a
near-duplicate pair can no longer each create a group for it. It reuses the project levels'
machinery with the core plan (``plan.core_plan``): the pending claim, the kNN edges, the
components, the run flip and the projection, with no profile and no human decisions.

Docs: Celery task retries and idempotency are the job runtime's (``src/tasks/runtime``); the
claim is ``SELECT ... FOR UPDATE SKIP LOCKED`` (PostgreSQL explicit locking).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.near_duplicates import projection, rescore
from src.services.near_duplicates.plan import core_plan
from src.services.near_duplicates.runs import Env


if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable

    from opensearchpy import AsyncOpenSearch

    from src.services.near_duplicates.runs import SessionScope
    from src.tasks.runtime.context import JobContext


@dataclass(frozen=True)
class CoreDrain:
    claimed: int
    """Pending rows claimed over every pass."""
    runs: int
    missing_vectors: int
    """Rows kept pending because their image has no vector in the index yet."""
    grouped_members: int
    """Members in a group in the last run written."""
    expired: tuple[str, ...] = ()
    """Images dropped as missing: no vector in the index within ``rescore.MISSING_MAX_AGE``."""


def core_env(client: AsyncOpenSearch, *, scope: SessionScope, job_id: uuid.UUID | None) -> Env:
    """The run environment of the core level: no project, no profile."""
    return Env(
        scope=scope, client=client, project_id=None, profile=None, plan=core_plan(), job_id=job_id
    )


async def drain(env: Env, *, cancelled: Callable[[], bool] = lambda: False) -> CoreDrain:
    """Rescore and project passes until the pending rows are drained.

    A pass that drains nothing (every claimed row is waiting for its vector) ends the loop, so a
    row that cannot be handled yet is left for the next job instead of spinning here.
    """
    claimed = runs = missing = grouped = 0
    expired: list[str] = []
    while not cancelled():
        outcome = await rescore.rescore(env)
        if outcome.summary is None:
            break
        claimed += outcome.claimed
        missing = outcome.missing_vectors
        runs += 1
        expired.extend(outcome.expired)
        grouped = outcome.summary.grouped_members
        await projection.apply_run(env, outcome.summary.run_id, None)
        if outcome.claimed == outcome.missing_vectors:
            break
    return CoreDrain(claimed, runs, missing, grouped, tuple(sorted(expired)))


async def run(ctx: JobContext) -> dict[str, Any]:
    """``near_dup_core_group``: drain the core pending rows into runs, then project the fields."""
    env = core_env(await ctx.client('opensearch'), scope=ctx.session, job_id=ctx.job_id)
    await ctx.progress(0, None, 'core_group')
    done = await drain(env, cancelled=ctx.cancelled)
    return {
        'claimed': done.claimed,
        'runs': done.runs,
        'missing_vectors': done.missing_vectors,
        'grouped_members': done.grouped_members,
        'missing': list(done.expired),
    }
