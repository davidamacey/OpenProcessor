"""Write a grouping run and flip it current, in one transaction.

The flip is the only place a run becomes ``current``: the old current run is superseded, the
new one finished with its counters, the projection task queued (``near_dup_run``) and the
``near_dup.run`` event written, all or nothing. Older runs beyond the retention are pruned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.db.repos import near_dup as repo
from src.domain import near_dup as rules
from src.services.events.publish import apublish
from src.services.near_duplicates.holdout import absorb


if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.near_duplicates.plan import LevelPlan
    from src.services.near_duplicates.profiles import ActiveProfile

type SessionScope = Callable[[], AbstractAsyncContextManager[AsyncSession]]


@dataclass(frozen=True)
class Env:
    """Everything a run needs: how to open a scoped session, the client, the plan.

    The core level has no project and no profile (``None`` for both): its session is a core or
    deployment scope and its plan is :func:`~src.services.near_duplicates.plan.core_plan`.
    """

    scope: SessionScope
    client: AsyncOpenSearch
    project_id: uuid.UUID | None
    profile: ActiveProfile | None
    plan: LevelPlan
    job_id: uuid.UUID | None


@dataclass(frozen=True)
class RunSummary:
    run_id: uuid.UUID
    members_scored: int
    dedup_groups: int
    leak_groups: int
    grouped_members: int
    largest_group: int
    oversized_split: int
    holdout_absorbed: int = 0


def _counters(
    scored: int,
    dedup: list[repo.GroupToWrite],
    leak: list[repo.GroupToWrite],
    oversized: int,
) -> dict[str, int]:
    sizes = [len(g.members) for g in (*dedup, *leak)]
    return {
        'members_scored': scored,
        'dedup_groups': len(dedup),
        'leak_groups': len(leak),
        'grouped_members': sum(len(g.members) for g in dedup),
        'largest_group': max(sizes, default=0),
        'oversized_split': oversized,
    }


async def write_and_flip(
    env: Env,
    *,
    mode: str,
    parent_run_id: uuid.UUID | None,
    dedup: list[repo.GroupToWrite],
    leak: list[repo.GroupToWrite],
    scored: int,
    oversized: int,
) -> RunSummary:
    """Insert the run and its groups as ``building``, then flip it current (one transaction)."""
    counters = _counters(scored, dedup, leak, oversized)
    async with env.scope() as session:
        await repo.discard_building(session, env.project_id, env.plan.level)
        run_id = await repo.start_run(
            session,
            project_id=env.project_id,
            level=env.plan.level,
            mode=mode,
            job_id=env.job_id,
            parent_run_id=parent_run_id,
            profile=None if env.profile is None else (env.profile.config_id, env.profile.revision),
            params=env.plan.params,
            space=env.plan.space_key,
            algorithm_version=rules.ALGORITHM_VERSION,
        )
        await repo.write_groups(session, run_id, 'dedup', dedup)
        await repo.write_groups(session, run_id, 'leak', leak)
        run = await repo.get_run(session, run_id)
        assert run is not None  # inserted above
        if env.plan.is_item:
            assert env.project_id is not None  # an item run is always a project run
            counters['holdout_absorbed'] = await absorb(
                session,
                env.client,
                index=env.plan.index,
                project_id=env.project_id,
                run_id=run_id,
                leak=leak,
            )
        await repo.flip_current(session, run, counters)
        await repo.prune_runs(session, env.project_id, env.plan.level, rules.KEPT_SUPERSEDED_RUNS)
        if env.project_id is not None:  # a core run has no project stream to publish on
            await apublish(
                'near_dup.run',
                {'run_id': str(run_id), 'level': env.plan.level, 'mode': mode, **counters},
                project=env.project_id,
                session=session,
            )
    return RunSummary(run_id=run_id, **counters)
