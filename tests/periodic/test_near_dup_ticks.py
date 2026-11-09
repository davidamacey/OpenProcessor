"""The near-duplicate beat ticks: a rescore for each built level with pending members, and the
weekly full build of the image level. Neither starts anything for a level that was never built or a
project without an active profile; both skip a level whose job is live and start one job when the
same slot is delivered twice.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import TYPE_CHECKING, Any

import pytest
from fixtures.job_rows import sql
from near_duplicates.conftest import make_project, scope_for

from src.db.repos import near_dup as repo
from src.services.near_duplicates import pending, profiles
from src.services.periodic import near_dup_ticks
from src.services.periodic.runner import run_periodic


if TYPE_CHECKING:
    import uuid


pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures('pg_registry', 'job_env', 'producer', 'wakes'),
]


def new_project(pg) -> uuid.UUID:
    project = asyncio.run(make_project())
    sql(pg, "UPDATE projects SET status = 'active' WHERE id = %s", project)
    return project


def build_level(project: uuid.UUID, level: str) -> None:
    async def go() -> None:
        async with scope_for(project)() as session:
            profile = await profiles.active_profile(session, project)
            run = await repo.start_run(
                session,
                project_id=project,
                level=level,
                mode='full',
                job_id=None,
                parent_run_id=None,
                profile=(profile.config_id, profile.revision),
                params={},
                space='pe@abc:pe_embedding',
                algorithm_version='nd-1',
            )
            row = await repo.get_run(session, run)
            assert row is not None
            await repo.flip_current(session, row, {})

    asyncio.run(go())


def mark_pending(project: uuid.UUID, level: str) -> None:
    async def go() -> None:
        async with scope_for(project)() as session:
            await pending.enqueue_pending(session, project, level, ['m1', 'm2'], 'vector_changed')

    asyncio.run(go())


def jobs(pg, project: uuid.UUID) -> list[tuple[str, Any]]:
    rows = sql(
        pg, 'SELECT kind, params FROM jobs WHERE project_id = %s ORDER BY created_at', project
    )
    return [(kind, params['level']) for kind, params in rows]


def rescore(now: float = 1_800_000_000.0) -> Any:
    return run_periodic('near_dup_rescore', near_dup_ticks.rescore_tick(now=now))


def test_a_built_level_with_pending_members_gets_one_rescore(pg) -> None:
    project = new_project(pg)
    build_level(project, 'image')
    mark_pending(project, 'image')

    rescore()
    rescore()  # the same slot delivered twice, and the first job is still live

    assert jobs(pg, project) == [('near_dup_rescore', 'image')]


def test_a_level_never_built_or_with_nothing_pending_starts_nothing(pg) -> None:
    unbuilt, quiet = new_project(pg), new_project(pg)
    mark_pending(unbuilt, 'image')  # pending, but there is no run to rescore from
    build_level(quiet, 'image')  # built, nothing pending

    rescore()

    assert jobs(pg, unbuilt) == []
    assert jobs(pg, quiet) == []


def test_each_built_level_with_pending_members_gets_its_own_job(pg) -> None:
    project = new_project(pg)
    for level in ('image', 'item'):
        build_level(project, level)
        mark_pending(project, level)

    rescore()

    assert sorted(jobs(pg, project)) == [
        ('near_dup_rescore', 'image'),
        ('near_dup_rescore', 'item'),
    ]


def test_a_project_without_an_active_profile_starts_nothing(pg) -> None:
    project = new_project(pg)
    build_level(project, 'image')
    mark_pending(project, 'image')
    sql(pg, "DELETE FROM config_activations WHERE project_id = %s AND axis = 'near_dup'", project)

    rescore()

    assert jobs(pg, project) == []


def test_the_weekly_full_build_starts_one_image_score_per_week_and_only_for_built_projects(
    pg,
) -> None:
    built, unbuilt = new_project(pg), new_project(pg)
    build_level(built, 'image')
    week = dt.date(2026, 11, 9)

    def build(today: dt.date) -> None:
        run_periodic('near_dup_full_build', near_dup_ticks.full_build_tick(today=today))

    build(week)
    build(week + dt.timedelta(days=2))  # the same ISO week: the idempotency key repeats

    assert jobs(pg, built) == [('near_dup_score', 'image')]
    assert jobs(pg, unbuilt) == []
