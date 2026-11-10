"""``op.periodic.region_text_consensus_sweep`` starts a consensus once the readings have settled.

The marker (``project_state`` key ``region_text_consensus_marker``) is touched by the reader; one
untouched for 30 s means the burst is over. No marker, a cleared one, or a fresh one starts
nothing. https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest
from fixtures.job_rows import make_project, sql

from src.db.session import async_session_factory
from src.services.jobs.service import start_job
from src.services.periodic import region_text_sweep
from src.services.periodic.runner import run_periodic


if TYPE_CHECKING:
    import uuid

pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures('pg_env', 'job_env', 'producer', 'wakes'),
]

KEY = 'region_text_consensus_marker'


def set_marker(pg, project: uuid.UUID, *, age_s: int, dirty: bool = True) -> None:
    sql(
        pg,
        'INSERT INTO project_state (project_id, key, value, revision, updated_at) '
        'VALUES (%s, %s, %s::jsonb, 1, now() - make_interval(secs => %s))',
        project,
        KEY,
        json.dumps({'dirty': dirty}),
        age_s,
    )


def consensus_jobs(pg, project: uuid.UUID) -> list[str]:
    rows = sql(
        pg, "SELECT kind FROM jobs WHERE project_id = %s AND kind LIKE 'region_text_%%'", project
    )
    return [kind for (kind,) in rows]


async def start_reread(project: uuid.UUID) -> None:
    async with async_session_factory()() as session:
        await start_job(session, 'region_text_reread', project, requested_by='test')


def sweep() -> object:
    return run_periodic('region_text_consensus_sweep', region_text_sweep.tick())


def test_a_settled_marker_starts_one_consensus(pg) -> None:
    project = make_project(pg)
    set_marker(pg, project, age_s=120)
    sweep()
    sweep()  # delivered twice: the idempotency key and the exclusive group keep it to one
    assert consensus_jobs(pg, project) == ['region_text_consensus']


def test_a_marker_touched_within_30_seconds_starts_nothing(pg) -> None:
    project = make_project(pg)
    set_marker(pg, project, age_s=5)
    sweep()
    assert consensus_jobs(pg, project) == []


def test_no_new_readings_means_no_job(pg) -> None:
    never, cleared = make_project(pg), make_project(pg)
    set_marker(pg, cleared, age_s=600, dirty=False)
    sweep()
    assert consensus_jobs(pg, never) == []
    assert consensus_jobs(pg, cleared) == []


def test_a_live_reread_defers_the_consensus_to_a_later_tick(pg) -> None:
    project = make_project(pg)
    set_marker(pg, project, age_s=120)
    asyncio.run(start_reread(project))
    sweep()  # job_busy is not an error: the next tick tries again
    assert consensus_jobs(pg, project) == ['region_text_reread']
