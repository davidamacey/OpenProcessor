# ruff: noqa: F811 - pytest fixtures are imported by name and used as test arguments
"""A loop does not start while a clustering run or projection of its project is live: its
activation would re-project the items under the loop. Another project's run is none of its
business."""

from __future__ import annotations

import sys
import uuid
from dataclasses import replace
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.job_fixtures import job_env  # noqa: F401
from fixtures.job_rows import make_project, sql

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import new_project_record
from src.db.session import async_session_factory
from src.services.curation.assist_loop.prepare import clustering_is_live


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def _job(pg, project: uuid.UUID, kind: str, status: str = 'queued') -> None:
    live = "s IN ('running', 'cancelling')"
    sql(
        pg,
        'INSERT INTO jobs (project_id, kind, queue, status, max_queue_wait, requested_by, owner, '
        "lease_expires_at, pause_reason) SELECT %s, %s, 'jobs', s, interval '1 hour', 'test', "
        f"CASE WHEN {live} THEN 'test' END, "
        f"CASE WHEN {live} THEN now() + interval '1 hour' END, "
        "CASE WHEN s = 'paused' THEN 'user' END FROM (SELECT %s::text AS s) t",
        project,
        kind,
        status,
    )


@pytest.fixture
def mine(pg, job_env):
    slug = f'pg-{uuid.uuid4().hex[:6]}'
    project = make_project(pg, slug)
    record = replace(new_project_record(slug, base_curation_config()), id=project)
    with bind_project(record):
        yield project


async def _live() -> bool:
    async with async_session_factory()() as session:
        return await clustering_is_live(session)


async def test_nothing_runs_so_a_loop_may_start(mine) -> None:
    assert await _live() is False


@pytest.mark.parametrize('kind', ['cluster_run', 'cluster_projection_sync'])
@pytest.mark.parametrize('status', ['queued', 'running', 'paused', 'cancelling'])
async def test_a_live_clustering_job_blocks_a_start(pg, mine, kind: str, status: str) -> None:
    _job(pg, mine, kind, status)

    assert await _live() is True


async def test_another_kind_of_job_does_not(pg, mine) -> None:
    _job(pg, mine, 'assist_loop')

    assert await _live() is False


async def test_another_projects_clustering_run_does_not(pg, mine) -> None:
    _job(pg, make_project(pg, f'other-{uuid.uuid4().hex[:6]}'), 'cluster_run')

    assert await _live() is False
