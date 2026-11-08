"""Two starts of one level give one active near_dup job; another level or project is free."""

from __future__ import annotations

import asyncio

import pytest
from fixtures.job_rows import make_project

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.jobs.service import start_job


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _start(kind: str, project, level: str):
    async with async_session_factory()() as session:
        return await start_job(session, kind, project, params={'level': level})


async def test_two_concurrent_starts_of_one_level_give_one_job(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    results = await asyncio.gather(
        _start('near_dup_score', project, 'image'),
        _start('near_dup_rescore', project, 'image'),
        return_exceptions=True,
    )
    started = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, RequestRejectedError)]
    assert len(started) == 1
    assert len(refused) == 1
    assert refused[0].status_code == 409
    assert refused[0].detail['error'] == 'job_busy'


async def test_the_other_level_and_the_projection_group_are_independent(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    await _start('near_dup_score', project, 'image')
    await _start('near_dup_score', project, 'item')
    await _start('near_dup_projection_sync', project, 'image')
    other = make_project(pg)
    await _start('near_dup_score', other, 'image')
