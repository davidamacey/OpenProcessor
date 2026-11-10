"""The plate-text kinds share one exclusive group, so no two writers touch the same boxes.

A reread (queue ``text_read``) while a consensus (queue ``jobs``) is live is ``409 job_busy``, and
the other way round. Docs: https://www.postgresql.org/docs/current/indexes-partial.html (the
``uq_jobs_one_active`` partial unique index that enforces the group).
"""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, sql

from src.config.job_kinds import KINDS
from src.config.queue import TaskQueues
from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.jobs.service import start_job


REREAD = 'region_text_reread'
CONSENSUS = 'region_text_consensus'


async def _start(kind: str, project):
    async with async_session_factory()() as session:
        return await start_job(session, kind, project, requested_by='test')


def test_both_kinds_share_one_group_over_the_whole_project() -> None:
    kinds = KINDS
    assert {kinds[REREAD].exclusive_group, kinds[CONSENSUS].exclusive_group} == {
        'region_text_write'
    }
    assert kinds[REREAD].exclusive_scope == kinds[CONSENSUS].exclusive_scope == ''


def test_each_kind_has_the_queue_of_the_plan() -> None:
    kinds = KINDS
    assert kinds[REREAD].queue == TaskQueues.TEXT_READ
    assert kinds[CONSENSUS].queue == TaskQueues.JOBS


@pytest.mark.db
@pytest.mark.asyncio
@pytest.mark.parametrize(('first', 'second'), [(CONSENSUS, REREAD), (REREAD, CONSENSUS)])
async def test_a_start_while_the_other_kind_is_live_is_job_busy(
    pg, job_env, producer, wakes, first: str, second: str
) -> None:
    project = make_project(pg)
    await _start(first, project)
    with pytest.raises(RequestRejectedError) as busy:
        await _start(second, project)
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')


@pytest.mark.db
@pytest.mark.asyncio
async def test_another_project_is_not_blocked_and_the_row_lands_on_its_queue(
    pg, job_env, producer, wakes
) -> None:
    busy_project, other = make_project(pg), make_project(pg)
    await _start(CONSENSUS, busy_project)
    result = await _start(REREAD, other)
    [(queue,)] = sql(pg, 'SELECT queue FROM jobs WHERE id = %s', result.job.id)
    assert queue == TaskQueues.TEXT_READ
