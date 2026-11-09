"""Starting the re-index kind: the kind table row, one build per space, idempotency, pause."""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, sql

from src.config.job_kinds import KINDS
from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.jobs.service import pause_job, start_job


pytestmark = [pytest.mark.db, pytest.mark.asyncio, pytest.mark.usefixtures('bound_async_engine')]

KIND = 'embedding_reindex'


def params(space_key: str = 'space_a', **extra):
    return {'space_key': space_key, 'target': 'item', 'mode': 'build', 'batch_size': 64, **extra}


async def start(project, **kwargs):
    async with async_session_factory()() as session:
        return await start_job(session, KIND, project, **kwargs)


def test_the_kind_row_matches_the_kind_table() -> None:
    kind = KINDS[KIND]
    assert (kind.queue, kind.task_name) == ('jobs', 'op.jobs.embedding_reindex')
    assert (kind.exclusive_group, kind.exclusive_scope) == (KIND, 'space_key')
    assert kind.auto_resume
    assert kind.pausable
    assert kind.slice_s == 600
    assert (kind.soft_limit_s, kind.hard_limit_s) == (30 * 60, 35 * 60)
    assert kind.max_queue_wait_s == 24 * 3600
    assert kind.gpu_affinity == 'any_card'


async def test_a_start_queues_the_job_with_its_scope_and_dispatches_it(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    result = await start(project, params=params())
    assert result.created
    assert sql(pg, 'SELECT status, scope, queue FROM jobs WHERE id = %s', result.job.id) == [
        ('queued', 'space_a', 'jobs')
    ]
    assert producer.sent() == [str(result.job.id)]


async def test_a_second_build_for_one_space_is_409_but_another_space_starts(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    first = await start(project, params=params('space_a'))
    with pytest.raises(RequestRejectedError) as busy:
        await start(project, params=params('space_a'))
    assert busy.value.status_code == 409
    assert busy.value.detail['error'] == 'job_busy'
    assert busy.value.detail['active_job_id'] == str(first.job.id)
    other = await start(project, params=params('space_b'))
    assert other.created


async def test_the_same_space_in_another_project_is_not_exclusive(
    pg, job_env, producer, wakes
) -> None:
    one, two = make_project(pg), make_project(pg)
    assert (await start(one, params=params())).created
    assert (await start(two, params=params())).created


async def test_an_idempotent_repeat_returns_the_first_job_and_other_params_are_422(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    key = 'reindex-key-0001'
    first = await start(project, params=params(), idempotency_key=key)
    repeat = await start(project, params=params(), idempotency_key=key)
    assert (first.created, repeat.created) == (True, False)
    assert repeat.job.id == first.job.id
    with pytest.raises(RequestRejectedError) as reused:
        await start(project, params=params(batch_size=8), idempotency_key=key)
    assert (reused.value.status_code, reused.value.detail['error']) == (
        422,
        'idempotency_key_reused',
    )


async def test_the_kind_is_pausable_at_a_boundary(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    job = (await start(project, params=params())).job
    async with async_session_factory()() as session:
        paused = await pause_job(session, job.id, project)
    assert paused.job.status == 'paused'
