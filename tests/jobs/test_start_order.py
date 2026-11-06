"""start_job: the commit comes before the message (trap T-9), and the failure paths (3A-3)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from fixtures.job_rows import make_project, sql
from sqlalchemy import text

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    from src.db.repos import jobs as repo


pytestmark = pytest.mark.db


async def start(project, kind='t_plain', **kwargs):
    async with async_session_factory()() as session:
        return await start_job(session, kind, project, **kwargs)


@pytest.mark.asyncio
async def test_the_row_is_committed_before_the_message_is_sent(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    seen: list[tuple[str, str, int]] = []

    def on_send(_name: str, task_id: str) -> None:
        # a different connection: only a committed row is visible to it
        rows = sql(pg, 'SELECT status, dispatch_count FROM jobs WHERE id = %s', uuid.UUID(task_id))
        seen.append((task_id, *rows[0]))

    producer.on_send = on_send
    result = await start(project)
    assert seen == [(str(result.job.id), 'queued', 0)]


@pytest.mark.asyncio
async def test_the_happy_path_queues_sends_marks_and_records_the_dispatch(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    result = await start(project)
    assert result.created
    assert result.job.status == 'queued'
    assert producer.calls == [('send', str(result.job.id)), ('mark', str(result.job.id))]
    [(count, dispatched)] = sql(
        pg,
        'SELECT dispatch_count, last_dispatched_at IS NOT NULL FROM jobs WHERE id = %s',
        result.job.id,
    )
    assert (count, dispatched) == (1, True)
    assert wakes  # the job.state event was announced after the commit


@pytest.mark.asyncio
async def test_a_job_state_event_is_written_with_the_row(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    result = await start(project)
    [(kind, payload)] = sql(
        pg,
        "SELECT type, payload FROM events WHERE job_id = %s AND type = 'job.state'",
        result.job.id,
    )
    assert kind == 'job.state'
    assert payload['status'] == 'queued'
    assert payload['kind'] == 't_plain'


@pytest.mark.asyncio
async def test_a_broker_failure_fails_the_row_and_answers_503(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    producer.fail_send = True
    with pytest.raises(RequestRejectedError) as refused:
        await start(project)
    assert refused.value.status_code == 503
    assert refused.value.detail['error'] == 'queue_unavailable'
    [(status, code, finished)] = sql(
        pg,
        'SELECT status, error_code, finished_at IS NOT NULL FROM jobs WHERE project_id = %s',
        project,
    )
    assert (status, code, finished) == ('failed', 'queue_unavailable', True)


@pytest.mark.asyncio
async def test_the_same_key_twice_makes_one_job_and_two_equal_answers(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    first = await start(project, params={'a': 1}, idempotency_key='key-12345')
    second = await start(project, params={'a': 1}, idempotency_key='key-12345')
    assert first.created
    assert not second.created
    assert first.job.id == second.job.id
    assert sql(pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', project) == [(1,)]
    assert producer.sent() == [str(first.job.id)]  # a repeat sends nothing


@pytest.mark.asyncio
async def test_the_same_key_with_other_params_is_422(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    await start(project, params={'a': 1}, idempotency_key='key-12345')
    with pytest.raises(RequestRejectedError) as refused:
        await start(project, params={'a': 2}, idempotency_key='key-12345')
    assert refused.value.status_code == 422
    assert refused.value.detail['error'] == 'idempotency_key_reused'


@pytest.mark.asyncio
@pytest.mark.parametrize('key', ['short', 'x' * 201])
async def test_a_key_outside_8_to_200_characters_is_422(pg, job_env, producer, wakes, key) -> None:
    project = make_project(pg)
    with pytest.raises(RequestRejectedError) as refused:
        await start(project, idempotency_key=key)
    assert (refused.value.status_code, refused.value.detail['error']) == (
        422,
        'idempotency_key_invalid',
    )


@pytest.mark.asyncio
async def test_a_second_start_of_an_exclusive_group_is_409_with_the_active_id(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    first = await start(project, 't_excl', params={'target': 'x'})
    with pytest.raises(RequestRejectedError) as refused:
        await start(project, 't_excl', params={'target': 'x'})
    assert refused.value.status_code == 409
    assert refused.value.detail['error'] == 'job_busy'
    assert refused.value.detail['active_job_id'] == str(first.job.id)
    assert (await start(project, 't_excl', params={'target': 'y'})).created  # another scope
    assert sql(pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', project) == [(2,)]
    assert len(producer.sent()) == 2


@pytest.mark.asyncio
async def test_with_rows_runs_in_the_jobs_transaction(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    inserted: list[uuid.UUID] = []

    async def with_rows(session, job: repo.JobRow) -> None:
        await session.execute(
            text("INSERT INTO job_items (job_id, position, item_id) VALUES (:j, 0, 'item-1')"),
            {'j': job.id},
        )
        inserted.append(job.id)

    result = await start(project, with_rows=with_rows)
    assert inserted == [result.job.id]
    assert sql(pg, 'SELECT item_id FROM job_items WHERE job_id = %s', result.job.id) == [
        ('item-1',)
    ]

    async def failing(_session, _job) -> None:
        raise RuntimeError('domain row refused')

    with pytest.raises(RuntimeError):
        await start(project, with_rows=failing)
    assert sql(pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', project) == [(1,)]


@pytest.mark.asyncio
async def test_with_rows_is_not_called_when_the_group_is_busy(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    await start(project, 't_excl', params={'target': 'x'})
    called: list[int] = []

    async def with_rows(_session, _job) -> None:
        called.append(1)

    with pytest.raises(RequestRejectedError):
        await start(project, 't_excl', params={'target': 'x'}, with_rows=with_rows)
    assert called == []


@pytest.mark.asyncio
async def test_a_deployment_job_starts_without_a_project(pg, job_env, producer, wakes) -> None:
    result = await start(None, 't_dep', params={'slug': 's1'})
    assert result.job.project_id is None
    assert result.job.scope == 's1'
    assert result.job.exclusive_group == 'dep'
    with pytest.raises(RequestRejectedError) as refused:
        await start(None, 't_dep', params={'slug': 's1'})
    assert refused.value.detail['error'] == 'job_busy'


@pytest.mark.asyncio
async def test_a_project_kind_without_a_project_is_a_programming_error(
    pg, job_env, producer, wakes
) -> None:
    with pytest.raises(ValueError, match='needs a project'):
        await start(None, 't_plain')
    with pytest.raises(ValueError, match='takes no project'):
        await start(make_project(pg), 't_dep', params={'slug': 'x'})
