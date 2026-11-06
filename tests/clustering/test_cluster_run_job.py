"""Starting a run: one queued row, one message on the cluster queue; a second start is 409."""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, sql

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.clustering.job import start_projection, start_run


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _run(project):
    async with async_session_factory()() as session:
        return await start_run(
            session,
            project,
            target='item',
            level='class',
            embedding_field='pe_embedding',
            requested_by='test',
        )


async def test_a_start_queues_one_row_on_the_cluster_queue(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    result = await _run(project)
    assert result.created
    assert producer.calls == [('send', str(result.job.id)), ('mark', str(result.job.id))]
    [(status, queue, kind)] = sql(
        pg, 'SELECT status, queue, kind FROM jobs WHERE id = %s', result.job.id
    )
    assert (status, queue, kind) == ('queued', 'cluster', 'cluster_run')


async def test_a_second_start_for_the_same_key_is_job_busy(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    await _run(project)
    with pytest.raises(RequestRejectedError) as busy:
        await _run(project)
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')
    assert sql(pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', project)[0][0] == 1


async def test_a_projection_start_is_idempotent_per_partition(pg, job_env, producer, wakes) -> None:
    import uuid

    project = make_project(pg)
    partition = uuid.uuid4()

    async def start():
        async with async_session_factory()() as session:
            return await start_projection(
                session,
                project,
                partition_id=partition,
                target='item',
                level='class',
                embedding_field='pe_embedding',
                requested_by='test',
            )

    first, second = await start(), await start()
    assert first.created
    assert not second.created
    assert second.job.id == first.job.id
