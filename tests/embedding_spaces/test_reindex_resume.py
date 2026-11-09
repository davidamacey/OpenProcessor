"""The re-index job end to end through the job wrapper, against a real PostgreSQL and fakes.

Kill mid-run and resume without re-encoding a finished document; a moved box is skipped as
deferred; a 100-doc build completes and the space becomes ready.
"""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, project_session, sql

from src.config.job_kinds import KINDS
from src.db.engine import get_sync_engine
from src.services.embedding_spaces.reindex import run_reindex
from src.services.jobs.service import insert_job
from src.tasks.runtime.job_task import run_job
from tests.embedding_spaces._reindex_fakes import KEY, CountingEncoder, FakeIndex, seed_space


pytestmark = pytest.mark.db

KIND = 'embedding_reindex'


@pytest.fixture
def engine(job_env, wakes, queue_hooks):
    return get_sync_engine()


def _handler(index, encoder):
    async def handler(ctx):
        return await run_reindex(ctx, index, encoder)

    return handler


def _start(engine, pg, *, batch=10, mode='build', target='item', status='pending'):
    project = make_project(pg)
    space = seed_space(pg, project, status=status)
    params = {'space_key': KEY, 'target': target, 'mode': mode, 'batch_size': batch}
    with project_session(engine, project) as session:
        job = insert_job(session, KIND, project, params=params).job
    sql(
        pg,
        'INSERT INTO embedding_reindex_runs (project_id, space_id, job_id, mode) '
        'VALUES (%s, %s, %s, %s)',
        project,
        space,
        job.id,
        mode,
    )
    return project, space, job


def _space_status(pg, space):
    return sql(pg, 'SELECT status FROM embedding_spaces WHERE id = %s', space)[0][0]


def test_a_100_doc_build_completes_and_the_space_is_ready(pg, engine) -> None:
    index, encoder = FakeIndex(100), CountingEncoder()
    _, space, job = _start(engine, pg)
    run_job(KINDS[KIND], _handler(index, encoder), {}, str(job.id))
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job.id) == [('completed',)]
    assert _space_status(pg, space) == 'ready'
    assert len(index.vectors) == 100
    # only the verify sample (the last five documents) is encoded a second time
    assert {d for d, n in encoder.calls.items() if n == 2} == {f'd{i:03d}' for i in range(95, 100)}
    assert max(encoder.calls.values()) == 2
    assert index.mapped == ['items']
    [(done, total, finished)] = sql(
        pg,
        'SELECT docs_done, docs_total, finished_at IS NOT NULL FROM embedding_reindex_runs '
        'WHERE job_id = %s',
        job.id,
    )
    assert (done, total, finished) == (100, 100, True)


def test_a_killed_run_resumes_without_re_encoding_a_finished_document(pg, engine) -> None:
    index, dying = FakeIndex(100), CountingEncoder(die_at=6)  # five batches of 10 are durable
    _, space, job = _start(engine, pg)
    with pytest.raises(SystemExit):
        run_job(KINDS[KIND], _handler(index, dying), {}, str(job.id))
    assert len(index.vectors) == 50
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job.id) == [('interrupted',)]
    assert _space_status(pg, space) == 'building'
    # reaper rule: an interrupted auto-resume job goes back to the queue
    sql(pg, "UPDATE jobs SET status = 'queued' WHERE id = %s", job.id)

    encoder = CountingEncoder()
    run_job(KINDS[KIND], _handler(index, encoder), {}, str(job.id))
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job.id) == [('completed',)]
    assert len(index.vectors) == 100
    finished_before_the_kill = {f'd{i:03d}' for i in range(50)}
    assert not finished_before_the_kill & set(encoder.calls)
    assert _space_status(pg, space) == 'ready'


def test_a_box_that_moved_after_the_read_is_skipped_as_deferred(pg, engine) -> None:
    index, encoder = FakeIndex(10, boxes=True), CountingEncoder()

    def move(idx):
        idx.bbox['d003'] = (0.1, 0.1, 0.6, 0.6)
        idx.on_write = None

    index.on_write = move
    _, space, job = _start(engine, pg, batch=10)
    run_job(KINDS[KIND], _handler(index, encoder), {}, str(job.id))
    assert index.state['d003'] == 'deferred'
    assert 'd003' not in index.vectors
    assert len(index.vectors) == 9
    [(done, skipped)] = sql(
        pg,
        'SELECT docs_done, docs_skipped FROM embedding_reindex_runs WHERE job_id = %s',
        job.id,
    )
    assert (done, skipped) == (9, 1)
    assert _space_status(pg, space) == 'ready'


def test_an_undecodable_document_is_failed_and_the_job_completes_with_errors(pg, engine) -> None:
    index, encoder = FakeIndex(10), CountingEncoder(skip=frozenset({'d004'}))
    _, _, job = _start(engine, pg)
    run_job(KINDS[KIND], _handler(index, encoder), {}, str(job.id))
    assert index.state['d004'] == 'failed'
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job.id) == [('completed_with_errors',)]


def test_a_failed_verify_marks_the_space_failed(pg, engine) -> None:
    index, encoder = FakeIndex(10), CountingEncoder()
    real_write = index.write

    async def corrupting(space, writes):
        out = await real_write(space, writes)
        index.vectors['d009'] = [0.0, 0.0, 1.0]  # stored vector no longer matches a re-encode
        return out

    index.write = corrupting
    _, space, job = _start(engine, pg)
    run_job(KINDS[KIND], _handler(index, encoder), {}, str(job.id))
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job.id) == [('failed',)]
    assert _space_status(pg, space) == 'failed'


def test_a_cancel_of_an_abandoned_build_clears_the_partial_field(pg, engine) -> None:
    index = FakeIndex(20)
    _, space, job = _start(engine, pg)

    def abandon_then_cancel(idx: FakeIndex) -> None:
        if idx.vectors:  # after the first durable batch: the route marked failed, cancel is asked
            sql(pg, "UPDATE embedding_spaces SET status = 'failed' WHERE id = %s", space)
            sql(
                pg,
                "UPDATE jobs SET status = 'cancelling', cancel_requested_at = now() WHERE id = %s",
                job.id,
            )

    index.on_write = abandon_then_cancel
    run_job(KINDS[KIND], _handler(index, CountingEncoder()), {}, str(job.id))
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job.id) == [('cancelled',)]
    assert _space_status(pg, space) == 'failed'
    assert index.vectors == {}
