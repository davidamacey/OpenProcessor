"""Two starts of one exclusive group: the index admits one, the loser reads the winner's id.

The insert runs in a savepoint (review R-5), so the loser's transaction survives the
``unique_violation`` and can still read the active job in the same transaction.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest
from fixtures.job_rows import make_project, new_job, project_session, sql, start
from sqlalchemy import text

from src.db.repos import jobs as repo


if TYPE_CHECKING:
    import uuid


pytestmark = pytest.mark.db


@pytest.fixture
def engine(pg):
    engine = pg.engine('app', pool_size=25)
    yield engine
    engine.dispose()


def test_a_single_start_is_queued(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project, exclusive_group='g')
    assert (job.status, job.attempt, job.retry_count, job.fencing_token) == ('queued', 0, 0, 0)
    assert job.queued_at is not None
    assert job.project_id == project


def test_the_second_start_is_busy_and_names_the_active_job(pg, engine) -> None:
    project = make_project(pg)
    first = start(engine, project, exclusive_group='g', scope='s')
    with project_session(engine, project) as session, pytest.raises(repo.JobBusyError) as busy:
        repo.insert_queued(session, new_job(project, exclusive_group='g', scope='s'))
    assert busy.value.active_job_id == first.id


def test_another_scope_or_group_is_not_busy(pg, engine) -> None:
    project = make_project(pg)
    start(engine, project, exclusive_group='g', scope='a')
    start(engine, project, exclusive_group='g', scope='b')
    start(engine, project, exclusive_group='h', scope='a')
    start(engine, project)
    start(engine, project)


def test_a_finished_job_frees_the_group(pg, engine) -> None:
    project = make_project(pg)
    first = start(engine, project, exclusive_group='g')
    sql(pg, "UPDATE jobs SET status = 'cancelled', finished_at = now() WHERE id = %s", first.id)
    second = start(engine, project, exclusive_group='g')
    assert second.id != first.id


def test_the_loser_transaction_survives_the_violation(pg, engine) -> None:
    project = make_project(pg)
    first = start(engine, project, exclusive_group='g')
    with project_session(engine, project) as session:
        with pytest.raises(repo.JobBusyError):
            repo.insert_queued(session, new_job(project, exclusive_group='g'))
        # the aborted statement only rolled back the savepoint: this read runs in the same tx
        assert session.execute(text('SELECT count(*) FROM jobs')).scalar_one() >= 1
        assert repo.get(session, first.id) is not None


def test_a_concurrent_second_start_waits_then_loses(pg, engine) -> None:
    project = make_project(pg)
    inserted, release = threading.Event(), threading.Event()
    winner: list[uuid.UUID] = []

    def holder() -> None:
        with project_session(engine, project) as session:
            winner.append(repo.insert_queued(session, new_job(project, exclusive_group='g')).job.id)
            inserted.set()
            release.wait(10)

    def loser() -> uuid.UUID | None:
        inserted.wait(10)
        with project_session(engine, project) as session:
            try:
                repo.insert_queued(session, new_job(project, exclusive_group='g'))
            except repo.JobBusyError as exc:
                return exc.active_job_id
        return None

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(holder)
        second = pool.submit(loser)
        release.wait(0.5)  # the loser is now blocked on the holder's uncommitted index entry
        release.set()
        first.result(15)
        assert second.result(15) == winner[0]


def test_twenty_racing_starts_make_exactly_one_job(pg, engine) -> None:
    project = make_project(pg)
    barrier = threading.Barrier(20)

    def attempt() -> str:
        barrier.wait(10)
        with project_session(engine, project) as session:
            try:
                repo.insert_queued(session, new_job(project, exclusive_group='g'))
            except repo.JobBusyError:
                return 'busy'
            return 'created'

    with ThreadPoolExecutor(20) as pool:
        outcomes = [f.result(30) for f in [pool.submit(attempt) for _ in range(20)]]
    assert outcomes.count('created') == 1
    assert outcomes.count('busy') == 19


def test_a_repeated_idempotency_key_returns_the_first_job(pg, engine) -> None:
    project = make_project(pg)
    with project_session(engine, project) as session:
        first = repo.insert_queued(session, new_job(project, idempotency_key='key-12345'))
    with project_session(engine, project) as session:
        second = repo.insert_queued(session, new_job(project, idempotency_key='key-12345'))
    assert first.created
    assert not second.created
    assert second.job.id == first.job.id


def test_the_same_key_for_another_kind_or_project_is_a_new_job(pg, engine) -> None:
    one, two = make_project(pg), make_project(pg)
    first = start(engine, one, idempotency_key='key-12345')
    assert start(engine, two, idempotency_key='key-12345').id != first.id
    assert start(engine, one, 'probe', idempotency_key='key-12345').id != first.id


def test_a_deployment_job_without_a_project_is_exclusive_too(pg, engine) -> None:
    from fixtures.job_rows import deployment_session

    with deployment_session(engine) as session:
        repo.insert_queued(session, new_job(None, exclusive_group='d', scope='slug'))
    with deployment_session(engine) as session, pytest.raises(repo.JobBusyError):
        repo.insert_queued(session, new_job(None, exclusive_group='d', scope='slug'))
