"""The claim admits one delivery; every holder write is fenced by its token (02b section 3.1)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fixtures.job_rows import make_project, project_session, sql, start

from src.db.repos import jobs as repo


pytestmark = pytest.mark.db

TTL = 120


@pytest.fixture
def engine(pg):
    engine = pg.engine('app', pool_size=15)
    yield engine
    engine.dispose()


def claim(engine, project, job_id, owner='w:1'):
    with project_session(engine, project) as session:
        return repo.claim(session, job_id, owner, TTL)


def test_a_single_claim_runs_the_job_and_opens_an_attempt(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    result = claim(engine, project, job.id)
    assert isinstance(result, repo.Claimed)
    row = result.job
    assert (row.status, row.attempt, row.fencing_token, row.owner) == ('running', 1, 1, 'w:1')
    assert row.lease_expires_at is not None
    assert row.started_at is not None
    attempts = sql(
        pg,
        'SELECT attempt, fencing_token, owner, outcome FROM job_attempts WHERE job_id = %s',
        job.id,
    )
    assert attempts == [(1, 1, 'w:1', None)]


def test_a_single_finish_ends_the_attempt(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    token = claim(engine, project, job.id).job.fencing_token
    with project_session(engine, project) as session:
        done = repo.finish(
            session, job.id, token, status='completed', outcome='completed', result={'n': 3}
        )
    assert done is not None
    assert (done.status, done.owner, done.lease_expires_at) == ('completed', None, None)
    assert done.finished_at is not None
    assert done.result == {'n': 3}
    assert done.retry_count == 0
    [(outcome, ended)] = sql(
        pg, 'SELECT outcome, ended_at IS NOT NULL FROM job_attempts WHERE job_id = %s', job.id
    )
    assert (outcome, ended) == ('completed', True)


def test_twenty_concurrent_deliveries_claim_once(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    barrier = threading.Barrier(12)

    def deliver(n: int):
        barrier.wait(10)
        return claim(engine, project, job.id, f'w:{n}')

    with ThreadPoolExecutor(12) as pool:
        results = [f.result(30) for f in [pool.submit(deliver, n) for n in range(12)]]
    assert sum(isinstance(r, repo.Claimed) for r in results) == 1
    refused = [r for r in results if isinstance(r, repo.ClaimRefused)]
    assert {(r.reason, r.status) for r in refused} == {('not_queued', 'running')}
    assert sql(pg, 'SELECT count(*) FROM job_attempts WHERE job_id = %s', job.id) == [(1,)]


@pytest.mark.parametrize(
    ('status', 'extra'),
    [
        ('paused', ", pause_reason = 'user'"),
        ('cancelled', ', finished_at = now()'),
        ('completed', ', finished_at = now()'),
        ('failed', ', finished_at = now()'),
        ('interrupted', ''),
    ],
)
def test_a_stale_message_for_a_row_that_is_not_queued_claims_nothing(
    pg, engine, status, extra
) -> None:
    project = make_project(pg)
    job = start(engine, project)
    # reach the status by a legal path: queued rows are claimed first when the move needs it
    if status in ('completed', 'failed', 'interrupted'):
        token = claim(engine, project, job.id).job.fencing_token
        with project_session(engine, project) as session:
            if status == 'interrupted':
                repo.interrupt(session, job.id, token)
            else:
                repo.finish(session, job.id, token, status=status, outcome=status)
        if status == 'interrupted':
            sql(pg, 'DELETE FROM job_attempts WHERE job_id = %s', job.id)
    else:
        sql(pg, f'UPDATE jobs SET status = %s{extra} WHERE id = %s', status, job.id)
    result = claim(engine, project, job.id)
    assert isinstance(result, repo.ClaimRefused)
    assert (result.reason, result.status) == ('not_queued', status)


def test_a_missing_job_is_refused(pg, engine) -> None:
    import uuid

    project = make_project(pg)
    result = claim(engine, project, uuid.uuid4())
    assert result == repo.ClaimRefused('missing')


def test_a_spent_retry_budget_is_refused_as_exhausted(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    sql(pg, 'UPDATE jobs SET retry_count = max_attempts WHERE id = %s', job.id)
    assert claim(engine, project, job.id) == repo.ClaimRefused('exhausted', 'queued')
    with project_session(engine, project) as session:
        failed = repo.fail_queued(session, job.id, 'attempts_exhausted')
    assert failed is not None
    assert (failed.status, failed.error_code) == ('failed', 'attempts_exhausted')


def test_a_stale_token_cannot_write_after_a_takeover(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    old = claim(engine, project, job.id, 'w:old').job.fencing_token
    # the reaper took the job over and a new delivery claimed it
    sql(
        pg,
        "UPDATE jobs SET status = 'interrupted', owner = NULL, lease_expires_at = NULL WHERE id = %s",
        job.id,
    )
    sql(pg, "UPDATE jobs SET status = 'queued' WHERE id = %s", job.id)
    new = claim(engine, project, job.id, 'w:new').job.fencing_token
    assert new == old + 1
    with project_session(engine, project) as session:
        assert repo.heartbeat(session, job.id, old, TTL) is None
        assert repo.progress(session, job.id, old, {'done': 9}) is None
        assert repo.checkpoint(session, job.id, old, {'cursor': 9}) is None
        assert repo.finish(session, job.id, old, status='failed', outcome='failed') is None
        assert repo.interrupt(session, job.id, old) is None
        assert repo.heartbeat(session, job.id, new, TTL) is not None
    [(status, progress, checkpoint, owner)] = sql(
        pg, 'SELECT status, progress, checkpoint, owner FROM jobs WHERE id = %s', job.id
    )
    assert (status, progress, checkpoint, owner) == ('running', {}, None, 'w:new')


def test_heartbeat_progress_and_checkpoint_report_the_requests(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    token = claim(engine, project, job.id).job.fencing_token
    with project_session(engine, project) as session:
        control = repo.heartbeat(session, job.id, token, TTL)
        assert control == repo.Control('running', False, False, None)
        stored = repo.progress(session, job.id, token, {'done': 1, 'total': 4})
        assert stored == control
        assert repo.checkpoint(session, job.id, token, {'after': 'x'}) == control
    with project_session(engine, project) as session:
        cancelled = repo.cancel(session, job.id, 'tester')
        assert cancelled is not None
        assert cancelled.status == 'cancelling'
    with project_session(engine, project) as session:
        control = repo.heartbeat(session, job.id, token, TTL)
        assert control == repo.Control('cancelling', True, False, None)
    row = sql(pg, 'SELECT progress, checkpoint FROM jobs WHERE id = %s', job.id)
    assert row == [({'done': 1, 'total': 4}, {'after': 'x'})]


def test_a_job_that_finishes_after_a_cancel_request_ends_cancelled(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    token = claim(engine, project, job.id).job.fencing_token
    with project_session(engine, project) as session:
        repo.cancel(session, job.id, 'tester')
        done = repo.finish(
            session, job.id, token, status='completed', outcome='completed', result={'n': 1}
        )
    assert done is not None
    assert done.status == 'cancelled'
    assert done.result == {'n': 1}
    assert sql(pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s', job.id) == [('cancelled',)]


@pytest.mark.parametrize(
    ('outcome', 'status', 'retry'),
    [
        ('time_limit', 'failed', 1),
        ('failed', 'failed', 0),
        ('completed', 'completed', 0),
    ],
)
def test_only_a_failed_delivery_spends_a_retry(pg, engine, outcome, status, retry) -> None:
    project = make_project(pg)
    job = start(engine, project)
    token = claim(engine, project, job.id).job.fencing_token
    with project_session(engine, project) as session:
        done = repo.finish(session, job.id, token, status=status, outcome=outcome)
    assert done is not None
    assert done.retry_count == retry


def test_an_interrupt_spends_a_retry_and_records_the_outcome(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project, auto_resume=True)
    token = claim(engine, project, job.id).job.fencing_token
    with project_session(engine, project) as session:
        moved = repo.interrupt(session, job.id, token)
    assert moved is not None
    assert (moved.status, moved.retry_count, moved.owner) == ('interrupted', 1, None)
    assert sql(pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s', job.id) == [
        ('interrupted',)
    ]


def test_an_interrupt_while_cancelling_ends_cancelled(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    token = claim(engine, project, job.id).job.fencing_token
    with project_session(engine, project) as session:
        repo.cancel(session, job.id, 'tester')
        moved = repo.interrupt(session, job.id, token)
    assert moved is not None
    assert moved.status == 'cancelled'
    assert moved.finished_at is not None


def test_cancel_acts_at_once_on_a_row_without_a_holder(pg, engine) -> None:
    project = make_project(pg)
    queued = start(engine, project)
    with project_session(engine, project) as session:
        row = repo.cancel(session, queued.id, 'tester')
    assert row is not None
    assert (row.status, row.finished_at is not None, row.cancel_requested_by) == (
        'cancelled',
        True,
        'tester',
    )
    with project_session(engine, project) as session:
        assert repo.cancel(session, queued.id, 'tester') is None  # terminal


def test_cancel_twice_while_running_changes_nothing_the_second_time(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    claim(engine, project, job.id)
    with project_session(engine, project) as session:
        first = repo.cancel(session, job.id, 'a')
    with project_session(engine, project) as session:
        second = repo.cancel(session, job.id, 'b')
    assert first is not None
    assert second is not None
    assert second.status == 'cancelling'
    assert second.cancel_requested_at == first.cancel_requested_at
    assert second.cancel_requested_by == 'a'


def test_the_connection_of_one_project_cannot_claim_another_projects_job(pg, engine) -> None:
    mine, other = make_project(pg), make_project(pg)
    job = start(engine, other)
    result = claim(engine, mine, job.id)
    assert result == repo.ClaimRefused('missing')
