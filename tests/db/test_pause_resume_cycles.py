"""A pause, a resume or a yield never spends an attempt (research gap G-1, 02b section 1).

The claim tests ``retry_count < max_attempts``; ``retry_count`` grows only when an attempt ends
``interrupted``, ``lost`` or ``time_limit``. ``attempt`` keeps numbering the ``job_attempts`` rows.
"""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, project_session, sql, start

from src.db.repos import jobs as repo


pytestmark = pytest.mark.db

TTL = 120


@pytest.fixture
def engine(pg):
    engine = pg.engine('app')
    yield engine
    engine.dispose()


def claim(engine, project, job_id):
    with project_session(engine, project) as session:
        return repo.claim(session, job_id, 'w:1', TTL)


def test_three_pause_and_resume_cycles_then_the_fourth_claim_runs(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    for _ in range(3):
        running = claim(engine, project, job.id)
        assert isinstance(running, repo.Claimed)
        with project_session(engine, project) as session:
            assert repo.request_pause(session, job.id, running.job.fencing_token, 'audit')
            paused = repo.pause_running(session, job.id, running.job.fencing_token)
        assert paused is not None
        assert paused.status == 'paused'
        assert paused.retry_count == 0
        with project_session(engine, project) as session:
            repo.resume(session, job.id)
    fourth = claim(engine, project, job.id)
    assert isinstance(fourth, repo.Claimed)
    assert fourth.job.attempt == 4
    assert fourth.job.retry_count == 0
    outcomes = sql(
        pg, 'SELECT outcome FROM job_attempts WHERE job_id = %s ORDER BY attempt', job.id
    )
    assert outcomes == [('paused',)] * 3 + [(None,)]


def test_yields_never_spend_an_attempt(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    for _ in range(5):
        running = claim(engine, project, job.id)
        assert isinstance(running, repo.Claimed)
        with project_session(engine, project) as session:
            yielded = repo.queue_yield(session, job.id, running.job.fencing_token)
        assert yielded is not None
        assert (yielded.status, yielded.retry_count, yielded.last_dispatched_at) == (
            'queued',
            0,
            None,
        )
    assert isinstance(claim(engine, project, job.id), repo.Claimed)


def test_a_job_that_crashes_three_times_still_ends_attempts_exhausted(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project, auto_resume=True)
    for crash in range(3):
        running = claim(engine, project, job.id)
        assert isinstance(running, repo.Claimed)
        with project_session(engine, project) as session:
            moved = repo.interrupt(session, job.id, running.job.fencing_token)
        assert moved is not None
        assert moved.retry_count == crash + 1
        with project_session(engine, project) as session:
            repo.resume(session, job.id)
    refused = claim(engine, project, job.id)
    assert refused == repo.ClaimRefused('exhausted', 'queued')


def test_pauses_and_crashes_are_counted_apart(pg, engine) -> None:
    project = make_project(pg)
    job = start(engine, project)
    running = claim(engine, project, job.id)
    with project_session(engine, project) as session:
        repo.request_pause(session, job.id, running.job.fencing_token, 'user')
        repo.pause_running(session, job.id, running.job.fencing_token)
        repo.resume(session, job.id)
    running = claim(engine, project, job.id)
    with project_session(engine, project) as session:
        moved = repo.interrupt(session, job.id, running.job.fencing_token)
    assert moved is not None
    assert (moved.attempt, moved.retry_count) == (2, 1)
