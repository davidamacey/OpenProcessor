"""The heartbeat renews on its own thread and its own connection; Lost and requests set events."""

from __future__ import annotations

import time

import pytest
from fixtures.job_rows import make_project, project_session, sql, start
from sqlalchemy.exc import OperationalError

from src.db.repos import jobs as repo
from src.tasks.runtime import gpu_app
from src.tasks.runtime.heartbeat import JobHeartbeat


pytestmark = pytest.mark.db

TTL = 2
TICK = 0.2
MARGIN = 0.5


@pytest.fixture
def engine(pg_env):
    from src.db.engine import get_sync_engine

    return get_sync_engine()


@pytest.fixture
def claimed(pg, engine):
    project = make_project(pg)
    job = start(engine, project)
    with project_session(engine, project) as session:
        got = repo.claim(session, job.id, 'w:1', TTL)
    assert isinstance(got, repo.Claimed)
    return project, got.job


def heartbeat(project, job, **kwargs) -> JobHeartbeat:
    kwargs.setdefault('tick_s', TICK)
    kwargs.setdefault('ttl_s', TTL)
    kwargs.setdefault('fence_margin_s', MARGIN)
    return JobHeartbeat(job.id, project, job.fencing_token, **kwargs)


@pytest.mark.asyncio
async def test_a_handler_blocking_its_loop_for_three_ttls_keeps_the_lease(pg, claimed) -> None:
    project, job = claimed
    beat = heartbeat(project, job)
    beat.start()
    try:
        time.sleep(3 * TTL)  # blocks this event loop; an asyncio ticker would starve and expire
        [(live,)] = sql(pg, 'SELECT lease_expires_at > now() FROM jobs WHERE id = %s', job.id)
        assert live
        assert beat.may_write()
        assert not beat.lost.is_set()
    finally:
        beat.stop()


def test_a_row_deleted_under_the_heartbeat_is_lost_and_stops_writes(pg, claimed) -> None:
    project, job = claimed
    beat = heartbeat(project, job)
    beat.start()
    try:
        sql(pg, 'DELETE FROM jobs WHERE id = %s', job.id)
        assert beat.lost.wait(5)
        assert not beat.may_write()
    finally:
        beat.stop()


def test_a_takeover_with_a_new_token_is_lost_for_the_old_holder(pg, engine, claimed) -> None:
    project, job = claimed
    sql(pg, 'UPDATE jobs SET fencing_token = fencing_token + 1 WHERE id = %s', job.id)
    beat = heartbeat(project, job)
    beat.start()
    try:
        assert beat.lost.wait(5)
    finally:
        beat.stop()


def test_stop_ends_the_thread_and_the_renewals(pg, claimed) -> None:
    project, job = claimed
    beat = heartbeat(project, job)
    beat.start()
    time.sleep(TICK * 3)
    beat.stop()
    assert beat._thread is not None
    assert not beat._thread.is_alive()
    [(before,)] = sql(pg, 'SELECT heartbeat_at FROM jobs WHERE id = %s', job.id)
    time.sleep(TICK * 4)
    assert sql(pg, 'SELECT heartbeat_at FROM jobs WHERE id = %s', job.id) == [(before,)]


def test_cancel_and_pause_requests_set_their_events(pg, engine, claimed) -> None:
    project, job = claimed
    beat = heartbeat(project, job)
    beat.start()
    try:
        with project_session(engine, project) as session:
            repo.request_pause(session, job.id, job.fencing_token, 'audit')
        assert beat.pause_requested.wait(5)
        assert not beat.cancel_requested.is_set()
        with project_session(engine, project) as session:
            repo.cancel(session, job.id, 'tester')
        assert beat.cancel_requested.wait(5)
    finally:
        beat.stop()


def test_each_beat_refreshes_the_gpu_inventory(claimed, monkeypatch) -> None:
    project, job = claimed
    refreshed: list[int] = []
    monkeypatch.setattr(gpu_app, 'refresh_inventory', lambda: refreshed.append(1))
    beat = heartbeat(project, job)
    beat.start()
    try:
        deadline = time.monotonic() + 5
        while len(refreshed) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        beat.stop()
    assert len(refreshed) >= 2


def test_without_a_database_the_holder_fences_itself_before_the_ttl(claimed) -> None:
    project, job = claimed

    def broken() -> None:
        raise OperationalError('select 1', {}, OSError('database is down'))

    beat = heartbeat(project, job, session_factory=broken)
    started = time.monotonic()
    beat.start()
    try:
        assert beat.lost.wait(TTL)
        assert time.monotonic() - started < TTL
        assert not beat.may_write()
    finally:
        beat.stop()


def test_may_write_goes_false_when_the_last_good_renew_is_too_old(claimed) -> None:
    project, job = claimed
    beat = heartbeat(project, job)
    assert beat.may_write()  # the claim counts as the last good renew
    time.sleep(TTL - MARGIN + 0.1)
    assert not beat.may_write()
