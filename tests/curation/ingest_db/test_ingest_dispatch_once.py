"""``dispatch_once`` over real source rows: what it starts, what it stamps, and the one-run and
one-poll-per-source rules the job kinds enforce."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from fixtures.job_rows import insert_row, make_project, project_session, sql

from curation.ingest_db.app_fixtures import record_for
from src.config.project_context import bind_project
from src.db.repos import ingest_sources as repo
from src.db.session import async_session_factory
from src.services.curation.bulk_ingest.dispatcher import Action, dispatch_once
from src.services.curation.bulk_ingest.sources import IngestSource, ScheduleTrigger, WatchTrigger


pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures('bound_async_engine', 'job_env', 'producer', 'wakes'),
]
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC).timestamp()


@pytest.fixture
def project(pg):
    pid = make_project(pg)
    slug = sql(pg, 'SELECT slug FROM projects WHERE id = %s', pid)[0][0]
    with bind_project(record_for(pid, slug)):
        yield pid


def add_source(pg, pid: uuid.UUID, **over: Any) -> uuid.UUID:
    source = IngestSource(
        name=f's{uuid.uuid4().hex[:8]}', backend='s3', bucket='b', prefix='p/', **over
    )
    with project_session(pg.engine('app'), pid) as session:
        return repo.insert(
            session,
            pid,
            name=source.name,
            kind='s3',
            config=source.config(),
            schedule=source.triggers(),
            enabled=source.triggered,
        ).id


def pend(pg, pid: uuid.UUID, source_id: uuid.UUID, *keys: str) -> None:
    with project_session(pg.engine('app'), pid) as session:
        repo.enqueue(session, source_id, [f's3://b/{k}#etag=e' for k in keys])


def tick(pid: uuid.UUID, now: float = NOW) -> list[Action]:
    async def go() -> list[Action]:
        async with async_session_factory()() as session:
            return await dispatch_once(session, pid, now=now)

    return asyncio.run(go())


def state(pg, source_id: uuid.UUID) -> dict[str, Any]:
    return sql(pg, 'SELECT scan_state FROM ingest_sources WHERE id = %s', source_id)[0][0]


def kinds_started(pg, pid: uuid.UUID) -> list[str]:
    rows = sql(pg, 'SELECT kind FROM jobs WHERE project_id = %s ORDER BY created_at', pid)
    return [r[0] for r in rows]


def test_a_source_without_triggers_is_never_acted_on(pg, project) -> None:
    source = add_source(pg, project)
    pend(pg, project, source, 'a.jpg')
    assert tick(project) == []
    assert kinds_started(pg, project) == []


def test_a_due_watcher_starts_a_poll_job_and_stamps_it_and_is_not_polled_again_while_live(
    pg, project
) -> None:
    source = add_source(pg, project, watch=WatchTrigger(True, 60))

    first = tick(project)

    assert first == [Action('poll', str(source))]
    assert kinds_started(pg, project) == ['source_poll']
    assert state(pg, source)['last_poll_started_at'] == NOW
    # inside the interval nothing is due; past it the poll is due but the last one is still live
    assert tick(project, NOW + 30) == []
    assert tick(project, NOW + 61) == []
    assert kinds_started(pg, project) == ['source_poll']


def test_a_schedule_counts_from_the_moment_it_was_first_seen(pg, project) -> None:
    source = add_source(pg, project, schedule=ScheduleTrigger(True, '*/5 * * * *'))

    assert tick(project) == []  # not fired on sight
    stamped = state(pg, source)['last_fire_at']
    assert datetime.fromisoformat(stamped).timestamp() == NOW

    assert tick(project, NOW + 301) == [Action('run_scan', str(source))]
    run = sql(
        pg, "SELECT params FROM jobs WHERE kind = 'bulk_ingest_run' AND project_id = %s", project
    )
    assert [(r[0]['source_id'], r[0]['scan']) for r in run] == [(str(source), True)]
    assert state(pg, source)['last_run_id']


def test_pending_rows_of_a_triggered_source_are_drained_and_only_one_run_holds_the_project(
    pg, project
) -> None:
    first = add_source(pg, project, webhook_enabled=True)
    second = add_source(pg, project, webhook_enabled=True)
    pend(pg, project, first, 'a.jpg')
    pend(pg, project, second, 'b.jpg')

    taken = tick(project)

    assert [a.kind for a in taken] == ['drain']  # the second finds the project's run slot taken
    assert kinds_started(pg, project) == ['bulk_ingest_run']
    ((params,),) = sql(
        pg, "SELECT params FROM jobs WHERE kind = 'bulk_ingest_run' AND project_id = %s", project
    )
    assert params['scan'] is False


def finished_run(pg, pid: uuid.UUID, source: uuid.UUID, status: str, finished_at: float) -> None:
    """The source's last run, finished as ``status`` at ``finished_at``."""
    job = insert_row(
        pg,
        pid,
        status,
        kind='bulk_ingest_run',
        finished_at=datetime.fromtimestamp(finished_at, UTC),
        **({'error_code': 'x', 'error_message': 'x'} if status == 'failed' else {}),
    )
    with project_session(pg.engine('app'), pid) as session:
        repo.state_update(session, source, {'last_run_id': str(job)})


def test_a_cancelled_run_is_not_restarted_by_the_dispatcher(pg, project) -> None:
    source = add_source(pg, project, webhook_enabled=True)
    pend(pg, project, source, 'a.jpg')
    finished_run(pg, project, source, 'cancelled', NOW - 3600)
    assert tick(project) == []


def test_a_failed_run_is_held_for_five_minutes_and_then_drained_again(pg, project) -> None:
    source = add_source(pg, project, webhook_enabled=True)
    pend(pg, project, source, 'a.jpg')
    finished_run(pg, project, source, 'failed', NOW - 60)
    assert tick(project) == []
    assert [a.kind for a in tick(project, NOW + 300)] == ['drain']


def test_nothing_happens_during_backoff(pg, project) -> None:
    source = add_source(pg, project, watch=WatchTrigger(True, 10), webhook_enabled=True)
    pend(pg, project, source, 'a.jpg')
    with project_session(pg.engine('app'), project) as session:
        repo.state_update(session, source, {'backoff_until': NOW + 30})

    assert tick(project) == []
    assert [a.kind for a in tick(project, NOW + 31)] == ['poll']


def test_one_shot_sources_are_left_to_the_person_who_started_them(pg, project) -> None:
    source = add_source(pg, project, webhook_enabled=True, one_shot=True)
    pend(pg, project, source, 'a.jpg')
    assert tick(project) == []
