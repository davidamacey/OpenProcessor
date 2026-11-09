"""``op.periodic.ingest_dispatch``: one beat tick runs ``dispatch_once`` for every active project.

It replaces the API-lifespan asyncio loop. A project whose advisory lock another tick holds is
skipped for this tick, one project's failure does not stop the next, and the tick still fails so
the failure is on the task.
"""

from __future__ import annotations

import uuid
from typing import Any

import psycopg
import pytest
from fixtures.job_rows import make_project, project_session, sql

from src.config.advisory_locks import INGEST_DISPATCH
from src.db.repos import ingest_sources as repo
from src.services.curation.bulk_ingest.sources import IngestSource
from src.services.periodic import ingest_dispatch
from src.services.periodic.runner import PeriodicTickError, run_periodic


pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures('pg_env', 'job_env', 'producer', 'wakes'),
]


def add_pending_webhook_source(pg, pid: uuid.UUID) -> uuid.UUID:
    source = IngestSource(
        name=f's{uuid.uuid4().hex[:8]}', backend='s3', bucket='b', prefix='p/', webhook_enabled=True
    )
    with project_session(pg.engine('app'), pid) as session:
        row = repo.insert(
            session,
            pid,
            name=source.name,
            kind='s3',
            config=source.config(),
            schedule=source.triggers(),
            enabled=source.triggered,
        )
        repo.enqueue(session, row.id, ['s3://b/a.jpg#etag=e'])
        return row.id


def runs_of(pg, pid: uuid.UUID) -> list[str]:
    return [r[0] for r in sql(pg, 'SELECT kind FROM jobs WHERE project_id = %s', pid)]


def tick() -> Any:
    return run_periodic('ingest_dispatch', ingest_dispatch.tick())


def test_every_active_project_with_work_gets_its_run(pg) -> None:
    first, second = make_project(pg), make_project(pg)
    add_pending_webhook_source(pg, first)
    add_pending_webhook_source(pg, second)

    report = tick()

    assert sorted(report.done) == sorted(
        r[0] for r in sql(pg, 'SELECT slug FROM projects WHERE id IN (%s, %s)', first, second)
    )
    assert runs_of(pg, first) == ['bulk_ingest_run']
    assert runs_of(pg, second) == ['bulk_ingest_run']


def test_a_paused_or_inactive_project_is_not_visited(pg) -> None:
    paused = make_project(pg)
    archived = make_project(pg, status='archived')
    add_pending_webhook_source(pg, paused)
    add_pending_webhook_source(pg, archived)
    sql(pg, 'UPDATE projects SET paused_at = now(), paused_by = %s WHERE id = %s', 'test', paused)

    tick()

    assert runs_of(pg, paused) == []
    assert runs_of(pg, archived) == []


def test_a_project_whose_lock_is_held_is_skipped_and_the_next_one_still_runs(pg) -> None:
    busy, free = make_project(pg), make_project(pg)
    add_pending_webhook_source(pg, busy)
    add_pending_webhook_source(pg, free)
    ((busy_key,),) = sql(pg, 'SELECT lock_key FROM projects WHERE id = %s', busy)

    with psycopg.connect(pg.conninfo('super'), autocommit=True) as holder:
        holder.execute('SELECT pg_advisory_lock(%s, %s)', (INGEST_DISPATCH, busy_key))
        report = tick()
        holder.execute('SELECT pg_advisory_unlock(%s, %s)', (INGEST_DISPATCH, busy_key))

    assert len(report.skipped_busy) == 1
    assert runs_of(pg, busy) == []
    assert runs_of(pg, free) == ['bulk_ingest_run']


def test_one_projects_failure_does_not_stop_the_next_and_fails_the_tick(
    pg, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.exc import OperationalError

    first, second = make_project(pg, 'aaa-first'), make_project(pg, 'bbb-second')
    add_pending_webhook_source(pg, first)
    add_pending_webhook_source(pg, second)
    real = ingest_dispatch.dispatch_once
    seen: list[uuid.UUID] = []

    async def flaky(session: Any, project: uuid.UUID, **kw: Any) -> Any:
        seen.append(project)
        if project == first:
            raise OperationalError('select', {}, Exception('database went away'))
        return await real(session, project, **kw)

    monkeypatch.setattr(ingest_dispatch, 'dispatch_once', flaky)

    with pytest.raises(PeriodicTickError, match='aaa-first'):
        tick()

    assert [p for p in seen if p in (first, second)] == [first, second]
    assert runs_of(pg, second) == ['bulk_ingest_run']
