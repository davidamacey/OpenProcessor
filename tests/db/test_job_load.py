"""The upgrade's job load keeps ids and clocks, and the rows read back through the normal path."""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from fixtures.job_rows import deployment_session, make_project
from sqlalchemy.exc import IntegrityError

from src.db.repos import jobs as repo, jobs_load


pytestmark = pytest.mark.db

CREATED = dt.datetime(2026, 1, 5, 10, 0, tzinfo=dt.UTC)
FINISHED = dt.datetime(2026, 1, 5, 11, 0, tzinfo=dt.UTC)


@pytest.fixture
def engine(pg):
    engine = pg.engine('app', pool_size=5)
    yield engine
    engine.dispose()


def _load(project: uuid.UUID | None, **overrides) -> jobs_load.JobLoad:
    fields = {
        'id': uuid.uuid4(),
        'project_id': project,
        'kind': 'noop',
        'queue': 'jobs',
        'status': 'completed',
        'max_queue_wait': dt.timedelta(hours=1),
        'requested_by': 'upgrade_from_060',
        'created_at': CREATED,
        'updated_at': FINISHED,
        'params': {'a': 1},
        'progress': {'done': 3},
        'result': {'ok': True},
        'started_at': CREATED,
        'finished_at': FINISHED,
    }
    fields.update(overrides)
    return jobs_load.JobLoad(**fields)


def test_a_loaded_job_reads_back_with_its_id_and_clocks(pg, engine) -> None:
    project = make_project(pg)
    done = _load(project)
    with deployment_session(engine) as session:
        assert jobs_load.load_jobs(session, [done]) == 1
    with deployment_session(engine) as session:
        row = repo.get(session, done.id)
    assert row is not None
    assert (row.id, row.project_id, row.status) == (done.id, project, 'completed')
    assert (row.created_at, row.started_at, row.finished_at) == (CREATED, CREATED, FINISHED)
    assert (row.params, row.progress, row.result) == ({'a': 1}, {'done': 3}, {'ok': True})
    assert (row.attempt, row.fencing_token, row.requested_by) == (0, 0, 'upgrade_from_060')


def test_a_deployment_wide_job_has_no_project(pg, engine) -> None:
    job = _load(None, status='failed', error_code='x', error_message='boom')
    with deployment_session(engine) as session:
        jobs_load.load_jobs(session, [job])
    with deployment_session(engine) as session:
        row = repo.get(session, job.id)
    assert row is not None
    assert (row.project_id, row.error_code) == (None, 'x')


def test_a_repeated_id_raises(pg, engine) -> None:
    job = _load(None)
    with deployment_session(engine) as session:
        jobs_load.load_jobs(session, [job])
    with pytest.raises(IntegrityError), deployment_session(engine) as session:
        jobs_load.load_jobs(session, [job])
