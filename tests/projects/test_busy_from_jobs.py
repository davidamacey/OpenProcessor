"""Is a project busy: one query on its active ``jobs`` rows (3A-8)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid

import pytest
from fixtures.job_rows import insert_row, make_project, sql

from src.config.curation_accessors import base_curation_config
from src.config.projects import ProjectRecord, resources_for_new
from src.services.projects.busy import running_jobs


pytestmark = pytest.mark.db


@pytest.fixture
def world(pg, pg_env):
    yield pg
    import asyncio

    from src.db.engine import dispose_engines

    asyncio.run(dispose_engines())


def record_for(project: uuid.UUID, slug: str) -> ProjectRecord:
    now = dt.datetime.now(dt.UTC).isoformat()
    record = ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )
    return dataclasses.replace(record, id=project)


def make(world, slug=None):
    slug = slug or f'p{uuid.uuid4().hex[:8]}'
    project = make_project(world, slug)
    return project, record_for(project, slug)


def test_an_idle_project_is_not_busy(world) -> None:
    _project, record = make(world)
    assert running_jobs(record) == []


@pytest.mark.parametrize(
    ('status', 'busy'),
    [
        ('queued', True),
        ('running', True),
        ('paused', True),
        ('cancelling', True),
        ('interrupted', False),
        ('completed', False),
        ('completed_with_errors', False),
        ('failed', False),
        ('cancelled', False),
    ],
)
def test_only_an_active_status_makes_a_project_busy(world, status, busy) -> None:
    project, record = make(world)
    job = insert_row(world, project, status, kind='reprocess')
    found = running_jobs(record)
    assert [(j.kind, j.job_id) for j in found] == ([('reprocess', str(job))] if busy else [])


def test_a_job_makes_only_its_own_project_busy(world) -> None:
    mine, record = make(world)
    _other, other_record = make(world)
    insert_row(world, mine, 'running')
    assert len(running_jobs(record)) == 1
    assert running_jobs(other_record) == []


def test_a_running_job_reports_when_it_started(world) -> None:
    project, record = make(world)
    started = dt.datetime(2026, 11, 1, 12, 0, tzinfo=dt.UTC)
    insert_row(world, project, 'running', started_at=started)
    [job] = running_jobs(record)
    assert dt.datetime.fromisoformat(job.started_at) == started


def test_a_queued_job_has_no_start_time(world) -> None:
    project, record = make(world)
    insert_row(world, project, 'queued')
    [job] = running_jobs(record)
    assert job.started_at is None


def test_a_combine_run_makes_its_source_projects_busy(world) -> None:
    source, source_record = make(world)
    target, _ = make(world)
    job = insert_row(world, target, 'running', kind='combine')
    sql(
        world,
        'INSERT INTO combine_runs (job_id, target_project_id, holdout_mode, class_mapping) '
        "VALUES (%s, %s, 'union', '{}')",
        job,
        target,
    )
    sql(
        world,
        'INSERT INTO combine_sources (job_id, position, source_project_id, source_slug) '
        'VALUES (%s, 0, %s, %s)',
        job,
        source,
        source_record.slug,
    )
    assert [(j.kind, j.job_id) for j in running_jobs(source_record)] == [('combine', str(job))]


def test_a_record_with_no_database_id_fails_closed(world) -> None:
    from src.config.project_context import ProjectHasNoIdError

    _project, record = make(world)
    with pytest.raises(ProjectHasNoIdError):
        running_jobs(dataclasses.replace(record, id=None))
