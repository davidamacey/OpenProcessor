"""Rows and sessions for the job tests: a project, scoped sessions, a ``NewJob`` builder."""

from __future__ import annotations

import datetime as dt
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import psycopg
from sqlalchemy.orm import Session

from src.db.repos import jobs as repo
from src.db.session import sync_deployment_tx, sync_project_tx


if TYPE_CHECKING:
    from collections.abc import Iterator

    from fixtures.postgres import PgDatabase
    from sqlalchemy.engine import Engine


def make_project(pg: PgDatabase, slug: str | None = None, status: str = 'active') -> uuid.UUID:
    slug = slug or f'p{uuid.uuid4().hex[:8]}'
    project = uuid.uuid4()
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
        conn.execute(
            'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
            'model_prefix, mlflow_experiment, created_by) '
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'test')",
            (project, slug, slug, status, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
        )
    return project


def sql(pg: PgDatabase, statement: str, *params: Any) -> list[tuple[Any, ...]]:
    """Run ``statement`` as the superuser (past row-level security); rows, or [] for no result."""
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        cursor = conn.execute(statement, params)
        return cursor.fetchall() if cursor.description else []


@contextmanager
def project_session(engine: Engine, project: uuid.UUID) -> Iterator[Session]:
    with Session(engine, expire_on_commit=False) as session, sync_project_tx(session, project):
        yield session


@contextmanager
def deployment_session(engine: Engine) -> Iterator[Session]:
    with (
        Session(engine, expire_on_commit=False) as session,
        sync_deployment_tx(session, 'test'),
    ):
        yield session


def new_job(project: uuid.UUID | None, kind: str = 'noop', **overrides: Any) -> repo.NewJob:
    fields: dict[str, Any] = {
        'kind': kind,
        'project_id': project,
        'queue': 'jobs',
        'max_queue_wait': dt.timedelta(hours=1),
        'requested_by': 'test',
    }
    fields.update(overrides)
    return repo.NewJob(**fields)


def start(engine: Engine, project: uuid.UUID, kind: str = 'noop', **overrides: Any) -> repo.JobRow:
    """Insert one queued job and commit."""
    with project_session(engine, project) as session:
        return repo.insert_queued(session, new_job(project, kind, **overrides)).job
