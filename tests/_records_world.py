"""A migrated database with the bound ``default`` project backed by a real ``projects`` row.

Load the fixtures with ``pytest_plugins = ['_records_world']`` in a test module;
the module must carry ``pytestmark = pytest.mark.db``. The conftest registry stub serves the same
``DEFAULT_RECORD`` object that every request binds, so setting its ``id`` makes a route see the
project's uuid.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
import pytest_asyncio


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession


def make_project_row(pg, slug: str) -> uuid.UUID:
    project = uuid.uuid4()
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
        conn.execute(
            'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
            "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, %s, 'active', %s, %s, %s, %s, 'test')",
            (project, slug, slug, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
        )
    return project


@pytest.fixture
def world(
    pg_env, bound_async_engine, api_async_engine, monkeypatch: pytest.MonkeyPatch
) -> uuid.UUID:
    """The bound project's id, with its row in the database.

    Services called from the test use the bound engine, the app driven through ``TestClient`` the
    API engine double.
    """
    from src.config.project_context import current_project

    project = make_project_row(pg_env, f'default-{uuid.uuid4().hex[:8]}')
    monkeypatch.setattr(current_project().record, 'id', project, raising=False)
    return project


@pytest_asyncio.fixture
async def session(world) -> AsyncIterator[AsyncSession]:
    from src.db.session import async_session_factory

    async with async_session_factory()() as s:
        yield s
