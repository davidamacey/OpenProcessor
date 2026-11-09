"""Fixtures of the near-duplicate tests: a real project row (with its seeded profile) and a
project-scoped session factory."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
import pytest_asyncio
from fixtures.job_fixtures import job_env, kinds, producer, queue_hooks, wakes

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import new_project_record
from src.db.session import async_session_factory, deployment_tx, project_tx
from src.services.projects.records import new_project_for
from src.services.projects.store import create_with_default_spaces


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession


async def make_project(slug: str | None = None) -> uuid.UUID:
    """A project created the way production creates one (default spaces, seeded profile)."""
    name = slug or f'nd-{uuid.uuid4().hex[:10]}'
    async with async_session_factory()() as session, deployment_tx(session, 'test create'):
        row = await create_with_default_spaces(session, new_project_for(name, display_name=name))
    return row.id


def scope_for(project: uuid.UUID) -> Callable[[], AbstractAsyncContextManager[AsyncSession]]:
    @asynccontextmanager
    async def scope() -> AsyncIterator[AsyncSession]:
        async with async_session_factory()() as session, project_tx(session, project):
            yield session

    return scope


@pytest_asyncio.fixture
async def project(pg_registry: Any) -> uuid.UUID:
    return await make_project()


@pytest.fixture
def bound(project: uuid.UUID, pg_registry: Any) -> Iterator[None]:
    """The test's project bound the way a job binds it, for code that resolves the embedding
    spaces of the current project (the OCC vector excludes)."""
    with psycopg.connect(pg_registry.conninfo('super'), autocommit=True) as conn:
        slug = conn.execute('SELECT slug FROM projects WHERE id = %s', (project,)).fetchone()[0]  # type: ignore[index]
    with bind_project(new_project_record(slug, base_curation_config())):
        yield


__all__ = ['job_env', 'kinds', 'producer', 'queue_hooks', 'wakes']
