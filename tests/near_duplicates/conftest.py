"""Fixtures of the near-duplicate tests: a real project row (with its seeded profile) and a
project-scoped session factory."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest_asyncio
from fixtures.job_fixtures import job_env, kinds, producer, queue_hooks, wakes

from src.db.session import async_session_factory, deployment_tx, project_tx
from src.services.projects.records import new_project_for
from src.services.projects.store import create_with_default_spaces


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
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


__all__ = ['job_env', 'kinds', 'producer', 'queue_hooks', 'wakes']
