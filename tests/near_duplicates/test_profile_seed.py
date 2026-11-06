"""A new project starts with the code-default near-duplicate profile active; reads fail closed."""

from __future__ import annotations

import uuid

import pytest

from src.db.session import async_session_factory, deployment_tx
from src.schemas.curation.near_duplicates import NearDupProfileBody
from src.services.near_duplicates.profiles import NoActiveProfileError, active_profile
from src.services.projects.records import new_project_for
from src.services.projects.store import create_with_default_spaces


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def test_project_creation_seeds_and_activates_revision_one(pg_registry) -> None:
    async with async_session_factory()() as session, deployment_tx(session, 'test create'):
        row = await create_with_default_spaces(
            session, new_project_for('nd-seed', display_name='S')
        )
    async with async_session_factory()() as session, deployment_tx(session, 'test read'):
        profile = await active_profile(session, row.id)
    assert profile.revision == 1
    assert profile.body == NearDupProfileBody()


async def test_a_project_without_an_active_profile_refuses(pg_registry) -> None:
    from src.db.repos import projects as repo

    async with async_session_factory()() as session, deployment_tx(session, 'test create'):
        row = await repo.create_project(session, new_project_for('nd-bare', display_name='B'))
    async with async_session_factory()() as session, deployment_tx(session, 'test read'):
        with pytest.raises(NoActiveProfileError):
            await active_profile(session, row.id)
    assert isinstance(row.id, uuid.UUID)
