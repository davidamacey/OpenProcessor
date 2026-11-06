"""Project markers (region drain, dependency health) with optimistic concurrency on revision."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from _seed import make_project
from sqlalchemy.exc import IntegrityError

from src.db.repos.records import (
    RecordRevisionConflict,
    get_project_state,
    load_project_state,
    modify_project_state,
    put_project_state,
)
from src.db.session import async_session_factory, deployment_tx, project_tx


pytestmark = pytest.mark.db


@pytest.mark.asyncio
async def test_a_marker_reads_back_what_was_written_at_revision_one(pg_env) -> None:
    project = make_project(pg_env, 'state-rw')
    async with async_session_factory()() as session, project_tx(session, project):
        assert await get_project_state(session, project, 'region_drain') is None
        assert (
            await put_project_state(
                session, project, 'region_drain', {'streak': 2}, expected_revision=0
            )
            == 1
        )
        assert (
            await put_project_state(
                session, project, 'region_drain', {'streak': 3}, expected_revision=1
            )
            == 2
        )
        row = await get_project_state(session, project, 'region_drain')
    assert row is not None
    assert (row.value, row.revision) == ({'streak': 3}, 2)


@pytest.mark.asyncio
async def test_a_stale_revision_or_a_second_create_is_a_conflict(pg_env) -> None:
    project = make_project(pg_env, 'state-occ')
    async with async_session_factory()() as session, project_tx(session, project):
        await put_project_state(
            session, project, 'region_dependency_health', {}, expected_revision=0
        )
        with pytest.raises(RecordRevisionConflict):
            await put_project_state(
                session, project, 'region_dependency_health', {}, expected_revision=0
            )
        with pytest.raises(RecordRevisionConflict):
            await put_project_state(
                session, project, 'region_dependency_health', {'x': 1}, expected_revision=5
            )
        row = await get_project_state(session, project, 'region_dependency_health')
    assert row is not None
    assert row.revision == 1


@pytest.mark.asyncio
async def test_concurrent_modifies_lose_no_increment(pg_env) -> None:
    project = make_project(pg_env, 'state-race')

    async def bump() -> None:
        async with async_session_factory()() as session, project_tx(session, project):
            await modify_project_state(
                session,
                project,
                'region_drain',
                lambda v: {'streak': (v or {}).get('streak', 0) + 1},
            )

    await asyncio.gather(*(bump() for _ in range(6)))
    async with async_session_factory()() as session, project_tx(session, project):
        row = await get_project_state(session, project, 'region_drain')
    assert row is not None
    assert row.value == {'streak': 6}
    assert row.revision == 6


@pytest.mark.asyncio
async def test_an_unknown_key_is_refused_before_any_statement(pg_env) -> None:
    project = make_project(pg_env, 'state-key')
    async with async_session_factory()() as session, project_tx(session, project):
        with pytest.raises(ValueError, match='unknown project state key'):
            await get_project_state(session, project, 'last_clustering')


@pytest.mark.asyncio
async def test_the_upgrade_load_keeps_the_revision_and_reads_back(pg_env) -> None:
    project = make_project(pg_env, 'state-load')
    async with async_session_factory()() as session, deployment_tx(session, 'test load'):
        await load_project_state(session, project, 'region_drain', {'streak': 4}, revision=9)
    async with async_session_factory()() as session, project_tx(session, project):
        row = await get_project_state(session, project, 'region_drain')
        assert row is not None
        assert (row.value, row.revision) == ({'streak': 4}, 9)
    async with async_session_factory()() as session:
        with pytest.raises(IntegrityError):
            async with deployment_tx(session, 'again'):
                await load_project_state(session, project, 'region_drain', {}, revision=1)
    with pytest.raises(ValueError, match='revision starts at 1'):
        async with async_session_factory()() as session, deployment_tx(session, 'zero'):
            await load_project_state(session, uuid.uuid4(), 'region_drain', {}, revision=0)
