"""The exact key is a constraint: concurrent ingests of one content store it once (#358)."""

from __future__ import annotations

import asyncio

import pytest

from src.db.repos import image_contents as repo
from src.db.session import async_session_factory, project_tx

from .conftest import make_project


pytestmark = [pytest.mark.db, pytest.mark.asyncio]

SHA = 'a' * 64


async def _claim(project, sha, image_id):
    async with async_session_factory()() as session, project_tx(session, project):
        return await repo.claim_content(session, project, sha, image_id, 10)


async def test_two_concurrent_claims_store_the_content_once(pg_registry) -> None:
    project = await make_project()
    results = await asyncio.gather(*[_claim(project, SHA, f'img{i}') for i in range(8)])

    winners = [r for r in results if r is None]
    holders = {r for r in results if r is not None}
    assert len(winners) == 1
    assert len(holders) == 1
    async with async_session_factory()() as session, project_tx(session, project):
        stored = await repo.find_contents(session, project, [SHA, 'b' * 64])
    assert stored == {SHA: holders.pop()}


async def test_release_frees_the_content_for_a_new_image(pg_registry) -> None:
    project = await make_project()
    assert await _claim(project, SHA, 'first') is None
    assert await _claim(project, SHA, 'second') == 'first'
    async with async_session_factory()() as session, project_tx(session, project):
        assert await repo.release_contents(session, project, ['first']) == 1
    assert await _claim(project, SHA, 'second') is None


async def test_another_project_does_not_see_the_content(pg_registry) -> None:
    one, other = await make_project(), await make_project()
    assert await _claim(one, SHA, 'img') is None
    assert await _claim(other, SHA, 'img') is None
    async with async_session_factory()() as session, project_tx(session, other):
        assert await repo.find_contents(session, other, [SHA]) == {SHA: 'img'}
