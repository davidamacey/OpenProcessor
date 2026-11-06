"""The epoch poll: one read of ``projects.config_epoch`` and the global counter, no OpenSearch."""

from __future__ import annotations

import uuid

import pytest
from _config_fixtures import make_project

from src.db.repos import config as repo
from src.db.session import async_session_factory, project_tx
from src.services.config_store import store_poll


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.fixture
def noted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(
        'src.services.embedding_spaces.resolve.note_config_epoch',
        lambda slug, epoch: seen.append((slug, epoch)),
    )
    return seen


@pytest.mark.asyncio
async def test_a_new_epoch_is_reported_once_and_tells_the_space_resolver(pg_env, pg, noted) -> None:
    slug = f'poll-{uuid.uuid4().hex[:12]}'
    project = make_project(pg, slug)
    known: dict[str, int | None] = {}
    first = await store_poll.poll_config_epochs(known)
    assert first[slug] == 0
    known.update(first)
    assert (slug, 0) in noted

    async with async_session_factory()() as session, project_tx(session, project):
        await repo.bump_epoch(session, project)  # an activation in another session
    moved = await store_poll.poll_config_epochs(known)
    assert moved == {slug: 1}
    assert (slug, 1) in noted


@pytest.mark.asyncio
async def test_an_unchanged_epoch_reloads_nothing(pg_env, pg, noted) -> None:
    slug = f'steady-{uuid.uuid4().hex[:12]}'
    make_project(pg, slug)
    known: dict[str, int | None] = await store_poll.poll_config_epochs({})
    assert await store_poll.poll_config_epochs(known) == {}
    assert noted.count((slug, 0)) == 2  # the resolver is told on every read


@pytest.mark.asyncio
async def test_the_global_epoch_is_reported_under_its_own_key(pg_env, pg, noted) -> None:
    from src.db.session import deployment_tx

    known = await store_poll.poll_config_epochs({})
    async with async_session_factory()() as session, deployment_tx(session, 'poll test'):
        await repo.bump_epoch(session, None)
    moved = await store_poll.poll_config_epochs(known)
    assert store_poll.GLOBAL_EPOCH_KEY in moved
