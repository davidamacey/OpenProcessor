"""A flip queues the cluster refresh and the near-duplicate re-score for its target."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fixtures.job_rows import sql

from src.db.repos import embedding_spaces as repo
from src.db.session import async_session_factory, project_tx
from src.domain.embedding_spaces import Target
from src.services.embedding_spaces.dependents import queue_dependents
from src.services.embedding_spaces.flip import promote
from tests.embedding_spaces._seed import make_project
from tests.embedding_spaces.test_spaces import _default_primary, _ready


pytestmark = [pytest.mark.db, pytest.mark.asyncio, pytest.mark.usefixtures('bound_async_engine')]


def kinds_of(pg, project) -> list[str]:
    rows = sql(pg, 'SELECT kind FROM jobs WHERE project_id = %s ORDER BY kind', project)
    return [r[0] for r in rows]


async def _flip(project, target=Target.ITEM):
    async with async_session_factory()() as session:
        await _default_primary(session, project, target)
        row = await _ready(session, project, target=target)
        result = await promote(session, project, target=target, key=row.space_key, force=True)
        return await queue_dependents(session, project, result.primary, requested_by='test')


async def test_an_item_flip_queues_cluster_and_near_dup(
    pg, pg_env, catalog, job_env, producer, wakes
):
    project, _ = make_project(pg_env.conninfo('super'))
    assert await _flip(project) == ['cluster_run', 'near_dup_score']
    assert kinds_of(pg, project) == ['cluster_run', 'near_dup_score']


async def test_a_frame_flip_queues_only_near_dup(pg, pg_env, catalog, job_env, producer, wakes):
    project, _ = make_project(pg_env.conninfo('super'))
    assert await _flip(project, Target.FRAME) == ['near_dup_score']


async def test_a_flip_with_no_dependents_queues_nothing(
    pg, pg_env, catalog, job_env, producer, wakes
):
    project, _ = make_project(pg_env.conninfo('super'))
    async with async_session_factory()() as session:
        await _default_primary(session, project)
        row = await _ready(session, project)
        text_space = replace(row, target=Target.ITEM_TEXT)
        assert await queue_dependents(session, project, text_space, requested_by='test') == []
    assert kinds_of(pg, project) == []


async def test_an_active_job_is_not_an_error(pg, pg_env, catalog, job_env, producer, wakes):
    project, _ = make_project(pg_env.conninfo('super'))
    await _flip(project)
    async with async_session_factory()() as session:
        async with project_tx(session, project):
            primary = await repo.get_primary_space(session, project, Target.ITEM)
        assert await queue_dependents(session, project, primary, requested_by='test') == []
    assert kinds_of(pg, project) == ['cluster_run', 'near_dup_score']
