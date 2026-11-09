"""The region text repository against a real PostgreSQL: open runs, replaced groups, scope."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
import pytest_asyncio
from _research_seed import make_profile, make_project
from sqlalchemy.exc import IntegrityError

from src.db.engine import dispose_engines
from src.db.repos import region_text as repo
from src.db.repos.region_text import GroupInput, MemberInput
from src.db.session import async_session_factory, project_tx


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = [pytest.mark.db, pytest.mark.asyncio, pytest.mark.usefixtures('bound_async_engine')]


@pytest_asyncio.fixture
async def repo_env(pg_env):
    """``pg_env`` plus an awaited engine disposal, so no pool outlives its event loop."""
    yield
    await dispose_engines()


@pytest.fixture
def world(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        a = make_project(conn, f'rtr-{uuid.uuid4().hex[:8]}')
        b = make_project(conn, f'rtr-{uuid.uuid4().hex[:8]}')
        return {'a': a, 'b': b, 'profile': make_profile(conn, a, 'region_profile')}


def _group(*boxes: str, item: str = 'i1', **kw) -> GroupInput:
    return GroupInput(
        basis='near_dup',
        method='best_frame',
        n_frames=len(boxes),
        n_agree=len(boxes),
        tier='confident',
        members=[MemberInput(item, box, 1.0, True) for box in boxes],
        **kw,
    )


async def _tx(project, fn, *args, **kw):
    async with async_session_factory()() as session, project_tx(session, project):
        return await fn(session, project, *args, **kw)


async def _open(project, **kw):
    kw = {'trigger': 'region_pass', 'reader': 'rec', 'reader_version': 'v1', 'created_by': 't'} | kw
    return await _tx(project, repo.open_run, **kw)


async def test_a_second_open_pass_for_one_configuration_is_a_unique_violation(
    repo_env: None, world
) -> None:
    profile = world['profile']
    first = await _open(world['a'], profile=profile)
    with pytest.raises(IntegrityError, match='uq_region_text_runs_open_pass'):
        await _open(world['a'], profile=profile)
    await _open(world['a'], profile=profile, reader_version='v2')
    await _open(world['a'], profile=profile, trigger='reread')
    assert await _tx(world['a'], repo.finish_run, first, {'boxes_read': 3}) is True
    assert await _tx(world['a'], repo.finish_run, first, {}) is False
    await _open(world['a'], profile=profile)


async def test_finish_run_is_scoped_to_its_project(repo_env: None, world) -> None:
    run = await _open(world['a'])
    assert await _tx(world['b'], repo.finish_run, run, {}) is False
    assert await _tx(world['a'], repo.finish_run, run, {}) is True


async def test_groups_are_read_back_after_they_are_written(repo_env: None, world) -> None:
    run = await _open(world['a'], trigger='consensus')
    ids = await _tx(
        world['a'],
        repo.replace_groups,
        run,
        [_group('b1', 'b2', jurisdiction='synthetic-state', jurisdiction_conf=0.5), _group('b3')],
    )
    found = await _tx(world['a'], repo.groups_for_boxes, [('i1', 'b1'), ('i1', 'b3'), ('i1', 'zz')])
    assert set(found) == {('i1', 'b1'), ('i1', 'b3')}
    assert found[('i1', 'b1')].id == ids[0]
    assert (found[('i1', 'b1')].jurisdiction, found[('i1', 'b1')].jurisdiction_conf) == (
        'synthetic-state',
        0.5,
    )
    assert found[('i1', 'b3')].jurisdiction is None
    members = await _tx(world['a'], repo.members_of, ids[0])
    assert [(m.box_id, m.weight, m.agrees) for m in members] == [
        ('b1', 1.0, True),
        ('b2', 1.0, True),
    ]


async def test_a_recompute_replaces_the_old_group_in_one_transaction(repo_env: None, world) -> None:
    run = await _open(world['a'], trigger='consensus')
    (old,) = await _tx(world['a'], repo.replace_groups, run, [_group('c1', 'c2', 'c3')])
    (new,) = await _tx(world['a'], repo.replace_groups, run, [_group('c1', 'c2')])
    found = await _tx(world['a'], repo.groups_for_boxes, [('i1', 'c1'), ('i1', 'c2'), ('i1', 'c3')])
    assert {k: g.id for k, g in found.items()} == {('i1', 'c1'): new, ('i1', 'c2'): new}
    assert old not in {g.id for g in found.values()}
    assert await _tx(world['a'], repo.members_of, old) == []


async def test_a_failed_replace_keeps_the_old_group(repo_env: None, world) -> None:
    run = await _open(world['a'], trigger='consensus')
    (old,) = await _tx(world['a'], repo.replace_groups, run, [_group('d1')])
    with pytest.raises(IntegrityError):
        await _tx(world['a'], repo.replace_groups, run, [_group('d1'), _group('d1')])
    found = await _tx(world['a'], repo.groups_for_boxes, [('i1', 'd1')])
    assert found[('i1', 'd1')].id == old


async def test_another_project_never_sees_or_replaces_a_group(repo_env: None, world) -> None:
    run_a = await _open(world['a'], trigger='consensus')
    (group,) = await _tx(world['a'], repo.replace_groups, run_a, [_group('e1')])
    assert await _tx(world['b'], repo.groups_for_boxes, [('i1', 'e1')]) == {}
    run_b = await _open(world['b'], trigger='consensus')
    await _tx(world['b'], repo.replace_groups, run_b, [_group('e1')])
    found = await _tx(world['a'], repo.groups_for_boxes, [('i1', 'e1')])
    assert found[('i1', 'e1')].id == group
