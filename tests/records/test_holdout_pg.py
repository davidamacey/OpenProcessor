"""Holdout freezes in Postgres: one current row per project, the history kept, the load exact."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid

import pytest
from _seed import make_project
from sqlalchemy.exc import IntegrityError

from src.db.repos.records import (
    HoldoutFreezeLoad,
    RecordNotFoundError,
    current_holdout_freeze,
    freeze_holdout,
    list_holdout_freezes,
    load_holdout_freezes,
)
from src.db.session import async_session_factory, deployment_tx, project_tx


pytestmark = pytest.mark.db


async def _freeze(project: uuid.UUID, who: str, count: int):
    async with async_session_factory()() as session, project_tx(session, project):
        return await freeze_holdout(
            session, project, source='manual', item_count=count, frozen_by=who
        )


@pytest.mark.asyncio
async def test_two_concurrent_freezes_leave_one_current_row_and_both_in_history(pg_env) -> None:
    project = make_project(pg_env, 'hold-race')
    await asyncio.gather(*(_freeze(project, f'u{i}', 10 + i) for i in range(4)))
    async with async_session_factory()() as session, project_tx(session, project):
        history = await list_holdout_freezes(session, project)
        current = await current_holdout_freeze(session, project)
    assert len(history) == 4
    assert [h.is_current for h in history].count(True) == 1
    assert current is not None
    assert current.id == next(h.id for h in history if h.is_current)


@pytest.mark.asyncio
async def test_a_new_freeze_demotes_the_previous_current_and_history_is_newest_first(
    pg_env,
) -> None:
    project = make_project(pg_env, 'hold-history')
    first = await _freeze(project, 'a', 5)
    second = await _freeze(project, 'b', 7)
    async with async_session_factory()() as session, project_tx(session, project):
        history = await list_holdout_freezes(session, project)
        current = await current_holdout_freeze(session, project)
    assert [h.id for h in history] == [second.id, first.id]
    assert current is not None
    assert current.id == second.id
    assert (current.item_count, current.frozen_by) == (7, 'b')


@pytest.mark.asyncio
async def test_a_freeze_that_is_not_current_leaves_the_current_row_alone(pg_env) -> None:
    project = make_project(pg_env, 'hold-import')
    curated = await _freeze(project, 'a', 5)
    async with async_session_factory()() as session, project_tx(session, project):
        await freeze_holdout(
            session, project, source='import:1', item_count=3, frozen_by='imp', make_current=False
        )
        current = await current_holdout_freeze(session, project)
        assert len(await list_holdout_freezes(session, project)) == 2
    assert current is not None
    assert current.id == curated.id


@pytest.mark.asyncio
async def test_a_freeze_for_a_missing_project_is_refused_and_writes_nothing(pg_env) -> None:
    ghost = uuid.uuid4()
    async with async_session_factory()() as session:
        with pytest.raises(RecordNotFoundError):
            async with project_tx(session, ghost):
                await freeze_holdout(session, ghost, source='manual', item_count=1, frozen_by='a')


@pytest.mark.asyncio
async def test_a_database_error_inside_the_freeze_rolls_it_back(pg_env) -> None:
    project = make_project(pg_env, 'hold-failclosed')
    await _freeze(project, 'a', 5)

    async def freeze_then_fail() -> None:
        async with async_session_factory()() as session, project_tx(session, project):
            await freeze_holdout(session, project, source='manual', item_count=9, frozen_by='b')
            await freeze_holdout(
                session,
                project,
                source='manual',
                item_count=9,
                frozen_by=None,  # type: ignore[arg-type]
            )

    with pytest.raises(IntegrityError):
        await freeze_then_fail()
    async with async_session_factory()() as session, project_tx(session, project):
        current = await current_holdout_freeze(session, project)
        assert len(await list_holdout_freezes(session, project)) == 1
    assert current is not None
    assert current.item_count == 5


@pytest.mark.asyncio
async def test_the_upgrade_load_keeps_ids_and_times_and_reads_back(pg_env) -> None:
    project = make_project(pg_env, 'hold-load')
    old, new = uuid.uuid4(), uuid.uuid4()
    t0 = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC)
    t1 = dt.datetime(2026, 2, 3, 4, 5, 6, tzinfo=dt.UTC)
    rows = [
        HoldoutFreezeLoad(
            id=old, is_current=False, source='upgrade', item_count=3, frozen_at=t0, frozen_by='old',
            seed=7, snapshot_storage_key='projects/x/holdout/a.json',
        ),
        HoldoutFreezeLoad(
            id=new, is_current=True, source='upgrade', item_count=4, frozen_at=t1, frozen_by='new'
        ),
    ]  # fmt: skip
    async with async_session_factory()() as session, deployment_tx(session, 'test load'):
        await load_holdout_freezes(session, project, rows)
    async with async_session_factory()() as session, project_tx(session, project):
        history = await list_holdout_freezes(session, project)
        current = await current_holdout_freeze(session, project)
    by_id = {h.id: h for h in history}
    assert set(by_id) == {old, new}
    assert by_id[old].frozen_at == t0
    assert (by_id[old].seed, by_id[old].snapshot_storage_key) == (7, 'projects/x/holdout/a.json')
    assert current is not None
    assert (current.id, current.frozen_at) == (new, t1)

    async with async_session_factory()() as session:
        with pytest.raises(IntegrityError):
            async with deployment_tx(session, 'test load again'):
                await load_holdout_freezes(session, project, rows[:1])


@pytest.mark.asyncio
async def test_the_load_refuses_two_current_freezes(pg_env) -> None:
    project = make_project(pg_env, 'hold-load-two')
    now = dt.datetime.now(dt.UTC)
    rows = [
        HoldoutFreezeLoad(
            id=uuid.uuid4(),
            is_current=True,
            source='upgrade',
            item_count=1,
            frozen_at=now,
            frozen_by='x',
        )
        for _ in range(2)
    ]
    async with async_session_factory()() as session, deployment_tx(session, 'test'):
        with pytest.raises(ValueError, match='at most one current'):
            await load_holdout_freezes(session, project, rows)
