"""``class_projection_sync`` applies one outbox row to the items; a re-run changes nothing (4B-5)."""

from __future__ import annotations

import uuid

import psycopg
import pytest
from _projects import make_project
from families_b.fake_items import FakeItems
from sqlalchemy import text

from src.db.session import async_session_factory, project_tx
from src.services.classes import projection_sync as sync, registry as registry_module
from src.services.classes.registry import ClassRegistry


pytestmark = pytest.mark.db


@pytest.fixture
def world(pg, pg_env, monkeypatch):
    slug = f'p{uuid.uuid4().hex[:8]}'
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        pid = make_project(conn, slug)
    monkeypatch.setattr(registry_module, '_project_ids', {})
    monkeypatch.setattr(registry_module, '_registries', {})
    monkeypatch.setattr('src.config.curation_accessors.items_index', lambda: 'items')
    return slug, pid


class _Ctx:
    """The slice of ``JobContext`` the two job bodies use."""

    def __init__(self, pid: uuid.UUID, params: dict) -> None:
        self.params = params
        self._pid = pid

    def session(self):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def open_session():
            async with async_session_factory()() as s, project_tx(s, self._pid):
                yield s

        return open_session()

    def cancelled(self) -> bool:
        return False

    async def progress(self, *_args, **_kwargs) -> None:
        return None


async def _pending(pid: uuid.UUID) -> list[tuple[uuid.UUID, str]]:
    async with async_session_factory()() as s, project_tx(s, pid):
        rows = await s.execute(
            text('SELECT id, kind FROM search_projection_tasks WHERE done_at IS NULL')
        )
        return [(i, k) for i, k in rows]


@pytest.mark.asyncio
async def test_a_rename_updates_exactly_the_items_with_that_class_id_and_a_rerun_is_a_noop(
    world, monkeypatch
) -> None:
    slug, pid = world
    reg = ClassRegistry(slug)
    sedan = await reg.add_class('sedan')
    coupe = await reg.add_class('coupe')
    await reg.update_class(sedan, name='saloon')
    monkeypatch.setattr(
        'src.services.classes.registry.get_class_registry', lambda: ClassRegistry(slug)
    )
    items = FakeItems(
        [
            {'class_id': sedan, 'class_name': 'sedan'},
            {'class_id': sedan, 'class_name': 'sedan'},
            {'class_id': coupe, 'class_name': 'coupe'},
        ]
    )
    [(task_id, kind)] = await _pending(pid)
    assert kind == 'class_rename'

    first = await sync.run_projection_sync(_Ctx(pid, {'projection_task_id': str(task_id)}), items)
    assert first['updated'] == 2
    assert [d['class_name'] for d in items.docs] == ['saloon', 'saloon', 'coupe']
    assert await _pending(pid) == []

    again = await sync.run_projection_sync(_Ctx(pid, {'projection_task_id': str(task_id)}), items)
    assert again == {'task': str(task_id), 'updated': 0, 'already_done': True}
    assert items.update_calls == 1


@pytest.mark.asyncio
async def test_a_rename_that_already_matches_changes_no_item(world, monkeypatch) -> None:
    slug, pid = world
    reg = ClassRegistry(slug)
    sedan = await reg.add_class('sedan')
    await reg.update_class(sedan, name='saloon')
    monkeypatch.setattr(
        'src.services.classes.registry.get_class_registry', lambda: ClassRegistry(slug)
    )
    items = FakeItems([{'class_id': sedan, 'class_name': 'saloon'}])
    [(task_id, _)] = await _pending(pid)
    done = await sync.run_projection_sync(_Ctx(pid, {'projection_task_id': str(task_id)}), items)
    assert done['updated'] == 0


@pytest.mark.asyncio
async def test_a_merge_row_moves_the_leftover_items_but_never_the_frozen_holdout(
    world, monkeypatch
) -> None:
    slug, pid = world
    reg = ClassRegistry(slug)
    a = await reg.add_class('sedan')
    b = await reg.add_class('coupe')
    await reg.merge_class(a, b)
    monkeypatch.setattr(
        'src.services.classes.registry.get_class_registry', lambda: ClassRegistry(slug)
    )
    items = FakeItems(
        [
            {'class_id': a, 'class_name': 'sedan'},
            {'class_id': a, 'class_name': 'sedan', 'test_holdout': True},
            {'class_id': b, 'class_name': 'coupe'},
        ]
    )
    [(task_id, kind)] = await _pending(pid)
    assert kind == 'class_merge'
    done = await sync.run_projection_sync(_Ctx(pid, {'projection_task_id': str(task_id)}), items)
    assert done['updated'] == 1
    assert [(d['class_id'], d['class_name']) for d in items.docs] == [
        (b, 'coupe'),
        (a, 'sedan'),
        (b, 'coupe'),
    ]


@pytest.mark.asyncio
async def test_the_merge_job_refuses_a_frozen_holdout_crop_and_writes_nothing(
    world, monkeypatch
) -> None:
    slug, pid = world
    reg = ClassRegistry(slug)
    a = await reg.add_class('sedan')
    b = await reg.add_class('coupe')
    monkeypatch.setattr(
        'src.services.classes.registry.get_class_registry', lambda: ClassRegistry(slug)
    )
    items = FakeItems([{'class_id': a, 'class_name': 'sedan', 'test_holdout': True}])
    with pytest.raises(sync.ProjectionTaskError, match='frozen test_holdout'):
        await sync.run_class_merge(_Ctx(pid, {'source_id': a, 'target_id': b}), items)
    assert (await reg.snapshot()).get(a).merged_into is None


@pytest.mark.asyncio
async def test_the_merge_job_merges_once_and_relabels_what_is_left(world, monkeypatch) -> None:
    slug, pid = world
    reg = ClassRegistry(slug)
    a = await reg.add_class('sedan')
    b = await reg.add_class('coupe')
    monkeypatch.setattr(
        'src.services.classes.registry.get_class_registry', lambda: ClassRegistry(slug)
    )
    seen: list[tuple[int, int, str]] = []

    async def relabel(_os, _index, source, target, name, *, cancelled):
        seen.append((source, target, name))
        return 3, 0

    monkeypatch.setattr(sync, 'relabel_merged_items', relabel)
    items = FakeItems([])
    ctx = _Ctx(pid, {'source_id': a, 'target_id': b})
    first = await sync.run_class_merge(ctx, items)
    second = await sync.run_class_merge(ctx, items)  # a retried delivery
    assert first == second == {'source_id': a, 'target_id': b, 'relabelled': 3, 'errors': 0}
    assert seen == [(a, b, 'coupe')] * 2
    merge_rows = [k for _, k in await _pending(pid) if k == 'class_merge']
    assert len(merge_rows) == 1  # the registry merge ran once


def test_a_class_merge_writes_no_cluster_field() -> None:
    """A merge relabels the class only: the item's place in the serving partition is not a class
    fact (it used to stamp the retired integer ``cluster_id = target class``)."""
    relabel = sync._merge_crop_merger(3, 'target')('c1', {'class_validated': True})
    assert relabel['class_id'] == 3
    assert not {k for k in relabel if k.startswith('cluster_')}
    assert 'cluster' not in sync._MERGE_SCRIPT['source']
