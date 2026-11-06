"""A rename or a merge through the service writes its outbox row in the same transaction (2B-3)."""

from __future__ import annotations

import uuid

import psycopg
import pytest
from _projects import make_project
from sqlalchemy import text

from src.db.session import async_session_factory, project_tx
from src.services.classes import registry as registry_module
from src.services.classes.registry import ClassRegistry


pytestmark = pytest.mark.db


@pytest.fixture
def service(pg, pg_env, monkeypatch) -> tuple[ClassRegistry, uuid.UUID]:
    slug = f'p{uuid.uuid4().hex[:8]}'
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        pid = make_project(conn, slug)
    monkeypatch.setattr(registry_module, '_project_ids', {})
    return ClassRegistry(slug), pid


async def _tasks(pid: uuid.UUID) -> list[tuple[str, dict]]:
    async with async_session_factory()() as s, project_tx(s, pid):
        rows = await s.execute(
            text('SELECT kind, payload FROM search_projection_tasks ORDER BY created_at')
        )
        return [(k, p) for k, p in rows]


@pytest.mark.asyncio
async def test_rename_and_merge_each_leave_one_outbox_row(service) -> None:
    reg, pid = service
    a = await reg.add_class('sedan')
    b = await reg.add_class('coupe')
    renamed = await reg.update_class(a, name='saloon')
    assert renamed.class_name == 'saloon'
    await reg.merge_class(a, b)

    assert await _tasks(pid) == [
        ('class_rename', {'class_id': a, 'new_name': 'saloon'}),
        ('class_merge', {'source_ids': [a], 'target_id': b}),
    ]


@pytest.mark.asyncio
async def test_a_group_change_writes_no_outbox_row(service) -> None:
    reg, pid = service
    a = await reg.add_class('sedan')
    await reg.update_class(a, group='cars')

    assert await _tasks(pid) == []


@pytest.mark.asyncio
async def test_the_snapshot_follows_the_revision(service) -> None:
    reg, _pid = service
    assert (await reg.snapshot()).classes == []
    await reg.add_class('sedan')
    first = await reg.snapshot()
    assert [c.class_name for c in first.classes] == ['sedan']
    again = await reg.snapshot()
    assert again.registry_revision == first.registry_revision
    other = ClassRegistry(reg.project)  # another process: its cache starts empty
    await other.add_class('coupe')
    third = await reg.snapshot()  # this process sees the other writer on its next read
    assert [c.class_name for c in third.classes] == ['sedan', 'coupe']
    assert third.registry_revision > first.registry_revision


@pytest.mark.asyncio
async def test_an_unknown_project_is_a_registry_error(pg_env, monkeypatch) -> None:
    from src.db.repos.classes import ClassRegistryError

    monkeypatch.setattr(registry_module, '_project_ids', {})
    with pytest.raises(ClassRegistryError, match='unknown project'):
        await ClassRegistry('no-such-project').snapshot()


@pytest.mark.asyncio
async def test_copy_from_replaces_the_target_registry_with_the_source(pg, service) -> None:
    source, _pid = service
    slug = f'p{uuid.uuid4().hex[:8]}'
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        make_project(conn, slug)
    target = ClassRegistry(slug)
    await source.add_class('sedan', group='cars')
    await source.add_class('coupe')
    await target.add_class('stale')

    assert await target.copy_from(source.project) == 2

    copied = await target.snapshot()
    assert [(c.class_id, c.class_name) for c in copied.classes] == [(0, 'sedan'), (1, 'coupe')]
    assert copied.next_class_id == 2
