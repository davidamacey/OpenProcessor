"""The registry reads Postgres: another session's create shows up within a poll, and nothing
touches OpenSearch."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import event

from src.db.engine import get_poll_engine
from src.db.repos import projects as repo
from src.db.session import async_session_factory, deployment_tx
from src.services.projects.records import new_project_for
from src.services.projects.registry import ProjectRegistry


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


async def _create(slug: str, **kw: Any) -> repo.ProjectRow:
    async with async_session_factory()() as session, deployment_tx(session, 'test create'):
        return await repo.create_project(
            session, new_project_for(slug, display_name=slug.title(), **kw)
        )


async def _write(fn: Any, *args: Any, **kw: Any) -> Any:
    async with async_session_factory()() as session, deployment_tx(session, 'test write'):
        return await fn(session, *args, **kw)


@pytest.fixture(autouse=True)
def _no_opensearch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any OpenSearch access from the registry is a failure."""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError('the registry must not call OpenSearch')

    monkeypatch.setattr('src.core.dependencies.get_opensearch', boom)
    monkeypatch.setattr('opensearchpy.AsyncOpenSearch.__init__', boom)


async def test_a_create_in_one_session_is_visible_to_a_second_registry_in_one_poll(
    pg_registry: Any,
) -> None:
    first, second = ProjectRegistry(), ProjectRegistry()
    await first.ensure_fresh()
    await second.ensure_fresh()
    assert second.get('seen') is None
    row = await _create('seen', status='active')
    await second.ensure_fresh()
    record = second.get('seen')
    assert record is not None
    assert record.id == row.id
    assert record.status == 'active'
    assert set(record.resources.indexes.values()) >= {'op_prj_seen__images', 'op_prj_seen__items'}
    assert second.refreshed
    assert not second.stale


async def test_a_steady_snapshot_does_one_indexed_read_per_poll(pg_registry: Any) -> None:
    registry = ProjectRegistry()
    await registry.ensure_fresh()
    statements: list[str] = []

    def record(_conn: Any, _cursor: Any, statement: str, *_a: Any) -> None:
        statements.append(statement)

    engine = get_poll_engine()
    event.listen(engine, 'before_cursor_execute', record)
    try:
        await registry.ensure_fresh()
    finally:
        event.remove(engine, 'before_cursor_execute', record)
    selects = [
        s for s in statements if s.lstrip().upper().startswith('SELECT') and 'set_config' not in s
    ]
    assert len(selects) == 1
    assert 'deployment_counters' in selects[0]
    assert 'FROM projects' not in selects[0]


async def test_a_transitional_status_is_confirmed_against_the_row(pg_registry: Any) -> None:
    await _create('building-now', status='building')
    registry = ProjectRegistry()
    await registry.ensure_fresh()
    assert registry.get('building-now').status == 'building'  # type: ignore[union-attr]
    # the counter is not bumped by a raw status write, so only the confirm read can see it
    from sqlalchemy import update

    from src.db.models.projects import Project

    async with async_session_factory()() as session, deployment_tx(session, 'test'):
        await session.execute(
            update(Project).where(Project.slug == 'building-now').values(status='active')
        )
    assert registry.get('building-now').status == 'building'  # type: ignore[union-attr]
    confirmed = await registry.lookup('building-now')
    assert confirmed is not None
    assert confirmed.status == 'active'


async def test_a_deleted_project_leaves_the_snapshot_and_its_slug_is_retired(
    pg_registry: Any,
) -> None:
    await _create('doomed', status='active')
    registry = ProjectRegistry()
    await registry.ensure_fresh()
    assert registry.get('doomed') is not None
    await _write(repo.mark_deleting, 'doomed', from_states=('active',))
    await _write(repo.finish_delete, 'doomed')
    await registry.ensure_fresh()
    assert registry.get('doomed') is None
    assert 'doomed' in registry.retired_slugs()


async def test_an_unreachable_database_keeps_the_snapshot_and_marks_it_stale(
    pg_registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _create('kept', status='active')
    registry = ProjectRegistry()
    await registry.ensure_fresh()

    def down() -> int:
        raise ConnectionError('database down')

    monkeypatch.setattr(registry._source, 'read_revision', down)
    await registry.ensure_fresh()
    assert registry.stale
    assert registry.get('kept') is not None
