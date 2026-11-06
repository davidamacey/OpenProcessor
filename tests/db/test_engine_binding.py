"""``async_session_factory`` follows the engine bound to the running context (02b section 8).

A process that is not the API binds its own engine; only the API falls back to the
``WEB_CONCURRENCY``-sized request-path engine. These tests need no database server: an engine
is lazy and nothing here connects.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from src.config.database import DatabaseConfigError
from src.db import engine as db_engine
from src.db.session import async_session_factory


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.db

_URL = 'postgresql+psycopg://nobody:none@127.0.0.1:9/none'


@pytest.fixture(autouse=True)
def _not_the_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The process under test is not the API: nothing sets ``WEB_CONCURRENCY``."""
    monkeypatch.delenv('WEB_CONCURRENCY', raising=False)
    password = tmp_path / 'app_password'
    password.write_text('unused')
    monkeypatch.setenv('OP_DB_APP_PASSWORD_FILE', str(password))


def _engine():
    return create_async_engine(_URL, poolclass=pool.NullPool)


def test_a_session_factory_uses_the_bound_engine_without_an_api_worker_count() -> None:
    engine = _engine()
    with db_engine.bind_async_engine(engine):
        assert db_engine.current_async_engine() is engine
        assert async_session_factory()().bind is engine


def test_leaving_the_block_unbinds_and_the_api_engine_stays_fail_closed() -> None:
    with db_engine.bind_async_engine(_engine()):
        pass
    with pytest.raises(DatabaseConfigError, match='WEB_CONCURRENCY'):
        db_engine.current_async_engine()
    with pytest.raises(DatabaseConfigError, match='WEB_CONCURRENCY'):
        db_engine.get_async_engine()


def test_a_nested_bind_restores_the_outer_engine_and_unbinds_on_an_error() -> None:
    outer, inner = _engine(), _engine()

    def fail_inside_inner() -> None:
        with db_engine.bind_async_engine(inner):
            assert db_engine.current_async_engine() is inner
            raise RuntimeError

    with db_engine.bind_async_engine(outer):
        with pytest.raises(RuntimeError):
            fail_inside_inner()
        assert db_engine.current_async_engine() is outer


def test_a_task_created_inside_the_bind_sees_it() -> None:
    engine = _engine()

    async def child() -> object:
        return db_engine.current_async_engine()

    async def parent() -> object:
        with db_engine.bind_async_engine(engine):
            return await asyncio.create_task(child())

    assert asyncio.run(parent()) is engine


def test_the_api_engine_is_used_when_nothing_is_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = _engine()
    monkeypatch.setattr(db_engine, 'get_async_engine', lambda: sentinel)
    assert db_engine.current_async_engine() is sentinel


def test_a_worker_engine_is_a_small_bounded_pool_and_a_one_shot_engine_holds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    async def probe(name: str, *, one_shot: bool) -> None:
        async def inside() -> None:
            seen[name] = db_engine.current_async_engine()

        await db_engine.run_with_engine(inside(), name, one_shot=one_shot)

    async def both() -> None:
        await probe('op-worker', one_shot=False)
        await probe('op-once', one_shot=True)

    monkeypatch.setenv('OP_DB_HOST', '127.0.0.1')
    asyncio.run(both())
    worker, once = seen['op-worker'], seen['op-once']
    assert worker.sync_engine.pool.size() == db_engine.CHILD_POOL_SIZE == 2
    assert worker.sync_engine.pool._max_overflow == 0
    assert isinstance(once.sync_engine.pool, pool.NullPool)


def test_run_with_engine_unbinds_and_disposes_even_when_the_awaitable_raises() -> None:
    disposed: list[object] = []

    async def failing() -> None:
        disposed.append(db_engine.current_async_engine())
        raise RuntimeError

    async def outer() -> None:
        with pytest.raises(RuntimeError):
            await db_engine.run_with_engine(failing(), 'op-test')
        with pytest.raises(DatabaseConfigError, match='WEB_CONCURRENCY'):
            db_engine.current_async_engine()

    asyncio.run(outer())
    assert disposed
