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
