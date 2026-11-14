"""A tick runs on one small pooled engine that is disposed on every exit path.

``tick_lock`` takes ``pg_try_advisory_xact_lock``, a transaction-level lock: PostgreSQL releases it
at COMMIT or ROLLBACK, so a pooled connection never carries it to the next checkout
(https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS). A session-level
lock would not be safe to pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.pool import NullPool

from src.db import engine as db_engine
from src.db.engine import TICK_POOL_SIZE, TICK_POOL_TIMEOUT_S, current_async_engine
from src.services.periodic.runner import run_periodic


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _db_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    secret = tmp_path / 'app_password'
    secret.write_text('x')
    monkeypatch.setenv('OP_DB_HOST', 'localhost')
    monkeypatch.setenv('OP_DB_APP_PASSWORD_FILE', str(secret))


def test_a_tick_runs_on_a_bounded_pool_that_fails_fast() -> None:
    async def body() -> tuple[type, int, int, int]:
        pool = current_async_engine().sync_engine.pool
        return type(pool), pool.size(), pool._max_overflow, pool.timeout()  # type: ignore[attr-defined]

    kind, size, overflow, timeout = run_periodic('probe', body())
    assert kind is not NullPool
    assert (size, overflow, timeout) == (TICK_POOL_SIZE, 0, TICK_POOL_TIMEOUT_S)
    assert TICK_POOL_SIZE >= 3  # lock connection + body session + a nested job-start session


def test_the_engine_is_disposed_when_the_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    disposed: list[bool] = []

    class _Engine:
        async def dispose(self) -> None:
            disposed.append(True)

    async def boom() -> None:
        raise RuntimeError('tick failed')

    monkeypatch.setattr(db_engine, 'new_tick_async_engine', lambda _name: _Engine())
    with pytest.raises(RuntimeError, match='tick failed'):
        run_periodic('boom', boom())
    assert disposed == [True]
