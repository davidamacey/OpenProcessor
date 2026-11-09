"""The API pool records how long a checkout waited, including a wait that times out."""

from __future__ import annotations

import asyncio

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import text
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from src.db.engine import get_async_engine


pytestmark = pytest.mark.db


def _count() -> float:
    return REGISTRY.get_sample_value('op_db_pool_wait_seconds_count') or 0.0


def _sum() -> float:
    return REGISTRY.get_sample_value('op_db_pool_wait_seconds_sum') or 0.0


@pytest.mark.asyncio
async def test_a_checkout_that_waits_for_a_free_connection_is_observed(
    pg_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_DB_API_CONNECTIONS_TOTAL', '1')
    monkeypatch.setenv('WEB_CONCURRENCY', '1')
    engine = get_async_engine()
    before_count, before_sum = _count(), _sum()
    async with engine.connect() as held:
        await held.execute(text('SELECT 1'))

        async def waiter() -> None:
            async with engine.connect() as conn:
                await conn.execute(text('SELECT 1'))

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.5)
    await task
    assert _count() - before_count >= 2
    assert _sum() - before_sum >= 0.4


@pytest.mark.asyncio
async def test_a_checkout_that_times_out_is_observed_too(
    pg_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_DB_API_CONNECTIONS_TOTAL', '1')
    monkeypatch.setenv('WEB_CONCURRENCY', '1')
    engine = get_async_engine()
    monkeypatch.setattr(engine.sync_engine.pool, '_timeout', 0.3)
    before = _count()
    async with engine.connect():
        with pytest.raises(PoolTimeoutError):
            async with engine.connect():
                pass
    assert _count() - before >= 2
