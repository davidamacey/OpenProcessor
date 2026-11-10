"""The real ``_start_pools`` sizes the executor and the Triton pool from the worker count.

Nothing is patched away except the serving-profile read (a database call); the pool and the
executor are built by the lifespan's own step and measured on the objects it publishes.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from src.app import lifespan as lifespan_module
from src.app.resources import AppResources
from src.config import get_settings
from src.config.api_pools import (
    EXECUTOR_THREADS_TOTAL,
    TRITON_CHANNELS_TOTAL,
    TRITON_INFLIGHT_TOTAL,
)
from src.config.database import DatabaseConfigError
from src.services import serving_router


@pytest.fixture(autouse=True)
def _no_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    async def none() -> None:
        return None

    monkeypatch.setattr(serving_router, 'active_serving_profile', none)


@pytest_asyncio.fixture(autouse=True)
async def _release_resources():
    yield
    pool, executor = AppResources.async_triton_pool, AppResources.shared_executor
    AppResources.async_triton_pool = None
    AppResources.shared_executor = None
    if pool is not None:
        await pool.close()
    if executor is not None:
        executor.shutdown(wait=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('workers', [1, 8, 32])
async def test_each_process_takes_its_share_of_the_totals(
    monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    monkeypatch.setenv('WEB_CONCURRENCY', str(workers))
    settings = get_settings()
    await lifespan_module._start_pools(settings)
    pool, executor = AppResources.async_triton_pool, AppResources.shared_executor
    assert pool is not None
    assert executor is not None
    assert executor._max_workers == EXECUTOR_THREADS_TOTAL // workers
    assert pool.pool_size == TRITON_CHANNELS_TOTAL // workers
    assert pool.max_concurrent == TRITON_INFLIGHT_TOTAL // workers
    assert pool.url == settings.triton_url
    assert len(pool._servers[settings.triton_url]) == TRITON_CHANNELS_TOTAL // workers


@pytest.mark.asyncio
async def test_without_a_worker_count_the_lifespan_refuses_to_size_the_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('WEB_CONCURRENCY', raising=False)
    with pytest.raises(DatabaseConfigError, match='WEB_CONCURRENCY'):
        await lifespan_module._start_pools(get_settings())
    assert AppResources.async_triton_pool is None
    assert AppResources.shared_executor is None
