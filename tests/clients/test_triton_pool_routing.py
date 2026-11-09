"""AsyncTritonPool sends each call to the server its router picks; one server is as before."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.clients import triton_pool_core
from src.clients.triton_pool_core import AsyncTritonPool
from src.clients.triton_router import NoServerError, TritonRouter


class _Client:
    def __init__(self) -> None:
        self.calls = 0
        self.gate = asyncio.Event()

    async def infer(self, *_a: Any, **_k: Any) -> str:
        self.calls += 1
        await self.gate.wait()
        return 'ok'


def _pool(router: TritonRouter | None, urls: tuple[str, ...]) -> tuple[AsyncTritonPool, dict]:
    pool = AsyncTritonPool(url=urls[0], pool_size=1, max_concurrent=64, router=router)
    clients = {u: _Client() for u in urls}
    pool._servers = {u: [c] for u, c in clients.items()}
    pool._indexes = dict.fromkeys(urls, 0)
    pool._semaphore = asyncio.Semaphore(64)
    pool._lock = asyncio.Lock()
    for u in urls:
        pool._stats.requests_per_channel[pool._channel_key(u, 0)] = 0
    pool._initialized = True
    return pool, clients


@pytest.fixture(autouse=True)
def _no_residency(monkeypatch: pytest.MonkeyPatch) -> None:
    async def loaded(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(triton_pool_core, 'ensure_loaded', loaded)


async def _run(pool: AsyncTritonPool, clients: dict, n: int) -> None:
    tasks = [asyncio.create_task(pool.infer('yolo', [], [])) for _ in range(n)]
    await asyncio.sleep(0.05)
    for client in clients.values():
        client.gate.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_a_second_ready_server_receives_calls() -> None:
    router = TritonRouter({'a': 'a:8001', 'b': 'b:8001'})
    router.update_ready({'a': ['yolo'], 'b': ['yolo']})
    pool, clients = _pool(router, ('a:8001', 'b:8001'))
    await _run(pool, clients, 20)
    assert (clients['a:8001'].calls, clients['b:8001'].calls) == (10, 10)


@pytest.mark.asyncio
async def test_a_server_that_does_not_report_the_model_gets_nothing() -> None:
    router = TritonRouter({'a': 'a:8001', 'b': 'b:8001'})
    router.update_ready({'a': ['yolo'], 'b': ['other']})
    pool, clients = _pool(router, ('a:8001', 'b:8001'))
    await _run(pool, clients, 6)
    assert (clients['a:8001'].calls, clients['b:8001'].calls) == (6, 0)


@pytest.mark.asyncio
async def test_no_ready_server_refuses_the_call() -> None:
    router = TritonRouter({'a': 'a:8001'})
    router.update_ready({})
    pool, _ = _pool(router, ('a:8001',))
    with pytest.raises(NoServerError):
        await pool.infer('yolo', [], [])


@pytest.mark.asyncio
async def test_without_a_router_every_call_goes_to_the_one_configured_server() -> None:
    pool, clients = _pool(None, ('only:8001',))
    await _run(pool, clients, 7)
    assert clients['only:8001'].calls == 7
    assert pool._clients is pool._servers['only:8001']
