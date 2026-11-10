"""The routed pool against real gRPC servers speaking the Triton v2 inference protocol.

Each server is a ``grpc.aio`` server running the ``GRPCInferenceService`` servicer generated in
``tritonclient`` (``service_pb2_grpc``), so the pool's real ``tritonclient.grpc.aio`` clients
connect, check readiness and infer over a socket. A server answers ``ModelInfer`` with its own
tag in the response id, which is how a test sees where a call went. Only the readiness source
(``serving_router.ready_models``, which reads ``serving_reports`` in PostgreSQL) is replaced.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, override

import grpc
import numpy as np
import pytest
import pytest_asyncio
import tritonclient.grpc as grpcclient
from tritonclient.grpc import service_pb2, service_pb2_grpc

from src.clients.triton_pool_core import AsyncTritonPool
from src.clients.triton_router import NoServerError, TritonRouter
from src.services import serving_router


class _Servicer(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.infers = 0
        self.ready_checks = 0

    @override
    async def ModelReady(self, request: Any, context: Any) -> Any:
        self.ready_checks += 1
        return service_pb2.ModelReadyResponse(ready=True)

    @override
    async def ModelInfer(self, request: Any, context: Any) -> Any:
        self.infers += 1
        return service_pb2.ModelInferResponse(model_name=request.model_name, id=self.tag)


class _Server:
    def __init__(self, tag: str) -> None:
        self.servicer = _Servicer(tag)
        self.server = grpc.aio.server()
        service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(self.servicer, self.server)
        self.port = self.server.add_insecure_port('127.0.0.1:0')

    @property
    def url(self) -> str:
        return f'127.0.0.1:{self.port}'


@pytest_asyncio.fixture
async def servers() -> AsyncIterator[dict[str, _Server]]:
    made = {tag: _Server(tag) for tag in ('a', 'b', 'c')}
    for s in made.values():
        await s.server.start()
    yield made
    for s in made.values():
        await s.server.stop(None)


def _input() -> list[grpcclient.InferInput]:
    tensor = grpcclient.InferInput('x', [1], 'FP32')
    tensor.set_data_from_numpy(np.zeros(1, dtype=np.float32))
    return [tensor]


async def _pool(servers: dict[str, _Server], *tags: str) -> AsyncTritonPool:
    router = TritonRouter({t: servers[t].url for t in tags})
    pool = AsyncTritonPool(url=servers[tags[0]].url, pool_size=1, max_concurrent=64, router=router)
    await pool.initialize()
    return pool


async def _where(pool: AsyncTritonPool, n: int) -> list[str]:
    results = await asyncio.gather(*[pool.infer('yolo', _input(), []) for _ in range(n)])
    return [r.get_response().id for r in results]


@pytest.mark.asyncio
async def test_concurrent_calls_spread_over_the_ready_servers(servers: dict[str, _Server]) -> None:
    pool = await _pool(servers, 'a', 'b')
    try:
        pool.router.update_ready({'a': ['yolo'], 'b': ['yolo']})
        tags = await _where(pool, 20)
        assert sorted(set(tags)) == ['a', 'b']
        assert servers['a'].servicer.infers + servers['b'].servicer.infers == 20
        assert servers['c'].servicer.infers == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_a_busy_server_is_skipped_for_an_idle_one(servers: dict[str, _Server]) -> None:
    pool = await _pool(servers, 'a', 'b', 'c')
    try:
        pool.router.update_ready({t: ['yolo'] for t in 'abc'})
        with pool.router.route('yolo') as r1, pool.router.route('yolo') as r2:
            assert r1.server != r2.server
            busy = {r1.server, r2.server}
            idle = ({'a', 'b', 'c'} - busy).pop()
            tags = await _where(pool, 1)
            assert tags == [idle]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_a_not_ready_server_gets_no_call(servers: dict[str, _Server]) -> None:
    pool = await _pool(servers, 'a', 'b', 'c')
    try:
        pool.router.update_ready({'a': ['yolo'], 'b': ['other'], 'c': ['yolo']})
        tags = await _where(pool, 12)
        assert set(tags) == {'a', 'c'}
        assert servers['b'].servicer.infers == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_no_ready_server_refuses_before_any_socket_call(
    servers: dict[str, _Server],
) -> None:
    pool = await _pool(servers, 'a', 'b')
    try:
        pool.router.update_ready({})
        with pytest.raises(NoServerError):
            await pool.infer('yolo', _input(), [])
        assert servers['a'].servicer.infers + servers['b'].servicer.infers == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_a_server_that_becomes_ready_is_followed_on_the_next_poll(
    servers: dict[str, _Server], monkeypatch: pytest.MonkeyPatch
) -> None:
    reports: dict[str, set[str]] = {'a': {'yolo'}}

    async def ready() -> dict[str, set[str]]:
        return {k: set(v) for k, v in reports.items()}

    async def active() -> Any:
        body = {'servers': [{'name': t, 'grpc_url': servers[t].url} for t in ('a', 'b')]}
        return SimpleNamespace(body=body), 1

    monkeypatch.setattr(serving_router, 'ready_models', ready)
    monkeypatch.setattr(serving_router, 'active_serving_profile', active)
    pool = AsyncTritonPool(url=servers['a'].url, pool_size=1, max_concurrent=64)
    await pool.initialize()
    try:
        await pool.adopt_router(await serving_router.build_router(servers['a'].url))
        assert set(await _where(pool, 8)) == {'a'}
        reports['b'] = {'yolo'}
        await serving_router.sync_router(pool)
        assert set(await _where(pool, 20)) == {'a', 'b'}
        assert servers['b'].servicer.infers > 0
    finally:
        await pool.close()
