"""The readiness probe over real gRPC: a probe the transport refuses is asked again on another
channel, an unreachable engine is judged within the stated bound, an answer is never retried.

The server is the engine's real front (``frame_engine/front.py``) with ``ModelReady`` made to
fail the way a channel closed by a GOAWAY does: ``UNAVAILABLE`` ("Too many pings")."""

from __future__ import annotations

import socket
import time
from typing import Any, override

import grpc
import pytest
import pytest_asyncio

from frame_engine.front import FrontService, make_server
from frame_engine.stage_metrics import EngineMetrics
from src.clients import frame_engine
from src.clients.triton_pool_core import AsyncTritonPool


class _Engine:
    def submit(self, request: Any) -> Any:
        raise AssertionError('a probe sends no photo')


class _Refusing(FrontService):
    """``ModelReady`` fails ``refusals`` times with UNAVAILABLE, then answers ``ready``."""

    def __init__(self, refusals: int, ready: bool = True) -> None:
        super().__init__(
            _Engine(), lambda: ready, EngineMetrics(), max_in_flight=1, request_deadline=5.0
        )
        self.refusals = refusals
        self.asked = 0

    @override
    async def ModelReady(self, request: Any, context: Any) -> Any:
        self.asked += 1
        if self.asked <= self.refusals:
            await context.abort(grpc.StatusCode.UNAVAILABLE, 'Too many pings')
        return await super().ModelReady(request, context)


async def _pool(service: FrontService) -> tuple[Any, AsyncTritonPool]:
    server, port = make_server(service, address='127.0.0.1:0', max_message_bytes=2**24)
    await server.start()
    pool = AsyncTritonPool(url=f'127.0.0.1:{port}', pool_size=3, max_concurrent=4)
    await pool.initialize()
    return server, pool


@pytest_asyncio.fixture
async def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_s: float) -> None:
        return None

    monkeypatch.setattr(frame_engine, 'RETRY_SLEEP', instant)


@pytest.mark.asyncio
@pytest.mark.usefixtures('no_wait')
async def test_a_refused_probe_succeeds_on_the_next_channel() -> None:
    service = _Refusing(refusals=1)
    server, pool = await _pool(service)
    try:
        readiness = await frame_engine.probe_ready(pool)
    finally:
        await pool.close()
        await server.stop(0)
    assert (readiness.outcome, readiness.attempts) == (frame_engine.PROBE_RECOVERED, 2)
    assert readiness.ready


@pytest.mark.asyncio
@pytest.mark.usefixtures('no_wait')
async def test_a_probe_refused_every_time_is_undelivered_not_not_ready() -> None:
    service = _Refusing(refusals=99)
    server, pool = await _pool(service)
    try:
        readiness = await frame_engine.probe_ready(pool)
    finally:
        await pool.close()
        await server.stop(0)
    assert readiness.outcome == frame_engine.PROBE_UNDELIVERED
    assert readiness.attempts == frame_engine.PROBE_ATTEMPTS == service.asked
    assert readiness.code == 'StatusCode.UNAVAILABLE'
    assert not readiness.ready


@pytest.mark.asyncio
@pytest.mark.usefixtures('no_wait')
async def test_an_answer_of_not_ready_is_not_retried() -> None:
    service = _Refusing(refusals=0, ready=False)
    server, pool = await _pool(service)
    try:
        readiness = await frame_engine.probe_ready(pool)
    finally:
        await pool.close()
        await server.stop(0)
    assert (readiness.outcome, readiness.attempts) == (frame_engine.PROBE_NOT_READY, 1)
    assert service.asked == 1


@pytest.mark.asyncio
async def test_an_engine_that_refuses_connections_is_judged_within_two_seconds() -> None:
    """The bound the docs state: connection refused answers in about 0.3 s (the backoff waits)."""
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]  # closed again: nothing listens here
    pool = AsyncTritonPool(url=f'127.0.0.1:{port}', pool_size=3, max_concurrent=4)
    await pool.initialize()
    try:
        started = time.monotonic()
        readiness = await frame_engine.probe_ready(pool)
        elapsed = time.monotonic() - started
    finally:
        await pool.close()
    assert readiness.outcome == frame_engine.PROBE_UNDELIVERED
    assert elapsed < 2.0


@pytest.mark.asyncio
@pytest.mark.usefixtures('no_wait')
async def test_the_ingest_gate_does_not_answer_503_for_one_refused_probe(tmp_path: Any) -> None:
    """The 2026-11-13 chain: the probe met a GOAWAY and the whole request was refused."""
    from curation.engine_fakes import engine_service

    from src.services.curation import ingest_gpu_decode

    ingest_gpu_decode._READY.clear()
    svc, *_ = engine_service(tmp_path)
    service = _Refusing(refusals=1)
    server, pool = await _pool(service)
    svc.frame_engine = pool
    try:
        assert await ingest_gpu_decode.engine_ready(svc) is True
    finally:
        await pool.close()
        await server.stop(0)
        ingest_gpu_decode._READY.clear()
