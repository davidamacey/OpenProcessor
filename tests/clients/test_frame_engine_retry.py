"""What the client repeats: a call the transport lost, never a photo the engine failed.

gRPC status-code guide (https://grpc.io/docs/guides/status-codes/): INTERNAL is "reserved for
serious errors"; UNAVAILABLE is transient and "not always safe to retry non-idempotent
operations". The engine call is a pure function of the photo and the plan, so repeating it is
safe; an INTERNAL the *engine* raised for the photo would fail again. The two are told apart by
the details: the engine's start with ``wire.ENGINE_ERROR_PREFIX``.

The server is the engine's real front. The lost stream is simulated at the status layer (the
front aborts with the status text the transport gave in the 2026-11-13 chain); a real reset of a
stream cannot be made from the Python API on demand.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any, override

import grpc
import numpy as np
import pytest
import pytest_asyncio

from frame_engine.front import FrontService, make_server
from frame_engine.messages import EngineRequest, EngineResult
from frame_engine.stage_metrics import EngineMetrics
from src.clients import frame_engine
from src.clients.triton_pool_core import AsyncTritonPool
from src.domain import frame_engine_wire as wire


PLAN = wire.EnginePlan(branches=('embed_frame',), embed=wire.EmbedSpec('pe', 336, 32), crops='none')
TRANSPORT_TEXTS = (
    'Stream removed',
    'Received RST_STREAM with error code 1',
    'frame of size 9 overflows local window of 4',
)


class _Engine:
    """Answers ok, or raises ``failure`` for the photo, and counts the photos it was given."""

    def __init__(self) -> None:
        self.failure: Exception | None = None
        self.photos = 0

    def submit(self, request: EngineRequest) -> Future[EngineResult]:
        self.photos += 1
        done: Future[EngineResult] = Future()
        if self.failure is not None:
            done.set_exception(self.failure)
        else:
            out = {
                wire.SIZE: np.array([96, 64], np.int32),
                wire.WHOLE_EMBEDDING: np.ones(4, np.float32),
            }
            done.set_result(EngineResult('ok', 'gpu', {'embed_frame': ('ok', '')}, out))
        return done


class _LosingStreams(FrontService):
    """The first ``losses`` calls end the way a reset stream does: INTERNAL, details not ours."""

    def __init__(self, engine: _Engine, losses: int, text: str) -> None:
        super().__init__(
            engine, lambda: True, EngineMetrics(), max_in_flight=8, request_deadline=10.0
        )
        self.losses, self.text, self.arrived = losses, text, 0

    @override
    async def ModelInfer(self, request: Any, context: Any) -> Any:
        self.arrived += 1
        if self.arrived <= self.losses:
            await context.abort(grpc.StatusCode.INTERNAL, self.text)
        return await super().ModelInfer(request, context)


@pytest_asyncio.fixture
async def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_s: float) -> None:
        return None

    monkeypatch.setattr(frame_engine, 'RETRY_SLEEP', instant)


async def _serve(service: FrontService) -> tuple[Any, AsyncTritonPool]:
    server, port = make_server(service, address='127.0.0.1:0', max_message_bytes=2**24)
    await server.start()
    pool = AsyncTritonPool(url=f'127.0.0.1:{port}', pool_size=2, max_concurrent=4)
    await pool.initialize()
    return server, pool


@pytest.mark.asyncio
@pytest.mark.usefixtures('no_wait')
@pytest.mark.parametrize('text', TRANSPORT_TEXTS)
async def test_a_call_lost_with_its_stream_is_sent_again(text: str) -> None:
    service = _LosingStreams(_Engine(), losses=2, text=text)
    server, pool = await _serve(service)
    try:
        answer = await frame_engine.process(pool, b'jpeg', PLAN)
    finally:
        await pool.close()
        await server.stop(0)
    assert answer.status == wire.STATUS_OK
    assert service.arrived == 3


@pytest.mark.asyncio
@pytest.mark.usefixtures('no_wait')
async def test_a_stream_lost_every_time_ends_after_the_bounded_retries() -> None:
    service = _LosingStreams(_Engine(), losses=99, text='Stream removed')
    server, pool = await _serve(service)
    try:
        with pytest.raises(frame_engine.EngineCallError):
            await frame_engine.process(pool, b'jpeg', PLAN)
    finally:
        await pool.close()
        await server.stop(0)
    assert service.arrived == 1 + frame_engine.RETRIES


@pytest.mark.asyncio
@pytest.mark.usefixtures('no_wait')
async def test_an_internal_the_engine_raised_for_the_photo_is_not_retried() -> None:
    engine = _Engine()
    engine.failure = RuntimeError('kernel launch failed')
    service = FrontService(
        engine, lambda: True, EngineMetrics(), max_in_flight=8, request_deadline=10.0
    )
    server, pool = await _serve(service)
    try:
        with pytest.raises(frame_engine.EngineCallError) as failed:
            await frame_engine.process(pool, b'jpeg', PLAN)
    finally:
        await pool.close()
        await server.stop(0)
    assert engine.photos == 1
    assert 'kernel launch failed' in str(failed.value)
    assert not failed.value.overloaded
