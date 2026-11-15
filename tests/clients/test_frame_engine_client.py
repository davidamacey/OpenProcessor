"""The API's frame-engine client (``src/clients/frame_engine.py``): one unary call per image
through the pool, the pool's bounded retry on the overload status, and what is left after it."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import grpc
import numpy as np
import pytest
from tritonclient.utils import InferenceServerException

from src.clients import frame_engine, triton_pool_core
from src.clients.triton_pool_core import AsyncTritonPool
from src.domain import frame_engine_wire as wire


PLAN = wire.EnginePlan(
    branches=('detect', 'embed_frame'),
    detector=wire.DetectorSpec('det', 640, 114),
    embed=wire.EmbedSpec('pe', 336, 32),
    crops='inline',
)


class _Result:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self.outputs = outputs

    def as_numpy(self, name: str) -> np.ndarray | None:
        return self.outputs.get(name)


def _text(value: str) -> np.ndarray:
    return np.array([value.encode()], dtype=np.object_)


def _ok_result(branch: str = 'ok') -> _Result:
    status = {'detect': {'status': branch}, 'embed_frame': {'status': 'ok'}}
    if branch == 'error':
        status['detect']['reason'] = 'model_error'
    return _Result(
        {
            'status': _text('ok'),
            'decode_route': _text('gpu'),
            'branch_status': _text(json.dumps(status)),
            'size': np.array([40, 30], np.int32),
            'det_boxes': np.zeros((1, 4), np.float32),
            'det_scores': np.ones(1, np.float32),
            'det_classes': np.zeros(1, np.int32),
            'letterbox': np.array([1.0, 0.0, 0.0]),
            'blur_sums': np.zeros((2, 3), np.int64),
            'whole_embedding': np.ones(4, np.float32),
            'crop_jpegs': np.array([b'\xff\xd8x'], dtype=np.object_),
        }
    )


def _status_error(code: grpc.StatusCode) -> InferenceServerException:
    return InferenceServerException(msg=f'{code.name}: engine', status=str(code))


class _Client:
    """A channel that answers the scripted errors in order, then the result."""

    def __init__(self, errors: list[Exception], result: Any = None) -> None:
        self.errors = list(errors)
        self.result = result
        self.calls: list[tuple[str, list[Any]]] = []

    async def infer(self, model: str, inputs: list[Any], **_k: Any) -> Any:
        self.calls.append((model, inputs))
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def _pool(client: _Client) -> AsyncTritonPool:
    pool = AsyncTritonPool(url='frame-engine:8001', pool_size=1, max_concurrent=8)
    pool._servers = {'frame-engine:8001': [client]}
    pool._indexes = {'frame-engine:8001': 0}
    pool._semaphore = asyncio.Semaphore(8)
    pool._lock = asyncio.Lock()
    pool._stats.requests_per_channel[0] = 0
    pool._initialized = True
    return pool


@pytest.fixture(autouse=True)
def _no_residency(monkeypatch: pytest.MonkeyPatch) -> None:
    async def loaded(*_a: Any, **_k: Any) -> None:
        return None

    async def no_wait(_s: float) -> None:
        return None

    monkeypatch.setattr(frame_engine, 'RETRY_SLEEP', no_wait)
    monkeypatch.setattr(triton_pool_core, 'ensure_loaded', loaded)
    monkeypatch.setattr(triton_pool_core.asyncio, 'sleep', no_wait)


@pytest.mark.asyncio
async def test_one_unary_call_per_image_with_the_image_and_the_plan() -> None:
    client = _Client([], _ok_result())
    answer = await frame_engine.process(_pool(client), b'jpeg-bytes', PLAN)
    [(model, inputs)] = client.calls
    assert model == wire.MODEL_NAME
    assert [i.name() for i in inputs] == ['image', 'plan']
    assert [i.datatype() for i in inputs] == ['BYTES', 'BYTES']
    assert answer.ok
    assert answer.decode_route == 'gpu'
    assert answer.outputs['crop_jpegs'] == [b'\xff\xd8x']
    assert set(answer.outputs) == {'size', *wire.expected_outputs(PLAN)}


@pytest.mark.asyncio
async def test_stored_boxes_travel_as_fp64() -> None:
    client = _Client([], _ok_result())
    rows = np.array([[0.1, 0.2, 0.3, 0.4]])
    await frame_engine.process(_pool(client), b'j', PLAN, rows)
    boxes = client.calls[0][1][2]
    assert (boxes.name(), boxes.datatype(), boxes.shape()) == ('boxes', 'FP64', [1, 4])


@pytest.mark.asyncio
async def test_an_overload_is_retried_and_the_next_answer_used() -> None:
    busy = _status_error(grpc.StatusCode.RESOURCE_EXHAUSTED)
    client = _Client([busy, busy], _ok_result())
    answer = await frame_engine.process(_pool(client), b'j', PLAN)
    assert answer.ok
    assert len(client.calls) == 3


@pytest.mark.asyncio
async def test_an_overload_is_retried_until_the_photos_deadline_not_a_fixed_count() -> None:
    busy = _status_error(grpc.StatusCode.RESOURCE_EXHAUSTED)
    many = 3 * (1 + frame_engine.RETRIES)
    client = _Client([busy] * many, _ok_result())
    assert (await frame_engine.process(_pool(client), b'j', PLAN)).ok
    assert len(client.calls) == many + 1


@pytest.mark.asyncio
async def test_an_overload_that_outlasts_the_deadline_is_reported_as_overloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def tick(_s: float) -> None:
        await asyncio.sleep(0.01)

    monkeypatch.setattr(frame_engine, 'DEADLINE_SECONDS', 0.1)
    monkeypatch.setattr(frame_engine, 'RETRY_SLEEP', tick)
    busy = _status_error(grpc.StatusCode.RESOURCE_EXHAUSTED)
    client = _Client([busy] * 100_000)
    with pytest.raises(frame_engine.EngineCallError) as caught:
        await frame_engine.process(_pool(client), b'j', PLAN)
    assert caught.value.overloaded is True
    assert len(client.calls) > 1 + frame_engine.RETRIES


@pytest.mark.parametrize(
    'error',
    [
        _status_error(grpc.StatusCode.INTERNAL),
        _status_error(grpc.StatusCode.INVALID_ARGUMENT),
        OSError('connection refused'),
    ],
)
@pytest.mark.asyncio
async def test_any_other_failure_is_not_an_overload(error: Exception) -> None:
    client = _Client([error] * 10)
    with pytest.raises(frame_engine.EngineCallError) as caught:
        await frame_engine.process(_pool(client), b'j', PLAN)
    assert caught.value.overloaded is False
    assert len(client.calls) == 1  # not retryable


@pytest.mark.asyncio
async def test_an_engine_that_stays_unavailable_is_retried_then_not_an_overload() -> None:
    down = _status_error(grpc.StatusCode.UNAVAILABLE)
    client = _Client([down] * 10)
    with pytest.raises(frame_engine.EngineCallError) as caught:
        await frame_engine.process(_pool(client), b'j', PLAN)
    assert caught.value.overloaded is False
    assert len(client.calls) == 1 + frame_engine.RETRIES


@pytest.mark.asyncio
async def test_a_failed_branch_is_an_answer_that_is_not_ok() -> None:
    client = _Client([], _ok_result(branch='error'))
    answer = await frame_engine.process(_pool(client), b'j', PLAN)
    assert answer.status == 'ok'
    assert answer.branches['detect'] == ('error', 'model_error')
    assert not answer.ok


@pytest.mark.asyncio
async def test_a_declined_image_carries_its_reason_and_no_outputs() -> None:
    client = _Client(
        [],
        _Result(
            {
                'status': _text('progressive'),
                'decode_route': _text('none'),
                'branch_status': _text(
                    json.dumps(
                        {b: {'status': 'error', 'reason': 'image_failed'} for b in PLAN.branches}
                    )
                ),
            }
        ),
    )
    answer = await frame_engine.process(_pool(client), b'j', PLAN)
    assert (answer.status, answer.decode_route, answer.outputs) == ('progressive', 'none', {})
    assert not answer.ok


def test_the_address_defaults_to_the_compose_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_FRAME_ENGINE_URL', raising=False)
    assert frame_engine.frame_engine_url() == 'frame-engine:8001'
    monkeypatch.setenv('OP_FRAME_ENGINE_URL', 'fe-gpu2:8001')
    assert frame_engine.frame_engine_url() == 'fe-gpu2:8001'


def test_the_engine_answers_every_request_before_this_client_gives_up() -> None:
    from frame_engine.settings import EngineSettings

    engine = EngineSettings()
    assert engine.triton_timeout_seconds < engine.request_deadline_seconds
    assert engine.request_deadline_seconds < frame_engine.TIMEOUT_SECONDS
