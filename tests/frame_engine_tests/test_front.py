"""The KServe v2 front over a real gRPC channel, with a stand-in engine: tensor names, datatypes
and shapes per branch combination, readiness, and the status codes the API retries on."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import Future
from typing import Any

import numpy as np
import pytest
from prometheus_client import CollectorRegistry
from tritonclient.grpc import InferInput, aio
from tritonclient.utils import InferenceServerException

from frame_engine.cuda_fault import ContextLostError
from frame_engine.front import FrontService, make_server, pack_bytes, unpack_bytes
from frame_engine.messages import EngineRequest, EngineResult
from frame_engine.stage_metrics import EngineMetrics
from frame_engine.transport import OverloadError, TritonUnavailableError
from src.domain import frame_engine_wire as wire


pytestmark = pytest.mark.asyncio(loop_scope='function')

DETECTOR = {'model': 'det', 'input_size': 640, 'letterbox_fill': 114}
EMBED = {'model': 'pe', 'size': 336, 'batch': 32}


def _plan(branches: list[str], **over: Any) -> str:
    body: dict[str, Any] = {'version': 1, 'branches': branches, 'crops': 'none', **over}
    if 'detect' in branches:
        body.setdefault('detector', DETECTOR)
    if {'embed_frame', 'embed_crops'} & set(branches):
        body.setdefault('embed', EMBED)
    return json.dumps(body)


class FakeEngine:
    """Answers like the engine for two detections (classes 1 and 2) and 4-wide vectors."""

    def __init__(self) -> None:
        self.raise_: Exception | None = None
        self.gate: threading.Event | None = None
        self.stuck_in = 'detect'
        self.seen: list[EngineRequest] = []

    def submit(self, request: EngineRequest) -> Future[EngineResult]:
        self.seen.append(request)
        fut: Future[EngineResult] = Future()
        gate = self.gate

        def work() -> None:
            if not fut.set_running_or_notify_cancel():  # as ThreadPoolExecutor does
                return
            if gate is not None:
                request.enter(self.stuck_in)
                gate.wait(10)
            if self.raise_ is not None:
                fut.set_exception(self.raise_)
                return
            fut.set_result(self.answer(request))

        threading.Thread(target=work).start()
        return fut

    @staticmethod
    def answer(request: EngineRequest) -> EngineResult:
        plan = request.plan
        out: dict[str, Any] = {wire.SIZE: np.array([96, 64], np.int32)}
        n = 2 if request.boxes is None else len(request.boxes)
        if plan.runs('detect'):
            out[wire.DET_BOXES] = np.ones((n, 4), np.float32)
            out[wire.DET_SCORES] = np.full(n, 0.9, np.float32)
            out[wire.DET_CLASSES] = np.array([1, 2], np.int32)
            out[wire.LETTERBOX] = np.array([6.6, 0.0, 106.0])
            out[wire.BLUR_SUMS] = np.zeros((n + 1, 3), np.int64)
        if plan.runs('embed_frame'):
            out[wire.WHOLE_EMBEDDING] = np.arange(4, dtype=np.float32)
        if plan.runs('embed_crops'):
            out[wire.CROP_EMBEDDINGS] = np.ones((n, 4), np.float32)
            out[wire.CROP_EMBEDDED] = np.ones(n, np.bool_)
        if plan.crops == 'inline':
            out[wire.CROP_JPEGS] = [b'\xff\xd8a', b'\xff\xd8bc'][:n]
        branches: dict[str, tuple[wire.BranchState, str]] = {
            b: ('ok', '') for b in plan.branches if b in wire.BUILT_BRANCHES
        }
        branches.update(dict.fromkeys(plan.unsupported(), ('error', wire.UNSUPPORTED_BRANCH)))
        return EngineResult('ok', 'gpu', branches, out)


class Harness:
    def __init__(self, engine: FakeEngine, *, max_in_flight: int = 8,
                 request_deadline: float = 30.0) -> None:  # fmt: skip
        self.engine = engine
        self.is_ready = True
        self.service = FrontService(
            engine, lambda: self.is_ready, EngineMetrics(CollectorRegistry()),
            max_in_flight=max_in_flight, request_deadline=request_deadline,
        )  # fmt: skip

    async def __aenter__(self) -> Harness:
        self.server, port = make_server(
            self.service, address='127.0.0.1:0', max_message_bytes=64 << 20
        )
        await self.server.start()
        self.client = aio.InferenceServerClient(f'127.0.0.1:{port}')
        return self

    async def __aexit__(self, *_a: Any) -> None:
        await self.client.close()
        await self.server.stop(None)

    async def infer(self, plan: str, *, image: bytes = b'\xff\xd8jpeg', boxes: Any = None,
                    model: str = 'frame_engine') -> Any:  # fmt: skip
        inputs = []
        for name, value in ((wire.IMAGE, image), (wire.PLAN, plan.encode())):
            inp = InferInput(name, [1], 'BYTES')
            inp.set_data_from_numpy(np.array([value], dtype=np.object_))
            inputs.append(inp)
        if boxes is not None:
            dtype = 'FP64' if boxes.dtype == np.float64 else 'FP32'
            inp = InferInput(wire.BOXES, list(boxes.shape), dtype)
            inp.set_data_from_numpy(boxes)
            inputs.append(inp)
        return await self.client.infer(model, inputs)


def _code(err: pytest.ExceptionInfo[InferenceServerException]) -> str:
    return str(err.value.status())


async def test_bytes_tensors_round_trip() -> None:
    items = [b'', b'a', b'\x00' * 5]
    assert unpack_bytes(pack_bytes(items)) == items


@pytest.mark.parametrize(
    ('branches', 'extra', 'want'),
    [
        (['detect'], {}, {'det_boxes': [2, 4], 'det_scores': [2], 'det_classes': [2],
                          'letterbox': [3], 'blur_sums': [3, 3]}),
        (['embed_frame'], {}, {'whole_embedding': [4]}),
        (['detect', 'embed_crops'], {'crops': 'inline'}, {
            'det_boxes': [2, 4], 'det_scores': [2], 'det_classes': [2], 'letterbox': [3],
            'blur_sums': [3, 3], 'crop_embeddings': [2, 4], 'crop_embedded': [2],
            'crop_jpegs': [2]}),
        (['detect', 'embed_frame', 'embed_crops'], {'crops': 'inline'}, {
            'det_boxes': [2, 4], 'det_scores': [2], 'det_classes': [2], 'letterbox': [3],
            'blur_sums': [3, 3], 'whole_embedding': [4], 'crop_embeddings': [2, 4],
            'crop_embedded': [2], 'crop_jpegs': [2]}),
    ],
)  # fmt: skip
async def test_each_branch_combination_answers_its_tensors(
    branches: list[str], extra: dict[str, Any], want: dict[str, list[int]]
) -> None:
    async with Harness(FakeEngine()) as h:
        result = await h.infer(_plan(branches, **extra))
    response = result.get_response()
    got = {o.name: (o.datatype, list(o.shape)) for o in response.outputs}
    always = {'status': ('BYTES', [1]), 'decode_route': ('BYTES', [1]),
              'branch_status': ('BYTES', [1]), 'size': ('INT32', [2])}  # fmt: skip
    assert got == {**always, **{n: (wire.OUTPUT_DATATYPES[n], s) for n, s in want.items()}}
    assert result.as_numpy('status')[0] == b'ok'
    assert result.as_numpy('decode_route')[0] == b'gpu'
    states = wire.decode_branch_status(result.as_numpy('branch_status')[0])
    assert states == dict.fromkeys(branches, ('ok', ''))
    if 'crop_jpegs' in want:
        assert [bytes(x) for x in result.as_numpy('crop_jpegs')] == [b'\xff\xd8a', b'\xff\xd8bc']


async def test_stored_boxes_reach_the_engine_as_fp64_rows() -> None:
    engine = FakeEngine()
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], np.float64)
    async with Harness(engine) as h:
        result = await h.infer(_plan(['embed_frame', 'embed_crops']), boxes=boxes)
    assert engine.seen[0].boxes is not None
    np.testing.assert_array_equal(engine.seen[0].boxes, boxes)
    assert result.as_numpy('crop_embeddings').shape == (1, 4)


async def test_unsupported_branches_are_reported_per_branch() -> None:
    async with Harness(FakeEngine()) as h:
        result = await h.infer(_plan(['detect', 'faces']))
    assert wire.decode_branch_status(result.as_numpy('branch_status')[0]) == {
        'detect': ('ok', ''),
        'faces': ('error', 'unsupported_branch'),
    }


@pytest.mark.parametrize(
    ('plan', 'kwargs'),
    [
        ('{"version": 1}', {}),
        (_plan(['embed_crops']), {}),
        (_plan(['detect']), {'boxes': np.zeros((1, 4), np.float64)}),
        (_plan(['embed_frame', 'embed_crops']), {'boxes': np.zeros((1, 3), np.float64)}),
        (_plan(['embed_frame', 'embed_crops']), {'boxes': np.zeros((1, 4), np.float32)}),
    ],
)
async def test_a_bad_request_is_invalid_argument(plan: str, kwargs: dict[str, Any]) -> None:
    engine = FakeEngine()
    async with Harness(engine) as h:
        with pytest.raises(InferenceServerException) as err:
            await h.infer(plan, **kwargs)
    assert _code(err) == 'StatusCode.INVALID_ARGUMENT'
    assert engine.seen == []


async def test_an_unknown_model_is_not_found() -> None:
    async with Harness(FakeEngine()) as h:
        with pytest.raises(InferenceServerException) as err:
            await h.infer(_plan(['detect']), model='some_other_model')
    assert _code(err) == 'StatusCode.NOT_FOUND'


async def test_readiness_follows_the_transport_and_draining() -> None:
    engine = FakeEngine()
    async with Harness(engine) as h:
        assert await h.client.is_server_live()
        assert await h.client.is_server_ready()
        assert await h.client.is_model_ready('frame_engine')
        assert not await h.client.is_model_ready('some_other_model')
        h.is_ready = False  # the model server went away or the region is not registered
        assert not await h.client.is_server_ready()
        assert not await h.client.is_model_ready('frame_engine')
        with pytest.raises(InferenceServerException) as err:
            await h.infer(_plan(['detect']))
        assert _code(err) == 'StatusCode.UNAVAILABLE'
        h.is_ready = True
        assert await h.client.is_model_ready('frame_engine')
        h.service.draining = True
        assert not await h.client.is_server_ready()
        assert await h.client.is_server_live()
    assert len(engine.seen) == 0


async def test_metadata_names_the_protocol_tensors() -> None:
    async with Harness(FakeEngine()) as h:
        meta = await h.client.get_model_metadata('frame_engine', as_json=True)
    assert {t['name']: t['datatype'] for t in meta['inputs']} == dict(wire.INPUT_DATATYPES)
    assert {t['name']: t['datatype'] for t in meta['outputs']} == dict(wire.OUTPUT_DATATYPES)


@pytest.mark.parametrize(
    ('exc', 'code'),
    [
        (OverloadError('budget', 'full'), 'StatusCode.RESOURCE_EXHAUSTED'),
        (OverloadError('slab', 'full'), 'StatusCode.RESOURCE_EXHAUSTED'),
        (TritonUnavailableError('det: down'), 'StatusCode.UNAVAILABLE'),
        (ContextLostError('cudaErrorIllegalAddress'), 'StatusCode.UNAVAILABLE'),
        (ValueError('kernel bug'), 'StatusCode.INTERNAL'),
    ],
)
async def test_engine_failures_map_to_their_status_codes(exc: Exception, code: str) -> None:
    engine = FakeEngine()
    engine.raise_ = exc
    async with Harness(engine) as h:
        with pytest.raises(InferenceServerException) as err:
            await h.infer(_plan(['detect']))
        assert h.service.in_flight == 0
    assert _code(err) == code


async def test_requests_above_the_in_flight_cap_are_an_overload() -> None:
    engine = FakeEngine()
    engine.gate = threading.Event()
    async with Harness(engine, max_in_flight=1) as h:
        first = asyncio.create_task(h.infer(_plan(['detect'])))
        while h.service.in_flight == 0:
            await asyncio.sleep(0.01)
        with pytest.raises(InferenceServerException) as err:
            await h.infer(_plan(['detect']))
        engine.gate.set()
        assert (await first).as_numpy('status')[0] == b'ok'
    assert _code(err) == 'StatusCode.RESOURCE_EXHAUSTED'
    assert len(engine.seen) == 1


async def test_a_request_past_its_deadline_is_answered_with_the_stage_it_was_in(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Whatever blocks inside the engine, the caller gets an answer at the request deadline."""
    engine = FakeEngine()
    engine.gate = threading.Event()  # the photo never finishes on its own
    engine.stuck_in = 'embed_pe'
    try:
        async with Harness(engine, request_deadline=0.3) as h:
            start = time.monotonic()
            with pytest.raises(InferenceServerException) as err:
                await h.infer(_plan(['detect']))
            elapsed = time.monotonic() - start
            assert h.service.in_flight == 0
            engine.gate.set()
            engine.gate = None
            assert (await h.infer(_plan(['detect']))).as_numpy('status')[0] == b'ok'
            deadlines = {
                s.labels['stage']: s.value
                for m in h.service.metrics.deadlines.collect()
                for s in m.samples
                if s.name.endswith('_total')
            }
    finally:
        if engine.gate is not None:
            engine.gate.set()
    assert _code(err) == 'StatusCode.DEADLINE_EXCEEDED'
    assert 'embed_pe' in str(err.value)
    assert elapsed < 2
    assert deadlines == {'embed_pe': 1.0}
    assert any('embed_pe' in r.getMessage() for r in caplog.records)
    assert engine.seen[0].deadline - engine.seen[0].arrived == pytest.approx(0.3)
