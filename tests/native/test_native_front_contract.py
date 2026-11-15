"""The native engine's front is the Python engine's, on the wire (plan 17 section 5 N0).

Each case runs against the release binary (``op-frame-engine --pipeline stub``, no GPU) and, where
the Python front can be started the same way (its real ``FrontService`` over a stand-in that
answers as the stub does: ``engines.py``), against the Python engine too: one parametrized case,
the same assertions. The API's client (``src/clients/frame_engine.py``) and ``tritonclient`` drive
both, so what the API relies on is what is checked: every status and message, the order of the
checks, the outputs filter, readiness while draining, and the engine's error prefix.

Run with ``pytest -m native tests/native`` (CI builds the release binary first; locally the fixture
runs ``cargo build --release``).
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np
import pytest
import pytest_asyncio
from tritonclient.grpc import InferInput, InferRequestedOutput, aio, service_pb2
from tritonclient.utils import InferenceServerException

from src.clients import frame_engine as client
from src.domain import frame_engine_wire as wire

from .engines import (
    FRAME,
    INGEST,
    PHOTO,
    Engine,
    inputs,
    plan_text,
    raw_of,
    script,
    shape_of,
    stub_outputs,
)


if TYPE_CHECKING:
    from .conftest import StartEngine

pytestmark = [pytest.mark.native, pytest.mark.asyncio(loop_scope='function')]

CONTRACTS = Path(__file__).resolve().parents[2] / 'contracts' / 'json'
PROTOCOL = json.loads((CONTRACTS / 'frame_engine_v1.json').read_text(encoding='utf-8'))
CASES = json.loads((CONTRACTS / 'frame_engine_plan_cases.json').read_text(encoding='utf-8'))[
    'cases'
]


class Triton:
    """``tritonclient``'s asyncio client on an engine, for the cases that go through it."""

    def __init__(self, engine: Engine) -> None:
        self.client = aio.InferenceServerClient(engine.target)

    async def infer(self, image: bytes, plan: str, *, boxes: np.ndarray | None = None,
                    outputs: list[str] | None = None, model: str = 'frame_engine') -> Any:  # fmt: skip
        wanted = [InferRequestedOutput(n) for n in outputs] if outputs else None
        return await self.client.infer(model, inputs(image, plan, boxes), outputs=wanted)

    async def close(self) -> None:
        await self.client.close()


@pytest_asyncio.fixture
async def triton() -> Any:
    opened: list[Triton] = []

    def open_(engine: Engine) -> Triton:
        opened.append(Triton(engine))
        return opened[-1]

    yield open_
    for t in opened:
        await t.close()


def code_of(err: pytest.ExceptionInfo[InferenceServerException]) -> str:
    return str(err.value.status())


# ------------------------------------------------------------------------------ what is answered


async def test_bytes_round_trip_in_both_directions(
    kind: str, start: StartEngine, triton: Any
) -> None:
    """BYTES elements keep their lengths and bytes in a request and in the answers (a rendition
    name with non-ASCII text comes back in a JPEG, the OCR text 'café' as UTF-8)."""
    engine = await start(kind)
    plan = plan_text(
        ['ocr'], derivatives=[{'name': 'thümb-é', 'side': 64}, {'name': 'b', 'side': 128}]
    )
    image = bytes(range(256)) * 4  # every byte value, NULs included, no script
    result = await triton(engine).infer(image, plan)
    assert [bytes(x) for x in result.as_numpy('ocr_texts')] == [b'STOP', 'café'.encode()]
    assert [bytes(x) for x in result.as_numpy('derivative_jpegs')] == [
        b'\xff\xd8' + 'thümb-é'.encode(), b'\xff\xd8b']  # fmt: skip
    assert result.as_numpy('status')[0] == b'ok'
    assert result.as_numpy('decode_route')[0] == b'gpu'


@pytest.mark.parametrize(
    ('branches', 'extra'),
    [
        (['detect'], {}),
        (['embed_frame'], {}),
        (['ocr'], {}),
        (['faces'], {}),
        (['faces', 'face_embed'], {}),
        (['detect', 'embed_crops'], {'crops': 'inline'}),
        (['detect', 'embed_frame', 'embed_crops'], {'crops': 'inline'}),
        (['detect'], {'crops': 'inline'}),
        (['detect', 'ocr'], {'if_detected': {'ocr': [2]}}),
        (['detect'], {'derivatives': [{'name': 'thumb', 'side': 256}]}),
    ],
)
async def test_each_branch_combination_answers_its_tensors(
    kind: str, start: StartEngine, triton: Any, branches: list[str], extra: dict[str, Any]
) -> None:
    engine = await start(kind)
    text = plan_text(branches, **extra)
    result = await triton(engine).infer(PHOTO, text)
    plan = wire.parse_plan(text, has_boxes=False)
    expected = stub_outputs(plan, 2)
    response = result.get_response()
    got = {o.name: (o.datatype, list(o.shape)) for o in response.outputs}
    always = {
        'status': ('BYTES', [1]),
        'decode_route': ('BYTES', [1]),
        'branch_status': ('BYTES', [1]),
    }
    assert got == {
        **always,
        **{n: (wire.OUTPUT_DATATYPES[n], shape_of(n, v)) for n, v in expected.items()},
    }
    raws = dict(zip((o.name for o in response.outputs), response.raw_output_contents, strict=True))
    for name, value in expected.items():
        assert raws[name] == raw_of(name, value), name
    states = wire.decode_branch_status(result.as_numpy('branch_status')[0])
    assert states == dict.fromkeys(plan.reported(), ('ok', ''))


async def test_branch_status_is_written_as_json_dumps_with_sorted_keys(
    kind: str, start: StartEngine, triton: Any
) -> None:
    engine = await start(kind)
    result = await triton(engine).infer(PHOTO, INGEST)
    assert (
        bytes(result.as_numpy('branch_status')[0])
        == wire.encode_branch_status(
            dict.fromkeys(['detect', 'embed_crops', 'embed_frame'], ('ok', ''))
        ).encode()
    )
    refused = await triton(engine).infer(script(status='corrupt'), INGEST)
    assert refused.as_numpy('status')[0] == b'corrupt'
    assert refused.as_numpy('decode_route')[0] == b'none'
    assert refused.as_numpy('size') is None, 'nothing was decoded: no size, no tensor'
    assert wire.decode_branch_status(refused.as_numpy('branch_status')[0]) == dict.fromkeys(
        ['detect', 'embed_crops', 'embed_frame'], ('error', wire.IMAGE_FAILED)
    )


async def test_stored_boxes_reach_the_engine_as_fp64_rows(
    kind: str, start: StartEngine, triton: Any
) -> None:
    engine = await start(kind)
    text = plan_text(['embed_frame', 'embed_crops'])
    boxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.2, 0.9, 0.9], [0.0, 0.0, 1.0, 1.0]], np.float64)
    result = await triton(engine).infer(PHOTO, text, boxes=boxes)
    assert result.as_numpy('crop_embeddings').shape == (3, 4)
    assert list(result.as_numpy('crop_embedded')) == [True, True, True]
    zero = await triton(engine).infer(
        PHOTO, plan_text(['embed_frame'], crops='inline'), boxes=np.zeros((0, 4), np.float64)
    )
    assert zero.as_numpy('crop_jpegs').shape == (0,), 'a reprocess sends no rows and gets none'


async def test_the_outputs_filter_applies_to_every_output(
    kind: str, start: StartEngine, triton: Any
) -> None:
    engine = await start(kind)
    t = triton(engine)
    only = await t.infer(PHOTO, INGEST, outputs=['det_scores', 'status', 'branch_status'])
    names = {o.name for o in only.get_response().outputs}
    assert names == {'det_scores', 'status', 'branch_status'}
    route = await t.infer(PHOTO, INGEST, outputs=['decode_route'])
    assert [o.name for o in route.get_response().outputs] == ['decode_route']
    nothing_known = await t.infer(PHOTO, INGEST, outputs=['not_an_output'])
    assert list(nothing_known.get_response().outputs) == []
    everything = await t.infer(PHOTO, INGEST)
    assert {o.name for o in everything.get_response().outputs} >= {
        'status',
        'size',
        'det_boxes',
        'crop_jpegs',
    }


async def test_the_request_id_and_the_model_version_are_echoed(
    kind: str, start: StartEngine
) -> None:
    engine = await start(kind)
    request = service_pb2.ModelInferRequest(model_name='frame_engine', id='photo-7')
    for tensor in inputs(PHOTO, INGEST):
        request.inputs.append(tensor._get_tensor())
        request.raw_input_contents.append(tensor._get_content())
    reply = await engine.stub().ModelInfer(request)
    assert (reply.model_name, reply.model_version, reply.id) == ('frame_engine', '1', 'photo-7')


# ------------------------------------------------------------------------------ what is refused


async def raw_call(engine: Engine, request: Any) -> tuple[grpc.StatusCode, str]:
    try:
        await engine.stub().ModelInfer(request)
    except grpc.aio.AioRpcError as exc:
        return exc.code(), exc.details() or ''
    return grpc.StatusCode.OK, ''


def _bytes_element(value: bytes) -> bytes:
    return struct.pack('<I', len(value)) + value


def _violations() -> dict[str, tuple[Any, str]]:
    """One request per rule of the contract, and the name its message names."""

    def tensor(request: Any, name: str, datatype: str, shape: list[int], raw: bytes) -> None:
        t = request.inputs.add()
        t.name, t.datatype = name, datatype
        t.shape.extend(shape)
        request.raw_input_contents.append(raw)

    def good() -> Any:
        request = service_pb2.ModelInferRequest(model_name='frame_engine')
        tensor(request, wire.IMAGE, 'BYTES', [1], _bytes_element(b'\xff\xd8'))
        tensor(request, wire.PLAN, 'BYTES', [1], _bytes_element(FRAME.encode()))
        return request

    counts = good()
    counts.raw_input_contents.pop()
    unknown = good()
    tensor(unknown, 'extra', 'BYTES', [1], _bytes_element(b'x'))
    missing = service_pb2.ModelInferRequest(model_name='frame_engine')
    tensor(missing, wire.PLAN, 'BYTES', [1], _bytes_element(FRAME.encode()))
    shape = good()
    shape.inputs[0].shape[0] = 2
    elements = good()
    elements.raw_input_contents[0] = _bytes_element(b'a') + _bytes_element(b'b')
    prefix = good()
    prefix.raw_input_contents[0] = b'\x01\x00'
    overrun = good()
    overrun.raw_input_contents[0] = struct.pack('<I', 9) + b'ab'
    datatype = good()
    tensor(datatype, wire.BOXES, 'FP32', [1, 4], b'\x00' * 16)
    size = good()
    tensor(size, wire.BOXES, 'FP64', [2, 4], b'\x00' * 32)
    return {
        'raw_contents_count': (counts, ''), 'unknown_input': (unknown, 'extra'),
        'missing_input': (missing, wire.IMAGE), 'bytes_input_shape': (shape, wire.IMAGE),
        'bytes_input_one_element': (elements, wire.IMAGE), 'bytes_length_prefix': (prefix, ''),
        'bytes_element_overrun': (overrun, ''), 'boxes_datatype_shape': (datatype, ''),
        'boxes_byte_size': (size, ''),
    }  # fmt: skip


@pytest.mark.parametrize('rule', PROTOCOL['request_rules'], ids=lambda r: r['id'])
async def test_each_request_rule_is_invalid_argument_with_the_contracts_text(
    kind: str, start: StartEngine, rule: dict[str, str]
) -> None:
    engine = await start(kind)
    request, name = _violations()[rule['id']]
    code, details = await raw_call(engine, request)
    assert code == grpc.StatusCode.INVALID_ARGUMENT
    assert details == rule['message'].format(name=name)


@pytest.mark.parametrize(
    ('text', 'boxes'),
    [
        ('{"version": 1}', None),
        (plan_text(['embed_crops']), None),
        (plan_text(['detect']), np.zeros((1, 4), np.float64)),
        (plan_text(['embed_frame', 'embed_crops']), np.zeros((1, 3), np.float64)),
        (plan_text(['embed_frame', 'embed_crops']), np.zeros((1, 4), np.float32)),
    ],
)
async def test_a_bad_request_or_plan_is_invalid_argument(
    kind: str, start: StartEngine, triton: Any, text: str, boxes: np.ndarray | None
) -> None:
    engine = await start(kind)
    t = triton(engine)
    tensors = inputs(PHOTO, text)
    if boxes is not None:
        wrong = InferInput(
            wire.BOXES, list(boxes.shape), 'FP32' if boxes.dtype == np.float32 else 'FP64'
        )
        wrong.set_data_from_numpy(boxes)
        tensors.append(wrong)
    with pytest.raises(InferenceServerException) as err:
        await t.client.infer('frame_engine', tensors)
    assert code_of(err) == 'StatusCode.INVALID_ARGUMENT'


async def test_an_unknown_model_is_not_found(kind: str, start: StartEngine, triton: Any) -> None:
    engine = await start(kind)
    with pytest.raises(InferenceServerException) as err:
        await triton(engine).infer(PHOTO, FRAME, model='some_other_model')
    assert code_of(err) == 'StatusCode.NOT_FOUND'
    assert err.value.message() == "unknown model 'some_other_model'"


async def test_a_call_the_engine_does_not_answer_is_unimplemented(
    kind: str, start: StartEngine
) -> None:
    """``LoadModel``, which the API's pool sends when the engine is not ready under the ingest
    load profile, and the model server's other calls."""
    engine = await start(kind)
    calls = [
        engine.stub().RepositoryModelLoad(
            service_pb2.RepositoryModelLoadRequest(model_name='frame_engine')
        ),
        engine.stub().ServerMetadata(service_pb2.ServerMetadataRequest()),
        engine.stub().ModelStatistics(service_pb2.ModelStatisticsRequest(name='frame_engine')),
    ]
    for call in calls:
        with pytest.raises(grpc.aio.AioRpcError) as err:
            await call
        assert err.value.code() == grpc.StatusCode.UNIMPLEMENTED


def _plan_request(case: dict[str, Any]) -> Any:
    request = service_pb2.ModelInferRequest(model_name='frame_engine')
    for name, value in ((wire.IMAGE, PHOTO), (wire.PLAN, case['plan'].encode())):
        t = request.inputs.add()
        t.name, t.datatype = name, 'BYTES'
        t.shape.append(1)
        request.raw_input_contents.append(_bytes_element(value))
    if case['has_boxes']:
        t = request.inputs.add()
        t.name, t.datatype = wire.BOXES, 'FP64'
        t.shape.extend([0, 4])
        request.raw_input_contents.append(b'')
    return request


async def _answer(engine: Engine, request: Any) -> tuple[Any, ...]:
    try:
        reply = await engine.stub().ModelInfer(request)
    except grpc.aio.AioRpcError as exc:
        return exc.code(), exc.details()
    names = [o.name for o in reply.outputs]
    shapes = {o.name: (o.datatype, list(o.shape)) for o in reply.outputs}
    return (
        grpc.StatusCode.OK,
        sorted(names),
        shapes,
        dict(zip(names, reply.raw_output_contents, strict=True)),
    )


async def test_every_plan_case_is_answered_alike_by_both_engines(start: StartEngine) -> None:
    """The generated plan cases through the wire: the same status, the same message, and for an
    accepted plan the same tensors to the byte, from the native binary and the Python front."""
    rust, python = await start('rust'), await start('python')
    differ: list[str] = []
    for case in CASES:
        request = _plan_request(case)
        a, b = await _answer(rust, request), await _answer(python, request)
        if 'fault' in case:
            want = (grpc.StatusCode.INVALID_ARGUMENT, case.get('message'))
            same_text = (
                a[1] == b[1] == want[1]
                if case['fault'] != 'bad_json'
                else a[1].startswith('bad_json')
            )
            if not (a[0] == b[0] == want[0] and same_text):
                differ.append(f'{case["id"]}: native {a} python {b}')
        elif a != b or a[0] != grpc.StatusCode.OK:
            differ.append(f'{case["id"]}: native {a[:2]} python {b[:2]}')
    assert not differ, f'{len(differ)} of {len(CASES)} cases differ:\n' + '\n'.join(differ)


# ------------------------------------------------------------------------------ readiness


async def test_readiness_follows_the_pipeline_and_draining(
    kind: str, start: StartEngine, triton: Any
) -> None:
    engine = await start(kind, request_deadline=10.0)
    t = triton(engine)
    assert await t.client.is_server_live()
    assert await t.client.is_server_ready()
    assert await t.client.is_model_ready('frame_engine')
    assert await t.client.is_model_ready('frame_engine', '1')
    assert not await t.client.is_model_ready('frame_engine', '2'), 'an unknown version'
    assert not await t.client.is_model_ready('some_other_model')
    await engine.set_ready(False)  # the model server went away or the region is not registered
    assert not await t.client.is_server_ready()
    assert not await t.client.is_model_ready('frame_engine')
    with pytest.raises(InferenceServerException) as err:
        await t.infer(PHOTO, FRAME)
    assert code_of(err) == 'StatusCode.UNAVAILABLE'
    assert err.value.message() == 'frame engine not ready'
    await engine.set_ready(True)
    assert await t.client.is_model_ready('frame_engine')
    held = asyncio.create_task(t.infer(script(hold_ms=1500), FRAME))
    await asyncio.sleep(0.3)
    await engine.begin_drain()
    await _wait_for(lambda: _not_ready(t), 'draining turns readiness false')
    assert await t.client.is_server_live(), 'a draining engine is alive'
    assert not await t.client.is_model_ready('frame_engine')
    with pytest.raises(InferenceServerException) as err:
        await t.infer(PHOTO, FRAME)
    assert code_of(err) == 'StatusCode.UNAVAILABLE'
    assert (await held).as_numpy('status')[0] == b'ok', (
        'a photo in flight when draining began is answered'
    )


async def _not_ready(t: Triton) -> bool:
    return not await t.client.is_server_ready()


async def _wait_for(check: Any, what: str, *, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if await check():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f'{what}: not within {timeout} s')


async def test_the_checks_run_in_the_contract_order(
    kind: str, start: StartEngine, triton: Any
) -> None:
    engine = await start(kind, max_in_flight=1, request_deadline=20.0)
    t = triton(engine)
    held = asyncio.create_task(t.infer(script(hold_ms=4000), FRAME))
    await _wait_for(lambda: _full(engine), 'the cap is full')
    await engine.set_ready(False)

    async def status(model: str, plan: str) -> str:
        with pytest.raises(InferenceServerException) as err:
            await t.infer(PHOTO, plan, model=model)
        return code_of(err)

    seen = [
        await status('other', '{'),  # every check fails: the first answers
        await status('frame_engine', '{'),
        await status('frame_engine', FRAME),  # not ready before the cap
    ]
    await engine.set_ready(True)
    seen.append(await status('frame_engine', FRAME))
    assert seen == [f'StatusCode.{step["status"]}' for step in PROTOCOL['check_order']]
    held.cancel()


async def _full(engine: Engine, cap: int = 1) -> bool:
    return await engine.in_flight() >= cap


async def test_metadata_is_the_contracts_tensors_and_platform(
    kind: str, start: StartEngine, triton: Any
) -> None:
    engine = await start(kind)
    meta = await triton(engine).client.get_model_metadata('frame_engine', as_json=True)
    model = PROTOCOL['model']
    assert (meta['name'], meta['versions'], meta['platform']) == (
        model['name'],
        [model['version']],
        model['platform'],
    )

    def table(side: list[dict[str, Any]]) -> list[tuple[str, str, list[int]]]:
        return [(t['name'], t['datatype'], [int(d) for d in t.get('shape', [])]) for t in side]

    assert table(meta['inputs']) == [
        (t['name'], t['datatype'], t['shape']) for t in PROTOCOL['inputs']
    ]
    assert table(meta['outputs']) == [
        (t['name'], t['datatype'], t['shape']) for t in PROTOCOL['outputs']
    ]
    with pytest.raises(InferenceServerException) as err:
        await triton(engine).client.get_model_metadata('some_other_model')
    assert code_of(err) == 'StatusCode.NOT_FOUND'


# ------------------------------------------------------------------------------ failures


@pytest.mark.parametrize(
    ('word', 'code', 'message'),
    [
        ('overload', 'RESOURCE_EXHAUSTED', 'overload: budget: stub'),
        ('unavailable', 'UNAVAILABLE', 'model server unavailable: stub: down'),
        ('context_lost', 'UNAVAILABLE', 'frame engine restarting: stub: lost'),
        ('internal', 'INTERNAL', 'frame engine: stub failure'),
    ],
)
async def test_engine_failures_map_to_their_status_codes(
    kind: str, start: StartEngine, triton: Any, word: str, code: str, message: str
) -> None:
    engine = await start(kind)
    with pytest.raises(InferenceServerException) as err:
        await triton(engine).infer(script(fail=word), FRAME)
    assert code_of(err) == f'StatusCode.{code}'
    assert err.value.message() == message
    ok = await triton(engine).infer(PHOTO, FRAME)
    assert ok.as_numpy('status')[0] == b'ok', 'the slot of a failed photo is free again'


async def test_the_engine_error_prefix_survives_grpc_message_encoding(
    kind: str, start: StartEngine, triton: Any
) -> None:
    """The text after the prefix holds a space, a colon, a percent sign, the characters percent
    encoding is written for, and non-ASCII text (grpc-message is percent-encoded on the wire)."""
    engine = await start(kind)
    with pytest.raises(InferenceServerException) as err:
        await triton(engine).infer(script(stage='café%25 100%'), FRAME)
    assert code_of(err) == 'StatusCode.INTERNAL'
    message = err.value.message()
    assert message.startswith(PROTOCOL['model']['engine_error_prefix'] + 'stub photo: ')
    assert 'café%25 100%' in message


async def test_requests_above_the_in_flight_cap_are_an_overload(
    kind: str, start: StartEngine, triton: Any
) -> None:
    engine = await start(kind, max_in_flight=1, request_deadline=10.0)
    t = triton(engine)
    first = asyncio.create_task(t.infer(script(hold_ms=1000), FRAME))
    await _wait_for(lambda: _full(engine), 'the cap is full')
    with pytest.raises(InferenceServerException) as err:
        await t.infer(PHOTO, FRAME)
    assert code_of(err) == 'StatusCode.RESOURCE_EXHAUSTED'
    assert err.value.message() == 'overload: 1 requests in flight (cap 1)'
    assert (await first).as_numpy('status')[0] == b'ok'
    assert (await t.infer(PHOTO, FRAME)).as_numpy('status')[0] == b'ok'
    assert (await engine.overloads()).get('in_flight', 0) >= 1


async def test_a_request_past_its_deadline_is_answered_with_the_stage_it_was_in(
    kind: str, start: StartEngine, triton: Any
) -> None:
    engine = await start(kind, request_deadline=1.0)
    t = triton(engine)
    started = time.monotonic()
    with pytest.raises(InferenceServerException) as err:
        await t.infer(script(hold_ms=4000, stage='embed_pe'), FRAME)
    elapsed = time.monotonic() - started
    assert code_of(err) == 'StatusCode.DEADLINE_EXCEEDED'
    assert err.value.message() == 'frame engine: no answer within 1 s (stage embed_pe)'
    assert 0.9 < elapsed < 3.0
    assert await engine.deadlines() == {'embed_pe': 1.0}
    assert (await t.infer(PHOTO, FRAME)).as_numpy('status')[0] == b'ok', (
        'and the next photo is served'
    )


async def test_a_message_above_the_limit_answers_resource_exhausted(
    kind: str, start: StartEngine, triton: Any
) -> None:
    engine = await start(kind, max_message_mib=1)
    request = service_pb2.ModelInferRequest(model_name='frame_engine')
    for name, value in ((wire.IMAGE, b'\x00' * (2 * 1024 * 1024)), (wire.PLAN, FRAME.encode())):
        t = request.inputs.add()
        t.name, t.datatype = name, 'BYTES'
        t.shape.append(1)
        request.raw_input_contents.append(_bytes_element(value))
    code, _details = await raw_call(engine, request)
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED
    small = await triton(engine).infer(PHOTO, FRAME)
    assert small.as_numpy('status')[0] == b'ok'


async def test_ten_megabyte_photos_go_through_both_ways(
    kind: str, start: StartEngine, triton: Any
) -> None:
    engine = await start(kind, max_message_mib=64)
    image = np.random.default_rng(7).integers(0, 256, 10 * 1000 * 1000, np.uint8).tobytes()
    t = triton(engine)
    results = await asyncio.gather(*[t.infer(image, INGEST) for _ in range(4)])
    assert all(r.as_numpy('status')[0] == b'ok' for r in results)
    assert all(list(r.as_numpy('size')) == [96, 64] for r in results)


# ------------------------------------------------------------------------------ the API's client


@pytest_asyncio.fixture
async def pool(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The API's pool on an engine's address with the API's production channel options."""
    opened: list[Any] = []

    async def open_(engine: Engine) -> Any:
        monkeypatch.setenv(client.FRAME_ENGINE_URL_ENV, engine.target)
        made = await client.open_pool(pool_size=2, max_concurrent=4)
        opened.append(made)
        return made

    yield open_
    for made in opened:
        await made.close()


PLAN_OBJ = wire.parse_plan(INGEST, has_boxes=False)


async def test_the_api_client_gets_the_answer_of_an_ingest_plan(
    kind: str, start: StartEngine, pool: Any
) -> None:
    engine = await start(kind)
    answer = await client.process(await pool(engine), PHOTO, PLAN_OBJ)
    assert answer.ok
    assert answer.decode_route == 'gpu'
    assert list(answer.outputs[wire.SIZE]) == [96, 64]
    assert [bytes(j) for j in answer.outputs[wire.CROP_JPEGS]] == [b'\xff\xd8a', b'\xff\xd8bc']
    refused = await client.process(await pool(engine), script(status='huge'), PLAN_OBJ)
    assert (refused.status, refused.ok, refused.outputs) == ('huge', False, {})


@pytest.mark.parametrize(
    ('word', 'attempts', 'code', 'overloaded'),
    [
        ('unavailable', 1 + client.RETRIES, 'UNAVAILABLE', False),  # retried RETRIES times
        ('internal', 1, 'INTERNAL', False),  # the engine's own failure is not retried
    ],
)
async def test_the_api_client_retries_what_it_should_and_nothing_else(
    kind: str, start: StartEngine, pool: Any, monkeypatch: pytest.MonkeyPatch,
    word: str, attempts: int, code: str, overloaded: bool,
) -> None:  # fmt: skip
    async def instant(_s: float) -> None:
        return None

    monkeypatch.setattr(client, 'RETRY_SLEEP', instant)
    engine = await start(kind)
    with pytest.raises(client.EngineCallError) as err:
        await client.process(await pool(engine), script(fail=word), PLAN_OBJ)
    assert (err.value.attempts, err.value.code, err.value.overloaded) == (
        attempts,
        code,
        overloaded,
    )


async def test_the_api_client_keeps_retrying_an_overload_until_the_photos_deadline(
    kind: str, start: StartEngine, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client, 'DEADLINE_SECONDS', 0.6)
    engine = await start(kind)
    with pytest.raises(client.EngineCallError) as err:
        await client.process(await pool(engine), script(fail='overload'), PLAN_OBJ)
    assert err.value.overloaded
    assert err.value.attempts >= 3


async def test_the_api_client_does_not_retry_a_request_that_ran_out_of_time(
    kind: str, start: StartEngine, pool: Any
) -> None:
    engine = await start(kind, request_deadline=1.0)
    with pytest.raises(client.EngineCallError) as err:
        await client.process(await pool(engine), script(hold_ms=4000), PLAN_OBJ)
    assert (err.value.attempts, err.value.code) == (1, 'DEADLINE_EXCEEDED')


async def test_the_api_client_retries_an_over_limit_message_as_an_overload(
    kind: str, start: StartEngine, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client, 'DEADLINE_SECONDS', 0.5)
    engine = await start(kind, max_message_mib=1)
    with pytest.raises(client.EngineCallError) as err:
        await client.process(await pool(engine), b'\x00' * (2 * 1024 * 1024), PLAN_OBJ)
    assert err.value.overloaded
    assert err.value.code == 'RESOURCE_EXHAUSTED'


async def test_the_readiness_probe_asks_once_and_believes_the_answer(
    kind: str, start: StartEngine, pool: Any
) -> None:
    engine = await start(kind)
    made = await pool(engine)
    ready = await client.probe_ready(made)
    assert (ready.outcome, ready.attempts, ready.ready) == (client.PROBE_READY, 1, True)
    await engine.set_ready(False)
    not_ready = await client.probe_ready(made)
    assert (not_ready.outcome, not_ready.attempts, not_ready.ready) == (
        client.PROBE_NOT_READY,
        1,
        False,
    )
    await engine.set_ready(True)
    assert (await client.probe_ready(made)).ready


async def test_an_engine_that_refuses_connections_is_judged_undelivered_within_two_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .engines import free_ports

    monkeypatch.setenv(client.FRAME_ENGINE_URL_ENV, f'127.0.0.1:{free_ports(1)[0]}')
    made = await client.open_pool(pool_size=3, max_concurrent=4)
    try:
        started = time.monotonic()
        readiness = await client.probe_ready(made)
        elapsed = time.monotonic() - started
    finally:
        await made.close()
    assert readiness.outcome == client.PROBE_UNDELIVERED
    assert elapsed < 2.0
