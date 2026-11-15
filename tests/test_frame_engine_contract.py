"""The generated frame engine contract (``contracts/json/frame_engine_*.json``) says what the Python
front and wire module do.

What the generator reads from constants needs no test here (the drift hook keeps the files equal to
it). What it writes out by hand (the order of the front's checks, the request rules, the implemented
calls, the readiness rule) and the plan cases are pinned to the behaviour below, so the native
engine, held to the contract, is held to the Python engine.
"""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import CollectorRegistry
from tritonclient.grpc import service_pb2, service_pb2_grpc

from frame_engine import front
from frame_engine.stage_metrics import EngineMetrics
from src.domain import frame_engine_wire as wire


CONTRACTS = Path(__file__).resolve().parents[1] / 'contracts' / 'json'
PROTOCOL = json.loads((CONTRACTS / 'frame_engine_v1.json').read_text(encoding='utf-8'))
CASES = json.loads((CONTRACTS / 'frame_engine_plan_cases.json').read_text(encoding='utf-8'))[
    'cases'
]
PLAN = json.dumps(
    {
        'version': 1,
        'branches': ['detect'],
        'detector': {'model': 'det', 'input_size': 640, 'letterbox_fill': 114},
    }
)


class _AbortedError(Exception):
    def __init__(self, code: Any) -> None:
        self.code = code.name


class _Context:
    async def abort(self, code: Any, _details: str) -> None:
        raise _AbortedError(code)


class _NoEngine:
    def submit(self, _request: Any) -> Any:
        raise AssertionError('the check order is decided before the engine is reached')


def _service(*, ready: bool, max_in_flight: int) -> front.FrontService:
    return front.FrontService(
        _NoEngine(), lambda: ready, EngineMetrics(CollectorRegistry()),
        max_in_flight=max_in_flight, request_deadline=1.0,
    )  # fmt: skip


def _bytes(value: bytes) -> bytes:
    return struct.pack('<I', len(value)) + value


def _request(*, model: str = 'frame_engine', plan: str = PLAN, drop_plan: bool = False) -> Any:
    request = service_pb2.ModelInferRequest(model_name=model)
    names = [wire.IMAGE] if drop_plan else [wire.IMAGE, wire.PLAN]
    for name in names:
        tensor = request.inputs.add()
        tensor.name, tensor.datatype = name, 'BYTES'
        tensor.shape.append(1)
        request.raw_input_contents.append(
            _bytes(b'\xff\xd8' if name == wire.IMAGE else plan.encode())
        )
    return request


def _first_status(request: Any, *, ready: bool, max_in_flight: int) -> str:
    service = _service(ready=ready, max_in_flight=max_in_flight)
    with pytest.raises(_AbortedError) as err:
        asyncio.run(service.ModelInfer(request, _Context()))
    return str(err.value.code)


def test_the_front_answers_the_first_failing_check_in_the_contract_order() -> None:
    order = [step['status'] for step in PROTOCOL['check_order']]
    # each request fails its own check and every check after it
    seen = [
        _first_status(_request(model='other', drop_plan=True), ready=False, max_in_flight=0),
        _first_status(_request(drop_plan=True), ready=False, max_in_flight=0),
        _first_status(_request(), ready=False, max_in_flight=0),
        _first_status(_request(), ready=True, max_in_flight=0),
    ]
    assert seen == order


def _bad_request_detail(request: Any) -> str:
    with pytest.raises(front.BadRequestError) as err:
        front.parse_request(request)
    return str(err.value)


def _violations() -> dict[str, tuple[Any, str]]:
    """One request per rule id, and the rule's message with its ``{name}`` filled in."""

    def tensor(request: Any, name: str, datatype: str, shape: list[int], raw: bytes) -> None:
        t = request.inputs.add()
        t.name, t.datatype = name, datatype
        t.shape.extend(shape)
        request.raw_input_contents.append(raw)

    def good() -> Any:
        request = service_pb2.ModelInferRequest(model_name='frame_engine')
        tensor(request, wire.IMAGE, 'BYTES', [1], _bytes(b'\xff\xd8'))
        tensor(request, wire.PLAN, 'BYTES', [1], _bytes(PLAN.encode()))
        return request

    counts = good()
    counts.raw_input_contents.pop()
    unknown = good()
    tensor(unknown, 'extra', 'BYTES', [1], _bytes(b'x'))
    missing = service_pb2.ModelInferRequest(model_name='frame_engine')
    tensor(missing, wire.PLAN, 'BYTES', [1], _bytes(PLAN.encode()))
    shape = good()
    shape.inputs[0].shape[0] = 2
    elements = good()
    elements.raw_input_contents[0] = _bytes(b'a') + _bytes(b'b')
    prefix = good()
    prefix.raw_input_contents[0] = b'\x01\x00'
    overrun = good()
    overrun.raw_input_contents[0] = struct.pack('<I', 9) + b'ab'
    datatype = good()
    tensor(datatype, wire.BOXES, 'FP32', [1, 4], b'\x00' * 16)
    size = good()
    tensor(size, wire.BOXES, 'FP64', [2, 4], b'\x00' * 32)
    return {
        'raw_contents_count': (counts, ''),
        'unknown_input': (unknown, 'extra'),
        'missing_input': (missing, wire.IMAGE),
        'bytes_input_shape': (shape, wire.IMAGE),
        'bytes_input_one_element': (elements, wire.IMAGE),
        'bytes_length_prefix': (prefix, ''),
        'bytes_element_overrun': (overrun, ''),
        'boxes_datatype_shape': (datatype, ''),
        'boxes_byte_size': (size, ''),
    }


@pytest.mark.parametrize('rule', PROTOCOL['request_rules'], ids=lambda r: r['id'])
def test_each_request_rule_is_the_front_refusing_that_request(rule: dict[str, str]) -> None:
    request, name = _violations()[rule['id']]
    assert _bad_request_detail(request) == rule['message'].format(name=name)


def test_every_request_rule_has_a_case_and_zero_box_rows_are_accepted() -> None:
    assert {r['id'] for r in PROTOCOL['request_rules']} == set(_violations())
    assert PROTOCOL['boxes_zero_rows_allowed'] is True
    request = _request(
        plan=json.dumps(
            {
                'version': 1,
                'branches': ['embed_frame'],
                'embed': {'model': 'pe', 'size': 336},
                'crops': 'inline',
            }
        )
    )
    boxes = request.inputs.add()
    boxes.name, boxes.datatype = wire.BOXES, 'FP64'
    boxes.shape.extend([0, 4])
    request.raw_input_contents.append(b'')
    _image, _plan, rows = front.parse_request(request)
    assert rows is not None
    assert rows.shape == (0, 4)


def test_the_implemented_calls_are_the_ones_the_front_overrides() -> None:
    base = service_pb2_grpc.GRPCInferenceServiceServicer
    overridden = {
        name
        for name in vars(front.FrontService)
        if hasattr(base, name) and not name.startswith('_')
    }
    assert sorted(overridden) == PROTOCOL['implemented_calls']
    for name in set(dir(base)) - overridden:
        if name[0].isupper():
            assert getattr(front.FrontService, name) is getattr(base, name)


def _model_ready(name: str, version: str, *, ready: bool, draining: bool = False) -> bool:
    service = _service(ready=ready, max_in_flight=1)
    service.draining = draining
    answer = asyncio.run(
        service.ModelReady(service_pb2.ModelReadyRequest(name=name, version=version), _Context())
    )
    return bool(answer.ready)


def test_model_ready_is_true_only_for_the_known_model_and_version_when_ready() -> None:
    rule = PROTOCOL['model_ready']
    model = PROTOCOL['model']
    assert all(_model_ready(model['name'], v, ready=True) for v in rule['versions_accepted'])
    assert not _model_ready('other', '', ready=True)
    assert not _model_ready(model['name'], '2', ready=True)
    assert not _model_ready(model['name'], '', ready=False)
    assert not _model_ready(model['name'], '', ready=True, draining=True)
    assert set(rule['false_when']) == {'unknown_model', 'unknown_version', 'not_ready', 'draining'}


def test_metadata_is_the_contract_tensors_and_platform() -> None:
    service = _service(ready=True, max_in_flight=1)
    answer = asyncio.run(
        service.ModelMetadata(service_pb2.ModelMetadataRequest(name='frame_engine'), _Context())
    )
    model = PROTOCOL['model']
    assert (answer.name, list(answer.versions), answer.platform) == (
        model['name'], [model['version']], model['platform'])  # fmt: skip
    for side, mine in ((answer.inputs, PROTOCOL['inputs']), (answer.outputs, PROTOCOL['outputs'])):
        assert [(t.name, t.datatype, list(t.shape)) for t in side] == [
            (t['name'], t['datatype'], t['shape']) for t in mine
        ]


def test_every_plan_fault_has_a_case_and_every_case_is_what_parse_plan_answers() -> None:
    faults = {c['fault'] for c in CASES if 'fault' in c}
    assert faults == set(PROTOCOL['plan']['faults'])
    for case in CASES:
        if 'fault' in case:
            with pytest.raises(wire.EnginePlanError) as err:
                wire.parse_plan(case['plan'], has_boxes=case['has_boxes'])
            assert err.value.fault == case['fault'], case['id']
            if case['fault'] == 'bad_json':
                assert 'message' not in case, case['id']
            else:
                assert str(err.value) == case['message'], case['id']
        else:
            plan = wire.parse_plan(case['plan'], has_boxes=case['has_boxes'])
            assert case['parsed'] == json.loads(plan.to_json()), case['id']


def test_the_four_quirks_are_cases_with_the_answers_the_plan_orders() -> None:
    by_id = {c['id']: c for c in CASES}
    quirks = {k: v for k, v in by_id.items() if k.startswith('quirk_')}
    assert {v['fault'] for k, v in quirks.items() if 'version' in k and 'duplicate' not in k
            and 'nan' not in k and 'infinity' not in k} == {'bad_version'}  # fmt: skip
    for key in ('quirk_nan_size', 'quirk_infinity_fill', 'quirk_minus_infinity_version',
                'quirk_duplicate_version', 'quirk_duplicate_nested'):  # fmt: skip
        assert quirks[key]['fault'] == 'bad_json'
    assert {'quirk_version_true', 'quirk_version_one_point_zero'} <= set(quirks)
    assert 'fault' not in by_id['same_key_in_different_objects_is_fine']


def test_the_json_rules_in_the_contract_name_the_faults_the_parser_raises() -> None:
    rules = PROTOCOL['plan']['json_rules']
    assert rules == {
        'version_is_the_integer': 1,
        'nan_and_infinity_literals': 'bad_json',
        'duplicate_keys': 'bad_json',
        'version_other_than_the_integer': 'bad_version',
    }
    assert rules['version_is_the_integer'] == wire.PROTOCOL_VERSION


def test_the_ping_policy_is_what_the_front_and_the_api_channels_are_set_to() -> None:
    from src.clients import frame_engine as client

    policy = PROTOCOL['ping_policy']
    server = dict(front.SERVER_OPTIONS)
    assert policy['client_keepalive_time_ms'] == client.ENGINE_KEEPALIVE_TIME_MS
    assert dict(client.ENGINE_CHANNEL_OPTIONS)['grpc.keepalive_permit_without_calls'] == 1
    assert policy['client_pings_without_calls'] is True
    assert (server['grpc.keepalive_permit_without_calls'] == 1) is policy[
        'server_permits_pings_without_calls'
    ]
    assert (
        server['grpc.http2.min_ping_interval_without_data_ms']
        == policy['server_min_ping_interval_ms']
    )
    assert policy['server_min_ping_interval_ms'] <= policy['client_keepalive_time_ms']


def test_the_model_version_is_the_protocol_version() -> None:
    assert PROTOCOL['model']['version'] == front.MODEL_VERSION == str(wire.PROTOCOL_VERSION)
