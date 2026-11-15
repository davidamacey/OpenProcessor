"""Render the frame engine's protocol contract (plan 17, section 3, "Contract").

Two files under ``contracts/json/``, read by the native engine and by its contract test:

* ``frame_engine_v1.json``: the tensors, datatypes and shapes, the statuses and their order, the
  plan faults, the outputs of each branch, the decode routes, the request validation rules and the
  ping policy of protocol version 1. Every list that exists as a constant in the Python wire module
  or the API's channel options is read from there; what exists only in the engine's front (which
  the layering rule keeps this script from importing: the shapes, the order of its checks, the
  request validation messages, the server's ping interval) is written out here, each pinned to
  the front's behaviour by ``tests/test_frame_engine_contract.py``.
* ``frame_engine_plan_cases.json``: every plan text of ``frame_engine_plan_cases.py`` with the
  answer ``parse_plan`` gives (the parsed plan, or the fault).

``src/domain/frame_engine_wire.py`` stays the one definition. The native parser is compared with it
on every case.
"""

from __future__ import annotations

import json
import typing
from typing import TYPE_CHECKING, Any

from frame_engine_plan_cases import plan_cases


if TYPE_CHECKING:
    from collections.abc import Mapping


SCRIPT_REL = 'scripts/codegen/export_api_contracts.py'
SOURCES = [
    'src/domain/frame_engine_wire.py',
    'src/domain/decode_route.py',
    'src/clients/frame_engine.py',
    'frame_engine/front.py',
    'scripts/codegen/frame_engine_plan_cases.py',
]

#: Tensor shapes (``-1`` is a free dimension), the table ``frame_engine/front.py`` serves in
#: ``ModelMetadata``. Written out here because the layering rule lets nothing import
#: ``frame_engine/``; ``tests/test_frame_engine_contract.py`` compares it with the front's answer.
SHAPES: dict[str, list[int]] = {
    'image': [1],
    'plan': [1],
    'boxes': [-1, 4],
    'status': [1],
    'decode_route': [1],
    'branch_status': [1],
    'size': [2],
    'det_boxes': [-1, 4],
    'det_scores': [-1],
    'det_classes': [-1],
    'letterbox': [3],
    'whole_embedding': [-1],
    'crop_embeddings': [-1, -1],
    'crop_embedded': [-1],
    'blur_sums': [-1, 3],
    'crop_jpegs': [-1],
    'face_boxes': [-1, 4],
    'face_scores': [-1],
    'face_landmarks': [-1, 10],
    'face_blur_sums': [-1, 3],
    'face_overflow': [1],
    'face_embeddings': [-1, -1],
    'face_embedded': [-1],
    'ocr_quads': [-1, 8],
    'ocr_texts': [-1],
    'ocr_scores': [-1],
    'ocr_char_min': [-1],
    'ocr_det_scores': [-1],
    'ocr_overflow': [1],
    'derivative_jpegs': [-1],
}

#: ``ModelInfer`` checks in the order ``frame_engine/front.py`` runs them; the first that fails
#: answers. Pinned by ``tests/test_frame_engine_contract.py``.
CHECK_ORDER: list[dict[str, str]] = [
    {'check': 'unknown_model', 'status': 'NOT_FOUND'},
    {'check': 'request_and_plan', 'status': 'INVALID_ARGUMENT'},
    {'check': 'not_ready', 'status': 'UNAVAILABLE'},
    {'check': 'in_flight_cap', 'status': 'RESOURCE_EXHAUSTED'},
]

#: The request faults of ``frame_engine.front.parse_request``. Every one answers
#: ``INVALID_ARGUMENT`` with the message below as its detail (``{name}`` is the input's name); the
#: status is the contract, the wording is what the Python engine says. A plan that does not parse
#: answers the same status with its fault name first (``frame_engine_plan_cases.json``).
REQUEST_RULES: list[dict[str, str]] = [
    {'id': 'raw_contents_count', 'message': 'every input must travel in raw_input_contents'},
    {'id': 'unknown_input', 'message': "unknown input '{name}'"},
    {'id': 'missing_input', 'message': "missing input '{name}'"},
    {'id': 'bytes_input_shape', 'message': '{name} must be BYTES [1]'},
    {'id': 'bytes_input_one_element', 'message': '{name} must hold one element'},
    {'id': 'bytes_length_prefix', 'message': 'BYTES tensor ends inside a length prefix'},
    {'id': 'bytes_element_overrun', 'message': 'BYTES tensor element runs past the end'},
    {'id': 'boxes_datatype_shape', 'message': 'boxes must be FP64 [n, 4]'},
    {'id': 'boxes_byte_size', 'message': 'boxes byte size does not match its shape'},
]

#: The server side of the ping policy (``frame_engine.front.SERVER_OPTIONS``; pinned by
#: ``tests/test_frame_engine_contract.py``): it accepts a ping with no call in flight, this far
#: apart, a third of the API's period.
SERVER_MIN_PING_INTERVAL_MS = 10000

#: How engine failures map to gRPC statuses (``FrontService.ModelInfer``).
FAILURE_STATUSES: list[dict[str, str]] = [
    {'failure': 'overload', 'status': 'RESOURCE_EXHAUSTED'},
    {'failure': 'model_server_unavailable', 'status': 'UNAVAILABLE'},
    {'failure': 'cuda_context_lost', 'status': 'UNAVAILABLE'},
    {'failure': 'request_deadline', 'status': 'DEADLINE_EXCEEDED'},
    {'failure': 'engine_error', 'status': 'INTERNAL'},
]


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def _tensors(datatypes: Mapping[str, str]) -> list[dict[str, Any]]:
    return [{'name': n, 'datatype': d, 'shape': SHAPES[n]} for n, d in datatypes.items()]


def render_protocol() -> dict[str, Any]:
    from src.clients import frame_engine as client
    from src.domain import decode_route, frame_engine_wire as wire
    from src.domain.image_geometry import CROP_FILL

    model_version = str(wire.PROTOCOL_VERSION)
    input_tensors = _tensors(wire.INPUT_DATATYPES)
    for tensor in input_tensors:
        tensor['required'] = tensor['name'] != wire.BOXES
    return {
        'generated_by': SCRIPT_REL,
        'sources': SOURCES,
        'model': {
            'name': wire.MODEL_NAME,
            'version': model_version,
            'platform': wire.MODEL_NAME,
            'protocol_version': wire.PROTOCOL_VERSION,
            'engine_error_prefix': wire.ENGINE_ERROR_PREFIX,
        },
        'inputs': input_tensors,
        'outputs': _tensors(wire.OUTPUT_DATATYPES),
        'always_present_outputs': [wire.STATUS, wire.DECODE_ROUTE, wire.BRANCH_STATUS],
        'size_output': wire.SIZE,
        'outputs_filter': {'applies_to_every_output': True, 'empty_request_means_all': True},
        'implemented_calls': sorted(
            ['ServerLive', 'ServerReady', 'ModelReady', 'ModelMetadata', 'ModelInfer']
        ),
        'unimplemented_calls_status': 'UNIMPLEMENTED',
        'model_ready': {
            'versions_accepted': ['', model_version],
            'false_when': ['unknown_model', 'unknown_version', 'not_ready', 'draining'],
        },
        'check_order': CHECK_ORDER,
        'request_rules': REQUEST_RULES,
        'boxes_zero_rows_allowed': True,
        'failure_statuses': FAILURE_STATUSES,
        'photo_statuses': [wire.STATUS_OK, *decode_route.PREP_STATUSES],
        'decode_routes': [
            wire.DECODED_GPU,
            *(f'cpu_{reason}' for reason in decode_route.CPU_REASONS),
            wire.DECODED_NONE,
        ],
        'branches': list(wire.ENGINE_BRANCHES),
        'built_branches': sorted(wire.BUILT_BRANCHES),
        'branch_outputs': {name: list(names) for name, names in wire.BRANCH_OUTPUTS.items()},
        'crops_inline_outputs': list(wire.CROPS_OUTPUTS),
        'derivatives': {
            'field': wire.DERIVATIVES,
            'output': wire.DERIVATIVE_JPEGS,
            'formats': list(wire.RENDITION_FORMATS),
            'side_range': list(wire.RENDITION_SIDE_RANGE),
            'default_quality': wire.DEFAULT_RENDITION_QUALITY,
        },
        'branch_states': list(typing.get_args(wire.BranchState)),
        'branch_error_reasons': [
            wire.UNSUPPORTED_BRANCH,
            wire.MODEL_ERROR,
            wire.DEPENDENCY_FAILED,
            wire.IMAGE_FAILED,
        ],
        'plan': {
            'faults': list(typing.get_args(wire.PlanFault)),
            'crops_modes': list(typing.get_args(wire.CropsMode)),
            'frame_recipes': list(wire.FRAME_RECIPES),
            'crop_recipes': list(wire.CROP_RECIPES),
            'crop_fill_default': CROP_FILL,
            'ocr_det_limits': list(wire.DET_LIMITS),
            'ocr_max_batch': wire.OCR_MAX_BATCH,
            'ocr_max_det_side': wire.OCR_MAX_DET_SIDE,
            'json_rules': {
                'version_is_the_integer': wire.PROTOCOL_VERSION,
                'nan_and_infinity_literals': 'bad_json',
                'duplicate_keys': 'bad_json',
                'version_other_than_the_integer': 'bad_version',
            },
        },
        'ping_policy': {
            'client_keepalive_time_ms': client.ENGINE_KEEPALIVE_TIME_MS,
            'client_pings_without_calls': True,
            'server_permits_pings_without_calls': True,
            'server_min_ping_interval_ms': SERVER_MIN_PING_INTERVAL_MS,
            'server_sends_goaway_too_many_pings': False,
        },
    }


def render_plan_cases() -> dict[str, Any]:
    from src.domain import frame_engine_wire as wire

    cases = []
    for case_id, text, has_boxes in plan_cases():
        case: dict[str, Any] = {'id': case_id, 'plan': text, 'has_boxes': has_boxes}
        try:
            plan = wire.parse_plan(text, has_boxes=has_boxes)
        except wire.EnginePlanError as exc:
            case['fault'] = exc.fault
        else:
            case['parsed'] = json.loads(plan.to_json())
        cases.append(case)
    return {'generated_by': SCRIPT_REL, 'sources': SOURCES, 'cases': cases}


def render_protocol_json() -> str:
    return _dump(render_protocol())


def render_plan_cases_json() -> str:
    return _dump(render_plan_cases())
