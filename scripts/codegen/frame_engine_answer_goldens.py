#!/usr/bin/env python3
"""Engine answers as the Python engine writes them and the Python client reads them (plan 18a, R1e).

For each case this script builds the answer the Python front sends
(``frame_engine.front.build_response`` on an ``EngineResult``), serialises the
``ModelInferResponse``, and reads it back with the API's own reader
(``src.clients.frame_engine._answer`` through ``tritonclient``'s ``InferResult``). The file
``native/op_engine_client/tests/goldens/answers.json`` holds, per case, the plan text, the response
bytes and what the Python client read: the status, the route, the branch states, and every output
as its datatype, shape and little-endian bytes (a BYTES output as its elements). The Rust parser
must read the same fields out of the same bytes.

``--check`` regenerates into memory and compares with the file. Deleted with the Python client.
"""

from __future__ import annotations

import base64
import json
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'native' / 'op_engine_client' / 'tests' / 'goldens'
SCRIPT = 'scripts/codegen/frame_engine_answer_goldens.py'

DET = {'model': 'det', 'input_size': 640, 'letterbox_fill': 114}
FACES = {
    'detector': 'scrfd', 'input_size': 640, 'recogniser': 'arcface', 'batch': 32,
    'score_min': 0.5, 'nms': 0.4, 'max_faces': 128,
}  # fmt: skip
OCR = {
    'detector': 'ocr_det', 'recogniser': 'ocr_rec', 'batch': 16, 'max_lines': 128, 'det_side': 960,
}  # fmt: skip


def _setup() -> None:
    for path in (ROOT, ROOT / 'scripts' / 'codegen'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    sys.modules.setdefault('cupy', types.ModuleType('cupy'))


def _plan(wire: Any, text: str) -> Any:
    return wire.parse_plan(text, has_boxes=False)


def _ok(plan: Any, wire: Any, **outputs: Any) -> Any:
    from frame_engine.messages import EngineResult

    states: dict[str, Any] = dict.fromkeys(plan.reported(), ('ok', ''))
    return EngineResult(wire.STATUS_OK, 'gpu', states, dict(outputs))


def _cases(wire: Any) -> list[tuple[str, str, bool, Any]]:
    import numpy as np

    from frame_engine.messages import EngineResult

    def plan_text(**body: Any) -> str:
        return json.dumps({'version': 1, **body}, sort_keys=True)

    ingest = plan_text(
        branches=['detect', 'embed_frame', 'embed_crops'], detector=DET, crops='inline',
        embed={'model': 'pe', 'size': 336}, crop_embed={'model': 'pe_crop', 'size': 224, 'batch': 32},
        embed_classes=[1],
    )  # fmt: skip
    ingest_all = plan_text(
        branches=['detect', 'embed_frame', 'embed_crops'], detector=DET, crops='inline',
        embed={'model': 'pe', 'size': 336}, crop_embed={'model': 'pe_crop', 'size': 224, 'batch': 32},
    )  # fmt: skip
    frame = plan_text(
        branches=['detect', 'embed_frame'], detector=DET, crops='none',
        embed={'model': 'pe', 'size': 336},
    )  # fmt: skip
    stored = plan_text(
        branches=['embed_frame', 'embed_crops'], crops='inline',
        embed={'model': 'pe', 'size': 336}, crop_embed={'model': 'pe_crop', 'size': 224, 'batch': 32},
    )  # fmt: skip
    faces = plan_text(branches=['faces', 'face_embed'], faces=FACES)
    text = plan_text(branches=['ocr'], ocr=OCR)
    both = plan_text(branches=['detect', 'faces', 'ocr'], detector=DET, faces=FACES, ocr=OCR,
                     if_detected={'faces': [1], 'ocr': [2]})  # fmt: skip
    renditions = plan_text(
        branches=['detect'], detector=DET,
        derivatives=[{'name': 'thumb', 'side': 256}, {'name': 'view', 'side': 1600, 'quality': 90}],
    )  # fmt: skip

    f32, i32, i64, f64 = np.float32, np.int32, np.int64, np.float64
    boxes3 = np.array([[1, 2, 30, 40], [5, 5, 50, 60], [0, 0, 9, 9]], f32)
    cases: list[tuple[str, str, bool, Any]] = []

    def add(case_id: str, text_: str, *, has_boxes: bool = False, **outputs: Any) -> None:
        plan = wire.parse_plan(text_, has_boxes=has_boxes)
        cases.append((case_id, text_, has_boxes, _ok(plan, wire, **outputs)))

    det3: dict[str, Any] = {
        'size': np.array([640, 480], i32),
        'det_boxes': boxes3, 'det_scores': np.array([0.9, 0.5, 0.25], f32),
        'det_classes': np.array([1, 2, 1], i32), 'letterbox': np.array([1.0, 0.0, 80.0], f64),
        'blur_sums': np.array([[100, 7, 90], [10, 3, 9], [11, 4, 10], [12, 5, 11]], i64),
    }  # fmt: skip
    add(
        'ingest_three_detections_two_embedded', ingest,
        whole_embedding=np.arange(4, dtype=f32) / 7,
        crop_embeddings=np.arange(8, dtype=f32).reshape(2, 4) / 3,
        crop_embedded=np.array([True, False, True]),
        crop_jpegs=[b'\xff\xd8a', b'\xff\xd8bb', b'\xff\xd8ccc'], **det3,
    )  # fmt: skip
    add(
        'ingest_every_box_embedded', ingest_all,
        whole_embedding=np.arange(4, dtype=f32), crop_embeddings=np.ones((3, 4), f32),
        crop_embedded=np.ones(3, np.bool_), crop_jpegs=[b'1', b'2', b'3'], **det3,
    )  # fmt: skip
    add(
        'ingest_no_detection', ingest,
        size=np.array([32, 24], i32), det_boxes=np.zeros((0, 4), f32), det_scores=np.zeros(0, f32),
        det_classes=np.zeros(0, i32), letterbox=np.array([20.0, 0.0, 6.0], f64),
        blur_sums=np.array([[768, 1, 2]], i64), whole_embedding=np.arange(4, dtype=f32),
        crop_embeddings=np.zeros((0, 0), f32), crop_embedded=np.zeros(0, np.bool_), crop_jpegs=[],
    )  # fmt: skip
    add(
        'frame_only', frame, whole_embedding=np.arange(6, dtype=f32) * 0.5,
        **det3,
    )  # fmt: skip
    add(
        'stored_boxes', stored, has_boxes=True, size=np.array([100, 80], i32),
        whole_embedding=np.arange(4, dtype=f32), crop_embeddings=np.ones((2, 4), f32),
        crop_embedded=np.ones(2, np.bool_), crop_jpegs=[b'a', b'b'],
    )  # fmt: skip
    add(
        'faces_two_one_embedded', faces, size=np.array([96, 64], i32),
        face_boxes=np.array([[8, 8, 40, 40], [48, 16, 88, 56]], f32),
        face_scores=np.array([0.99, 0.6], f32), face_landmarks=np.arange(20, dtype=f32).reshape(2, 10),
        face_blur_sums=np.array([[1024, 5, 50], [1600, 6, 60]], i64),
        face_overflow=np.array([3], i32), face_embeddings=np.ones((1, 8), f32) * 0.25,
        face_embedded=np.array([True, False]),
    )  # fmt: skip
    add(
        'faces_none', faces, size=np.array([96, 64], i32), face_boxes=np.zeros((0, 4), f32),
        face_scores=np.zeros(0, f32), face_landmarks=np.zeros((0, 10), f32),
        face_blur_sums=np.zeros((0, 3), i64), face_overflow=np.array([0], i32),
        face_embeddings=np.zeros((0, 0), f32), face_embedded=np.zeros(0, np.bool_),
    )  # fmt: skip
    add(
        'text_two_lines', text, size=np.array([200, 100], i32),
        ocr_quads=np.arange(16, dtype=f32).reshape(2, 8), ocr_texts=[b'STOP', 'café 中'.encode()],
        ocr_scores=np.array([0.9, 0.8], f32), ocr_char_min=np.array([0.7, 0.6], f32),
        ocr_det_scores=np.array([0.5, 0.4], f32), ocr_overflow=np.array([2], i32),
    )  # fmt: skip
    add(
        'renditions_after_detection', renditions,
        derivative_jpegs=[b'\xff\xd8thumb', b'\xff\xd8view'], **det3,
    )  # fmt: skip
    plan = wire.parse_plan(both, has_boxes=False)
    states: dict[str, Any] = dict.fromkeys(plan.reported(), ('ok', ''))
    states['faces'] = ('not_run', '')
    states['ocr'] = ('error', wire.MODEL_ERROR)
    cases.append((
        'detect_then_conditional_branches_not_run', both, False,
        EngineResult(wire.STATUS_OK, 'cpu_progressive', states, dict(det3)),
    ))  # fmt: skip
    plan = wire.parse_plan(ingest, has_boxes=False)
    states = dict.fromkeys(plan.reported(), ('ok', ''))
    states['embed_crops'] = ('error', wire.MODEL_ERROR)
    cases.append((
        'crop_encoder_failed', ingest, False,
        EngineResult(wire.STATUS_OK, 'gpu', states, {
            'size': np.array([640, 480], i32), 'whole_embedding': np.arange(4, dtype=f32), **{
                k: v for k, v in det3.items() if k != 'size'},
            'crop_jpegs': [b'1', b'2', b'3'],
        }),
    ))  # fmt: skip
    for status in ('corrupt', 'huge', 'over_budget', 'gpu_memory', 'decoder'):
        plan = wire.parse_plan(ingest, has_boxes=False)
        states = dict.fromkeys(plan.reported(), ('error', wire.IMAGE_FAILED))
        cases.append((f'not_decoded_{status}', ingest, False,
                      EngineResult(status, wire.DECODED_NONE, states)))  # fmt: skip
    return cases


def summarise(answer: Any, wire: Any) -> dict[str, Any]:
    """What the Python client read (an ``EngineAnswer``) as the fields the Rust parser is held to:
    the status, the route, the branch states, and each output's datatype, shape and bytes."""
    outputs: dict[str, Any] = {}
    for name, value in answer.outputs.items():
        if wire.OUTPUT_DATATYPES[name] == 'BYTES':
            items = [bytes(v) for v in (value if isinstance(value, list) else value.reshape(-1))]
            outputs[name] = {'items_hex': [i.hex() for i in items]}
        else:
            outputs[name] = {
                'datatype': wire.OUTPUT_DATATYPES[name],
                'shape': list(value.shape),
                'hex': value.tobytes().hex(),
            }
    return {
        'status': answer.status,
        'decode_route': answer.decode_route,
        'branches': {k: list(v) for k, v in sorted(answer.branches.items())},
        'ok': answer.ok,
        'outputs': outputs,
    }


def _expected(client: Any, wire: Any, plan: Any, response: Any) -> dict[str, Any]:
    from tritonclient.grpc import InferResult

    return summarise(client._answer(InferResult(response), plan), wire)


def render() -> dict[str, str]:
    _setup()
    from tritonclient.grpc import service_pb2

    from frame_engine.front import build_response
    from src.clients import frame_engine as client
    from src.domain import frame_engine_wire as wire

    out = []
    for case_id, text, has_boxes, result in _cases(wire):
        plan = wire.parse_plan(text, has_boxes=has_boxes)
        response = build_response(service_pb2.ModelInferRequest(id='7'), result)
        out.append(
            {
                'id': case_id,
                'plan': text,
                'response_b64': base64.b64encode(response.SerializeToString()).decode(),
                'expected': _expected(client, wire, plan, response),
            }
        )
    body = {
        'generated_by': SCRIPT,
        'sources': ['frame_engine/front.py', 'src/clients/frame_engine.py'],
        'cases': out,
    }
    return {'answers.json': json.dumps(body, sort_keys=True, separators=(',', ':')) + '\n'}


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import frame_engine_goldens

    return int(frame_engine_goldens.main(argv, render, SCRIPT, out=OUT))


if __name__ == '__main__':
    raise SystemExit(main())
