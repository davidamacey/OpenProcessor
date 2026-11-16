#!/usr/bin/env python3
"""The plan texts the Python client writes, for ``op_engine_client``'s plan builder (plan 18a, R1e).

Writes ``contracts/json/frame_engine_plan_build_cases.json``. Every text in it is what
``src/domain/frame_engine_wire.py`` ``EnginePlan.to_json`` produced for the plan described next to
it, so the Rust builder is held to the Python one byte for byte (``json.dumps(sort_keys=True)``:
sorted keys, ``", "`` and ``": "`` separators, ASCII only, floats as ``repr`` writes them):

* ``parsed``: every plan text of ``frame_engine_plan_cases.py`` that ``parse_plan`` accepts, with
  the text ``to_json`` writes for the plan parsed from it. The Rust side reads the same input
  text into its plan type and writes it again. This covers every branch and field of the
  vocabulary: faces, text, renditions, ``if_detected``, class filters, stored boxes, defaults;
* ``ingest``: the plan ``ingest_gpu_decode.ingest_engine_plan`` builds for each detector, policy
  and encoder setting here, with the detector, the encoders and the crop-vector choice that made
  it (the class ids come from the policy's class names and the detector's labels; the Rust
  builder is given the ids);
* ``stored_boxes``: the plan ``reprocess_gpu_embed.embed_on_gpu`` sends for stored boxes;
* ``strings``: detector model names with quotes, control characters, DEL, non-ASCII and
  characters outside the basic plane, and the text ``to_json`` writes with each;
* ``floats``: doubles (as IEEE bit patterns) and ``repr`` of each.

``--check`` regenerates into memory and compares with the file. Deleted with the Python client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contracts' / 'json' / 'frame_engine_plan_build_cases.json'
SCRIPT = 'scripts/codegen/frame_engine_plan_build_cases.py'
SOURCES = [
    'src/domain/frame_engine_wire.py',
    'src/services/curation/ingest_gpu_decode.py',
    'src/services/curation/reprocess_gpu_embed.py',
    'scripts/codegen/frame_engine_plan_cases.py',
]
#: ``(image model, crop model, crop size)`` the encoder settings take.
ENCODERS = [
    ('pe_image_encoder', 'pe_crop_encoder', 224),
    ('image_enc', 'crop_enc', 336),
]
LABELS = ['gadget', 'widget', 'gizmo', 'sprocket']
STRINGS = [
    'plain',
    'with "quotes" and \\ backslash',
    'tab\tnewline\ncarriage\rbell\x07form\x0cback\x08',
    'control \x01\x1f and del \x7f',
    'café über 中文',
    'astral \U0001f600 \U0001d11e',
    'slash / and separators \u2028 \u2029 \u00a0',
    '',
]


def _bits(x: float) -> int:
    return int(struct.unpack('<Q', struct.pack('<d', x))[0])


def _floats() -> list[dict[str, Any]]:
    rng = random.Random(20261116)
    values = [
        0.0, -0.0, 1.0, -1.0, 0.1, 0.5, 0.45, 0.4, 1.4, 0.2, 10.0, 0.30000000000000004,
        1e-4, 9.999e-5, 1e-5, 1.5e-7, 2.5e-5, 5e-324, 1.7976931348623157e308,
        123456789012345.0, 1e15, 9999999999999998.0, 1e16, 1.2345e17, 1e22, 1e23,
        0.0001, 0.00012345, 1234567.0, 100.0, 4503599627370496.0, 123456789.123456789,
    ]  # fmt: skip
    for _ in range(400):
        mantissa = rng.uniform(1, 10)
        values.append(rng.choice((-1, 1)) * mantissa * 10.0 ** rng.randint(-30, 30))
    # exact ties between two shortest candidates: Python rounds them to the even digit
    values.append(-704758667114124.25)
    values.extend(
        (2**49 + rng.randrange(2**48)) + rng.choice((0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875))
        for _ in range(60)
    )
    values.extend(
        struct.unpack('<d', struct.pack('<Q', rng.getrandbits(64)))[0] for _ in range(100)
    )
    out = []
    for v in values:
        if not math.isfinite(v):
            continue
        out.append({'bits': f'{_bits(v):016x}', 'repr': repr(v)})
    return out


def _parsed(wire: Any) -> list[dict[str, Any]]:
    from frame_engine_plan_cases import plan_cases

    out = []
    for case_id, text, has_boxes in plan_cases():
        try:
            plan = wire.parse_plan(text, has_boxes=has_boxes)
        except wire.EnginePlanError:
            continue
        out.append({'id': case_id, 'input': text, 'has_boxes': has_boxes, 'text': plan.to_json()})
    return out


def _strings(wire: Any) -> list[dict[str, str]]:
    out = []
    for s in STRINGS:
        plan = wire.EnginePlan(
            branches=('detect',),
            detector=wire.DetectorSpec(model=s, input_size=640, letterbox_fill=114),
        )
        out.append({'model': s, 'text': plan.to_json()})
    return out


def _ingest(ingest_gpu_decode: Any) -> list[dict[str, Any]]:
    from src.config import DetectionProfile
    from src.services.curation.ingest_policy import DetectFilter, EmbeddingPolicy, IngestPolicy

    policies: list[tuple[str, IngestPolicy]] = [
        ('default', IngestPolicy()),
        ('lazy', IngestPolicy(embedding=EmbeddingPolicy(mode='lazy'))),
        ('none', IngestPolicy(embedding=EmbeddingPolicy(mode='none'))),
        ('all_with_classes_ignored',
         IngestPolicy(embedding=EmbeddingPolicy(mode='all', classes=['widget']))),
        ('selected', IngestPolicy(embedding=EmbeddingPolicy(mode='selected', classes=['Widget']))),
        ('detect_classes', IngestPolicy(detect=DetectFilter(classes=['gadget', 'widget']))),
        ('selected_and_detect', IngestPolicy(
            detect=DetectFilter(classes=['gadget', 'widget']),
            embedding=EmbeddingPolicy(mode='selected', classes=['widget', 'gizmo']))),
        ('excluded', IngestPolicy(detect=DetectFilter(exclude_classes=['gadget']))),
        ('nothing_can_be_embedded', IngestPolicy(
            embedding=EmbeddingPolicy(mode='selected', classes=['nonesuch']))),
    ]  # fmt: skip
    detectors = [('det', 640, 114), ('yolo_b', 1280, 0)]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        labels = Path(tmp) / 'labels.txt'
        labels.write_text('\n'.join(LABELS) + '\n')
        os.environ['OP_TRITON_MODEL_REPO'] = tmp
        for image_model, crop_model, crop_size in ENCODERS:
            ingest_gpu_decode.PE_IMAGE_MODEL = image_model
            ingest_gpu_decode.PE_CROP_MODEL = crop_model
            ingest_gpu_decode.PE_CROP_SIZE = crop_size
            for model, size, fill in detectors:
                profile = DetectionProfile(
                    name='cases',
                    detector_model=model,
                    input_size=size,
                    letterbox_fill=(fill, fill, fill),
                    labels_path=str(labels),
                )
                for name, policy in policies:
                    service = SimpleNamespace(profile=profile, policy=policy)
                    plan = ingest_gpu_decode.ingest_engine_plan(service)
                    crops = 'deferred'
                    classes: list[int] | None = None
                    if 'embed_crops' in plan.branches:
                        crops = 'every' if plan.embed_classes is None else 'classes'
                        classes = None if plan.embed_classes is None else sorted(plan.embed_classes)
                    out.append(
                        {
                            'id': f'{model}_{size}_{crop_model}_{crop_size}_{name}',
                            'detector': {
                                'model': model,
                                'input_size': size,
                                'letterbox_fill': fill,
                            },
                            'encoders': {
                                'image_model': image_model,
                                'crop_model': crop_model,
                                'crop_size': crop_size,
                            },
                            'crop_vectors': {'kind': crops, 'classes': classes},
                            'text': plan.to_json(),
                        }
                    )
    return out


def _stored_boxes(reprocess: Any, ingest_gpu_decode: Any) -> list[dict[str, Any]]:
    """The plan ``embed_on_gpu`` hands to the engine call, captured from the real function."""
    captured: list[Any] = []

    async def capture(
        _pool: Any, _data: bytes, plan: Any, _boxes: Any = None, **_named: Any
    ) -> str:
        captured.append(plan)
        return 'corrupt'  # the function then takes the CPU path and returns None

    original = reprocess.call_frame_engine
    reprocess.call_frame_engine = capture
    out = []
    try:
        for image_model, crop_model, crop_size in ENCODERS:
            ingest_gpu_decode.PE_IMAGE_MODEL = image_model
            ingest_gpu_decode.PE_CROP_MODEL = crop_model
            ingest_gpu_decode.PE_CROP_SIZE = crop_size
            asyncio.run(
                reprocess.embed_on_gpu(reprocess.GpuEmbedRoute(None), b'', [(0.1, 0.1, 0.5, 0.6)])
            )
            out.append(
                {
                    'encoders': {
                        'image_model': image_model,
                        'crop_model': crop_model,
                        'crop_size': crop_size,
                    },
                    'text': captured[-1].to_json(),
                }
            )
    finally:
        reprocess.call_frame_engine = original
    return out


def render() -> str:
    for path in (ROOT, ROOT / 'scripts' / 'codegen'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from src.domain import frame_engine_wire as wire
    from src.services.curation import ingest_gpu_decode, reprocess_gpu_embed

    saved = (
        ingest_gpu_decode.PE_IMAGE_MODEL,
        ingest_gpu_decode.PE_CROP_MODEL,
        ingest_gpu_decode.PE_CROP_SIZE,
    )
    try:
        body = {
            'generated_by': SCRIPT,
            'sources': SOURCES,
            'parsed': _parsed(wire),
            'ingest': _ingest(ingest_gpu_decode),
            'stored_boxes': _stored_boxes(reprocess_gpu_embed, ingest_gpu_decode),
            'strings': _strings(wire),
            'floats': _floats(),
        }
    finally:
        (
            ingest_gpu_decode.PE_IMAGE_MODEL,
            ingest_gpu_decode.PE_CROP_MODEL,
            ingest_gpu_decode.PE_CROP_SIZE,
        ) = saved
    return json.dumps(body, indent=2, sort_keys=True) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='compare with the committed file')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding='utf-8') != text:
            sys.stderr.write(f'{args.out} is stale: run {SCRIPT}\n')
            return 1
        return 0
    args.out.write_text(text, encoding='utf-8')
    sys.stdout.write(f'wrote {args.out} ({len(text)} bytes)\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
