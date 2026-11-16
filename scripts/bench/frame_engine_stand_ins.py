#!/usr/bin/env python3
"""The stand-in models of the native engine's N0 comparison (plan 17 section 5, N0 "Stand-in models").

Two kinds, loaded in the model server beside the real models from a second repository
(``/stand_in``), so both engines reach them through their normal transport:

* **null models** (``null_detector_640``, ``null_encoder_336``, ``null_encoder_224``): the real
  model's config under another name, a TensorRT plan from a tiny ONNX graph that casts one input
  pixel, multiplies it by zero and adds it to a constant. They cost the card nearly nothing and
  answer a constant: the detector five fixed rows (centred boxes of 16, 48, 160, 320 and 600 px in
  the 640 canvas, score 0.9, classes 0, 2, 5, 7, 16: crops both smaller and larger than the 224
  square), the encoders one unit vector. They measure the engine, not the models (bar G2).
* **capture models** (``capture_detector_640``, ``capture_encoder_336``, ``capture_encoder_224``):
  Python-backend models with the same names of tensors and the same answers, that write every input
  row they receive to disk (``frame_engine_capture_model.py``), so the bytes both engines hand the
  models can be compared (``gpu_prep_parity.py compare-inputs``).

``generate`` writes the configs, the capture models' code and, with ``--onnx``, the graphs. The
TensorRT plans are built from those graphs by ``trtexec`` in the model server's image (the way
``export/build_pe_trt.sh`` builds a plan: ``docker compose run ... triton-server trtexec``);
``plan-commands`` prints the commands and runs them only with ``--run``. A lane runs no containers.

    python scripts/bench/frame_engine_stand_ins.py generate --out artifacts_local/native_engine/n0/stand_in --onnx
    python scripts/bench/frame_engine_stand_ins.py plan-commands --out artifacts_local/native_engine/n0/stand_in
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from collections.abc import Sequence

    from onnx import ModelProto

CAPTURE_MODEL_SOURCE = Path(__file__).with_name('frame_engine_capture_model.py')
#: Where the compose file mounts the repository and the capture directory in the model server.
SERVER_STAND_IN_DIR = '/stand_in'
SERVER_CAPTURE_DIR = '/capture'

DETECTOR_ROWS_MAX = 300
VECTOR_DIM = 1024
#: Centred boxes of these sides in the 640 canvas, with these classes, score 0.9.
BOX_SIDES = (16, 48, 160, 320, 600)
BOX_CLASSES = (0, 2, 5, 7, 16)
BOX_SCORE = 0.9
CANVAS = 640

#: The trtexec build profile of a model: the real model's own batch range (its config's maximum;
#: the PE builds use opt 8, ``export/build_pe_trt.sh``).
DETECTOR_PROFILE = (1, 16, 64)
ENCODER_PROFILE = (1, 8, 32)
TRTEXEC_WORKSPACE = '1G'


def detector_rows() -> list[list[float]]:
    """The five fixed detections ``[x1, y1, x2, y2, score, class]`` in letterboxed canvas pixels."""
    half = CANVAS / 2
    return [
        [half - s / 2, half - s / 2, half + s / 2, half + s / 2, BOX_SCORE, float(c)]
        for s, c in zip(BOX_SIDES, BOX_CLASSES, strict=True)
    ]


@dataclass(frozen=True)
class Spec:
    """One stand-in pair's shape: what the real model takes and returns."""

    kind: Literal['detector', 'encoder']
    null_name: str
    capture_name: str
    side: int
    output_name: str
    output_dims: tuple[int, ...]
    max_batch: int
    preferred: tuple[int, ...]
    delay_us: int
    #: the real model this one stands for, whose config the null model repeats
    real: str
    default_instances: int
    profile: tuple[int, int, int]


SPECS: tuple[Spec, ...] = (
    Spec('detector', 'null_detector_640', 'capture_detector_640', 640, 'output0',
         (DETECTOR_ROWS_MAX, 6), 64, (8, 16, 32, 64), 25_000, 'yolov11_small_trt_end2end', 2,
         DETECTOR_PROFILE),
    Spec('encoder', 'null_encoder_336', 'capture_encoder_336', 336, 'image_embeddings',
         (VECTOR_DIM,), 32, (4, 8, 16, 32), 15_000, 'pe_image_encoder', 1, ENCODER_PROFILE),
    Spec('encoder', 'null_encoder_224', 'capture_encoder_224', 224, 'image_embeddings',
         (VECTOR_DIM,), 32, (4, 8, 16, 32), 15_000, 'pe_crop_encoder', 1, ENCODER_PROFILE),
)  # fmt: skip


def model_names(kind: Literal['null', 'capture']) -> dict[str, str]:
    """The stand-ins' names by the role they play in a plan: the detector, the frame encoder
    (336) and the crop encoder (224)."""
    pick = (lambda sp: sp.null_name) if kind == 'null' else (lambda sp: sp.capture_name)
    detector, frame, crop = (pick(sp) for sp in SPECS)
    return {'detector': detector, 'frame': frame, 'crop': crop}


def null_names() -> list[str]:
    return [s.null_name for s in SPECS]


def capture_names() -> list[str]:
    return [s.capture_name for s in SPECS]


def _dims(dims: Sequence[int]) -> str:
    return ', '.join(str(d) for d in dims)


def _batching(s: Spec) -> str:
    sizes = ', '.join(str(b) for b in s.preferred)
    return (
        'dynamic_batching {\n'
        f'  preferred_batch_size: [ {sizes} ]\n'
        f'  max_queue_delay_microseconds: {s.delay_us}\n'
        '}\n'
    )


def null_config(s: Spec, instances: int | None = None) -> str:
    """``config.pbtxt`` of the null model: the plan's text for the detector, the real encoder's
    config under another name for the encoders. The detector stays on ``gpus: [ 0 ]`` (the model
    server sees one card)."""
    count = instances if instances is not None else s.default_instances
    gpus = '    gpus: [ 0 ]\n' if s.kind == 'detector' else ''
    return (
        f'name: "{s.null_name}"\n'
        'platform: "tensorrt_plan"\n'
        f'max_batch_size: {s.max_batch}\n'
        f'input [ {{ name: "images" data_type: TYPE_UINT8 dims: [ 3, {s.side}, {s.side} ] }} ]\n'
        f'output [ {{ name: "{s.output_name}" data_type: TYPE_FP32 dims: [ {_dims(s.output_dims)} ] }} ]\n'
        + _batching(s)
        + f'instance_group [\n  {{\n    count: {count}\n    kind: KIND_GPU\n{gpus}  }}\n]\n'
    )


def _parameter(key: str, value: str) -> str:
    return f'parameters {{ key: "{key}" value: {{ string_value: {json.dumps(value)} }} }}\n'


def capture_config(s: Spec) -> str:
    """``config.pbtxt`` of the capture model: the null model's tensors on the Python backend, one
    CPU instance (so the arrival order of rows is one order), inputs forced to host memory."""
    params = [
        ('FORCE_CPU_ONLY_INPUT_TENSORS', 'yes'),
        ('CAPTURE_DIR', SERVER_CAPTURE_DIR),
        ('OUTPUT_NAME', s.output_name),
        ('OUTPUT_KIND', s.kind),
    ]
    if s.kind == 'detector':
        params += [
            ('OUTPUT_ROWS', json.dumps(detector_rows())),
            ('OUTPUT_ROW_COUNT', str(DETECTOR_ROWS_MAX)),
        ]
    else:
        params.append(('VECTOR_DIM', str(VECTOR_DIM)))
    return (
        f'name: "{s.capture_name}"\n'
        'backend: "python"\n'
        f'max_batch_size: {s.max_batch}\n'
        f'input [ {{ name: "images" data_type: TYPE_UINT8 dims: [ 3, {s.side}, {s.side} ] }} ]\n'
        f'output [ {{ name: "{s.output_name}" data_type: TYPE_FP32 dims: [ {_dims(s.output_dims)} ] }} ]\n'
        + _batching(s)
        + 'instance_group [ { count: 1 kind: KIND_CPU } ]\n'
        + ''.join(_parameter(k, v) for k, v in params)
    )


def null_onnx(s: Spec) -> ModelProto:
    """The null model's graph: ``images`` (UINT8, ``[B, 3, S, S]``) -> FP32 -> its first pixel ->
    times zero -> added to the constant answer. TensorRT keeps the input bound because the output
    depends on it; the answer does not change with it (zero times a finite number). The cast comes
    first because TensorRT's Slice takes no UINT8 (operators page for Slice: bool, int4, int8,
    int32, int64, float8, float16, float32, bfloat16); Cast takes it."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if s.kind == 'detector':
        answer = np.zeros(s.output_dims, np.float32)
        answer[: len(BOX_SIDES)] = np.array(detector_rows(), np.float32)
    else:
        answer = np.full(s.output_dims, 1.0 / np.sqrt(VECTOR_DIM), np.float32)

    def ints(name: str, values: list[int]) -> TensorProto:
        return numpy_helper.from_array(np.array(values, np.int64), name)

    inits = [
        ints('starts', [0, 0, 0]),
        ints('ends', [1, 1, 1]),
        ints('axes', [1, 2, 3]),
        ints('batch_shape', [-1, *([1] * len(s.output_dims))]),
        numpy_helper.from_array(np.zeros((), np.float32), 'zero'),
        numpy_helper.from_array(answer, 'answer'),
    ]
    nodes = [
        helper.make_node('Cast', ['images'], ['images_f'], to=TensorProto.FLOAT),
        helper.make_node('Slice', ['images_f', 'starts', 'ends', 'axes'], ['pixel_f']),
        helper.make_node('Mul', ['pixel_f', 'zero'], ['nothing']),
        helper.make_node('Reshape', ['nothing', 'batch_shape'], ['nothing_b']),
        helper.make_node('Add', ['nothing_b', 'answer'], [s.output_name]),
    ]
    graph = helper.make_graph(
        nodes,
        s.null_name,
        [helper.make_tensor_value_info('images', TensorProto.UINT8, ['B', 3, s.side, s.side])],
        [helper.make_tensor_value_info(s.output_name, TensorProto.FLOAT, ['B', *s.output_dims])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def trtexec_args(s: Spec, repo: str = SERVER_STAND_IN_DIR) -> list[str]:
    """The ``trtexec`` arguments that build ``<repo>/<null>/1/model.plan`` from the ONNX graph at
    the real model's batch profile (no precision flag: TensorRT 11 builds are strongly typed;
    ``--skipInference``: build only)."""
    lo, opt, hi = s.profile
    shape = f'x3x{s.side}x{s.side}'
    return [
        f'--onnx={repo}/_onnx/{s.null_name}.onnx',
        f'--saveEngine={repo}/{s.null_name}/1/model.plan',
        f'--minShapes=images:{lo}{shape}',
        f'--optShapes=images:{opt}{shape}',
        f'--maxShapes=images:{hi}{shape}',
        f'--memPoolSize=workspace:{TRTEXEC_WORKSPACE}',
        '--skipInference',
    ]


#: How ``export/build_pe_trt.sh`` reaches trtexec: the model server's image carries it. The test
#: bed's compose file mounts the stand-in repository at ``/stand_in`` read-only (serving only
#: reads it). A ``compose run -v`` on that same target stays read-only (measured 2026-11-15: the
#: file's mount wins), so the build run mounts the host directory a second time, writable, at
#: ``BUILD_STAND_IN_DIR`` and reads and writes through that. ``{stand_in}`` is the host directory.
BUILD_STAND_IN_DIR = '/stand_in_build'
DEFAULT_RUNNER = (
    'docker compose -p op-n0 -f docker/frame-engine-native/compose.n0.yml '
    'run --rm --no-deps -T -v {stand_in}:' + BUILD_STAND_IN_DIR + ' triton-server trtexec'
)
DEFAULT_OUT = Path('artifacts_local/native_engine/n0/stand_in')


def plan_commands(out: Path = DEFAULT_OUT, runner: str = DEFAULT_RUNNER) -> list[list[str]]:
    prefix = shlex.split(runner.replace('{stand_in}', shlex.quote(str(out.resolve()))))
    return [[*prefix, *trtexec_args(s, BUILD_STAND_IN_DIR)] for s in SPECS]


def generate(
    out: Path, *, onnx: bool, detector_instances: int, encoder_instances: int
) -> list[str]:
    """Write the six models' configs, the capture models' code and, with ``onnx``, the graphs of
    the null models. Returns what was written."""
    written: list[str] = []

    def put(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written.append(str(path))

    for s in SPECS:
        count = detector_instances if s.kind == 'detector' else encoder_instances
        put(out / s.null_name / 'config.pbtxt', null_config(s, count))
        (out / s.null_name / '1').mkdir(parents=True, exist_ok=True)
        put(out / s.capture_name / 'config.pbtxt', capture_config(s))
        put(out / s.capture_name / '1' / 'model.py', CAPTURE_MODEL_SOURCE.read_text())
        if onnx:
            graph = out / '_onnx' / f'{s.null_name}.onnx'
            graph.parent.mkdir(parents=True, exist_ok=True)
            import onnx as onnx_lib

            onnx_lib.save(null_onnx(s), str(graph))
            written.append(str(graph))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    gen = sub.add_parser('generate', help='write the configs, capture code and (--onnx) graphs')
    gen.add_argument('--out', type=Path, required=True, help='the stand-in repository directory')
    gen.add_argument('--onnx', action='store_true', help="also write the null models' ONNX graphs")
    gen.add_argument('--detector-instances', type=int, default=2)
    gen.add_argument('--encoder-instances', type=int, default=1)
    cmds = sub.add_parser('plan-commands', help='the trtexec commands that build the null plans')
    cmds.add_argument('--out', type=Path, default=DEFAULT_OUT, help='the stand-in repository')
    cmds.add_argument('--runner', default=DEFAULT_RUNNER, help='what runs trtexec')
    cmds.add_argument('--run', action='store_true', help="run them (the coordinator's step)")
    sub.add_parser(
        'detections', help='how many detections (crops) a stand-in detector gives a photo'
    )
    names = sub.add_parser('names', help='the model names as JSON')
    names.add_argument('--kind', choices=['null', 'capture'], required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'generate':
        for path in generate(
            args.out,
            onnx=args.onnx,
            detector_instances=args.detector_instances,
            encoder_instances=args.encoder_instances,
        ):
            print(path)
        return 0
    if args.cmd == 'detections':
        print(len(BOX_SIDES))
        return 0
    if args.cmd == 'names':
        print(json.dumps(null_names() if args.kind == 'null' else capture_names()))
        return 0
    for command in plan_commands(args.out, args.runner):
        print(shlex.join(command))
        if args.run:
            if shutil.which(command[0]) is None:
                print(f'{command[0]}: not found', file=sys.stderr)
                return 1
            # argv list built from this file's own constants and the --runner argument, no shell
            if subprocess.run(command, check=False).returncode != 0:  # nosec B603
                return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
