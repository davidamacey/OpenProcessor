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
* **product nulls** (``null_scrfd_640``, ``null_arcface_112``, ``null_ocr_det``,
  ``null_ocr_rec``): the full product plan's face and text models, the real config under another
  name and a plan that answers a fixed result (one 128 px face per detector call, one unit face
  vector, four text lines per text map, one character per line), so a capacity row of the full
  plan measures the engine's face and text work, not the models (plan 17 N2).

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
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
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


#: The real models' configs, which the product nulls repeat under their own names.
REAL_MODELS_DIR = Path(__file__).resolve().parents[2] / 'models'

#: SCRFD's strides and anchors per grid cell (``frame_engine/faces/scrfd.py``).
SCRFD_STRIDES = (8, 16, 32)
SCRFD_ANCHORS = 2
SCRFD_NET = 640
#: The one face every null face-detector call answers: anchor 0 of the stride-32 cell at row 10,
#: column 10 (centre (320, 320) of the 640 input), score 0.9, box 2 strides from the centre on each
#: side (128 px), landmarks (eyes, nose, mouth corners) as stride multiples from the centre.
SCRFD_FACE_CELL = (10, 10)
SCRFD_FACE_SCORE = 0.9
SCRFD_FACE_DISTANCE = 2.0
SCRFD_FACE_LANDMARKS = ((-0.5, -0.4), (0.5, -0.4), (0.0, 0.0), (-0.4, 0.5), (0.4, 0.5))
FACE_VECTOR_DIM = 512
FACE_SIDE = 112
#: The null text detector's probability map: 1.0 in these row bands times this column band of the
#: detector input (input pixels), 0.0 elsewhere: four lines of 32 x 384 px whatever the photo.
TEXT_ROW_BANDS = ((64, 96), (160, 192), (256, 288), (352, 384))
TEXT_COL_BAND = (64, 448)
#: The null recogniser's answer: one step per this many input columns, every step this class
#: (not the CTC blank 0) at this probability: each line reads as one character.
TEXT_REC_STEP = 8
TEXT_REC_CLASS = 5
TEXT_REC_SCORE = 0.9


@dataclass(frozen=True)
class ProductNull:
    """A null stand-in for one of the full product plan's other models (faces and text): the real
    model's config under another name and a TensorRT plan that answers a fixed result for any
    input, so the engine's face and text work runs while the card does nearly nothing."""

    kind: Literal['scrfd', 'face_vector', 'text_map', 'text_lines']
    null_name: str
    #: the real model whose config (tensors, batching, instances) the null model repeats
    real: str
    input_name: str
    #: trtexec ``--minShapes`` / ``--optShapes`` / ``--maxShapes`` of the input, ``BxCxHxW``: the
    #: real model's build range (``export/export_scrfd.py``, ``export_face_recognition.py``,
    #: ``export_paddleocr_det.py`` uint8 build, ``export_paddleocr_rec.py``)
    shapes: tuple[str, str, str]


PRODUCT_NULLS: tuple[ProductNull, ...] = (
    ProductNull('scrfd', 'null_scrfd_640', 'scrfd_10g_bnkps_uint8', 'images',
                ('1x3x640x640', '16x3x640x640', '64x3x640x640')),
    ProductNull('face_vector', 'null_arcface_112', 'arcface_w600k_r50_uint8', 'images',
                ('1x3x112x112', '32x3x112x112', '128x3x112x112')),
    ProductNull('text_map', 'null_ocr_det', 'paddleocr_det_trt_uint8', 'x',
                ('1x3x32x32', '1x3x736x736', '1x3x1920x1920')),
    ProductNull('text_lines', 'null_ocr_rec', 'paddleocr_rec_trt_uint8', 'x',
                ('1x3x48x48', '8x3x48x320', '16x3x48x2048')),
)  # fmt: skip


def product_null_names() -> dict[str, str]:
    """The product nulls' names by the plan field they fill (``faces.detector``,
    ``faces.recogniser``, ``ocr.detector``, ``ocr.recogniser``)."""
    by_kind = {p.kind: p.null_name for p in PRODUCT_NULLS}
    return {
        'face_detector': by_kind['scrfd'],
        'face_recogniser': by_kind['face_vector'],
        'text_detector': by_kind['text_map'],
        'text_recogniser': by_kind['text_lines'],
    }


def product_null_config(p: ProductNull, models_dir: Path = REAL_MODELS_DIR) -> str:
    """The real model's ``config.pbtxt`` with the null model's name: the same tensors, batching
    and instances, so the engine calls it exactly as it calls the real one."""
    text = (models_dir / p.real / 'config.pbtxt').read_text()
    renamed, count = re.subn(
        r'^name:\s*"[^"]*"', f'name: "{p.null_name}"', text, count=1, flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError(f'{p.real}: no name line in its config')
    return renamed


def scrfd_answer() -> dict[str, np.ndarray]:
    """The null face detector's nine outputs for one input (``[anchors, d]`` each): every anchor
    scores 0 but :data:`SCRFD_FACE_CELL`'s first, which carries the fixed face."""
    import numpy as np

    out: dict[str, np.ndarray] = {}
    for stride in SCRFD_STRIDES:
        cells = SCRFD_NET // stride
        anchors = cells * cells * SCRFD_ANCHORS
        score = np.zeros((anchors, 1), np.float32)
        bbox = np.zeros((anchors, 4), np.float32)
        kps = np.zeros((anchors, 10), np.float32)
        if stride == SCRFD_STRIDES[-1]:
            row, col = SCRFD_FACE_CELL
            k = (row * cells + col) * SCRFD_ANCHORS
            score[k] = SCRFD_FACE_SCORE
            bbox[k] = SCRFD_FACE_DISTANCE
            kps[k] = np.array(SCRFD_FACE_LANDMARKS, np.float32).reshape(10)
        out[f'score_{stride}'], out[f'bbox_{stride}'], out[f'kps_{stride}'] = score, bbox, kps
    return out


def scrfd_output_names() -> list[str]:
    """The real config's output order: every score, every box, every landmark set."""
    return [f'{part}_{s}' for part in ('score', 'bbox', 'kps') for s in SCRFD_STRIDES]


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


def _constant_answers(
    input_name: str, side: int, answers: dict[str, np.ndarray], graph_name: str
) -> ModelProto:
    """A graph whose outputs are fixed arrays (all of one rank), one per input row, bound to the
    ``[B, 3, side, side]`` input the way :func:`null_onnx` binds it (cast, first pixel, times
    zero, added to each answer)."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    def ints(name: str, values: list[int]) -> TensorProto:
        return numpy_helper.from_array(np.array(values, np.int64), name)

    inits = [
        ints('starts', [0, 0, 0]),
        ints('ends', [1, 1, 1]),
        ints('axes', [1, 2, 3]),
        ints('batch_shape', [-1, *([1] * {a.ndim for a in answers.values()}.pop())]),
        numpy_helper.from_array(np.zeros((), np.float32), 'zero'),
    ]
    nodes = [
        helper.make_node('Cast', [input_name], ['in_f'], to=TensorProto.FLOAT),
        helper.make_node('Slice', ['in_f', 'starts', 'ends', 'axes'], ['pixel_f']),
        helper.make_node('Mul', ['pixel_f', 'zero'], ['nothing']),
        helper.make_node('Reshape', ['nothing', 'batch_shape'], ['nothing_b']),
    ]
    outputs = []
    for name, answer in answers.items():
        inits.append(numpy_helper.from_array(answer, f'{name}_answer'))
        nodes.append(helper.make_node('Add', ['nothing_b', f'{name}_answer'], [name]))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ['B', *answer.shape]))
    graph = helper.make_graph(
        nodes,
        graph_name,
        [helper.make_tensor_value_info(input_name, TensorProto.UINT8, ['B', 3, side, side])],
        outputs,
        inits,
    )
    return _checked(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)]))


def _checked(model: ModelProto) -> ModelProto:
    import onnx

    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def _text_map_onnx(p: ProductNull) -> ModelProto:
    """``x`` (UINT8 ``[B, 3, H, W]``) -> ``fetch_name_0`` (FP32 ``[B, 1, H, W]``): 1.0 where a
    row of :data:`TEXT_ROW_BANDS` meets :data:`TEXT_COL_BAND`, else 0.0, for any H and W. The
    row and column indices come from ``Range`` over the input's own shape (TensorRT's fill
    layer), so the map follows the size the engine sent."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    def f32(name: str, v: float) -> TensorProto:
        return numpy_helper.from_array(np.array(v, np.float32), name)

    inits = [
        numpy_helper.from_array(np.array([0], np.int64), 'starts'),
        numpy_helper.from_array(np.array([1], np.int64), 'ends'),
        numpy_helper.from_array(np.array([1], np.int64), 'axis1'),
        numpy_helper.from_array(np.array(2, np.int64), 'h_at'),
        numpy_helper.from_array(np.array(3, np.int64), 'w_at'),
        numpy_helper.from_array(np.array([1], np.int64), 'unsq_1'),
        numpy_helper.from_array(np.array([0], np.int64), 'unsq_0'),
        f32('zero', 0.0),
        f32('one', 1.0),
    ]
    nodes = [
        helper.make_node('Cast', ['x'], ['x_f'], to=TensorProto.FLOAT),
        helper.make_node('Slice', ['x_f', 'starts', 'ends', 'axis1'], ['plane']),
        helper.make_node('Mul', ['plane', 'zero'], ['zeros']),
        helper.make_node('Shape', ['x_f'], ['shape']),
        helper.make_node('Gather', ['shape', 'h_at'], ['h']),
        helper.make_node('Gather', ['shape', 'w_at'], ['w']),
        helper.make_node('Cast', ['h'], ['h_f'], to=TensorProto.FLOAT),
        helper.make_node('Cast', ['w'], ['w_f'], to=TensorProto.FLOAT),
        helper.make_node('Range', ['zero', 'h_f', 'one'], ['rows']),
        helper.make_node('Range', ['zero', 'w_f', 'one'], ['cols']),
    ]

    def band(index: str, lo: int, hi: int, tag: str) -> str:
        """Nodes for ``lo <= index < hi`` (as ``index > lo - 0.5`` and ``index < hi - 0.5``)."""
        inits.extend([f32(f'{tag}_lo', lo - 0.5), f32(f'{tag}_hi', hi - 0.5)])
        nodes.extend([
            helper.make_node('Greater', [index, f'{tag}_lo'], [f'{tag}_ge']),
            helper.make_node('Less', [index, f'{tag}_hi'], [f'{tag}_lt']),
            helper.make_node('And', [f'{tag}_ge', f'{tag}_lt'], [f'{tag}_in']),
        ])  # fmt: skip
        return f'{tag}_in'

    row_bands = [band('rows', lo, hi, f'r{k}') for k, (lo, hi) in enumerate(TEXT_ROW_BANDS)]
    acc = row_bands[0]
    for k, nxt in enumerate(row_bands[1:]):
        nodes.append(helper.make_node('Or', [acc, nxt], [f'r_or{k}']))
        acc = f'r_or{k}'
    col = band('cols', *TEXT_COL_BAND, 'c')
    nodes += [
        helper.make_node('Cast', [acc], ['r_f'], to=TensorProto.FLOAT),
        helper.make_node('Cast', [col], ['c_f'], to=TensorProto.FLOAT),
        helper.make_node('Unsqueeze', ['r_f', 'unsq_1'], ['r_col']),
        helper.make_node('Unsqueeze', ['c_f', 'unsq_0'], ['c_row']),
        helper.make_node('Mul', ['r_col', 'c_row'], ['map']),
        helper.make_node('Add', ['zeros', 'map'], ['fetch_name_0']),
    ]
    graph = helper.make_graph(
        nodes,
        p.null_name,
        [helper.make_tensor_value_info('x', TensorProto.UINT8, ['B', 3, 'H', 'W'])],
        [helper.make_tensor_value_info('fetch_name_0', TensorProto.FLOAT, ['B', 1, 'H', 'W'])],
        inits,
    )
    return _checked(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)]))


def _text_lines_onnx(p: ProductNull) -> ModelProto:
    """``x`` (UINT8 ``[B, 3, 48, W]``) -> ``indices`` (INT32 ``[B, T]``) and ``scores`` (FP32
    ``[B, T]``), T = ceil(W / :data:`TEXT_REC_STEP`): every step :data:`TEXT_REC_CLASS` at
    :data:`TEXT_REC_SCORE`."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    inits = [
        numpy_helper.from_array(np.array([0, 0, 0], np.int64), 'starts'),
        numpy_helper.from_array(np.array([1, 1, np.iinfo(np.int64).max], np.int64), 'ends'),
        numpy_helper.from_array(np.array([1, 2, 3], np.int64), 'axes'),
        numpy_helper.from_array(np.array([1, 1, TEXT_REC_STEP], np.int64), 'steps'),
        numpy_helper.from_array(np.array([0, -1], np.int64), 'bt'),
        numpy_helper.from_array(np.array(0.0, np.float32), 'zero'),
        numpy_helper.from_array(np.array(TEXT_REC_SCORE, np.float32), 'score'),
        numpy_helper.from_array(np.array(float(TEXT_REC_CLASS), np.float32), 'klass'),
    ]
    nodes = [
        helper.make_node('Cast', ['x'], ['x_f'], to=TensorProto.FLOAT),
        helper.make_node('Slice', ['x_f', 'starts', 'ends', 'axes', 'steps'], ['steps_f']),
        helper.make_node('Reshape', ['steps_f', 'bt'], ['steps_bt']),
        helper.make_node('Mul', ['steps_bt', 'zero'], ['zeros']),
        helper.make_node('Add', ['zeros', 'score'], ['scores']),
        helper.make_node('Add', ['zeros', 'klass'], ['klass_f']),
        helper.make_node('Cast', ['klass_f'], ['indices'], to=TensorProto.INT32),
    ]
    graph = helper.make_graph(
        nodes,
        p.null_name,
        [helper.make_tensor_value_info('x', TensorProto.UINT8, ['B', 3, 48, 'W'])],
        [
            helper.make_tensor_value_info('indices', TensorProto.INT32, ['B', 'T']),
            helper.make_tensor_value_info('scores', TensorProto.FLOAT, ['B', 'T']),
        ],
        inits,
    )
    return _checked(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)]))


def product_null_onnx(p: ProductNull) -> ModelProto:
    """The product null's graph (see :class:`ProductNull` and the constants above)."""
    import numpy as np

    if p.kind == 'scrfd':
        answers = scrfd_answer()
        return _constant_answers(p.input_name, SCRFD_NET,
                                 {n: answers[n] for n in scrfd_output_names()}, p.null_name)  # fmt: skip
    if p.kind == 'face_vector':
        vector = np.full((FACE_VECTOR_DIM,), 1.0 / np.sqrt(FACE_VECTOR_DIM), np.float32)
        return _constant_answers(p.input_name, FACE_SIDE, {'output': vector}, p.null_name)
    if p.kind == 'text_map':
        return _text_map_onnx(p)
    return _text_lines_onnx(p)


def product_trtexec_args(p: ProductNull, repo: str = SERVER_STAND_IN_DIR) -> list[str]:
    """``trtexec`` arguments that build a product null's plan over the real model's input range."""
    lo, opt, hi = p.shapes
    return [
        f'--onnx={repo}/_onnx/{p.null_name}.onnx',
        f'--saveEngine={repo}/{p.null_name}/1/model.plan',
        f'--minShapes={p.input_name}:{lo}',
        f'--optShapes={p.input_name}:{opt}',
        f'--maxShapes={p.input_name}:{hi}',
        f'--memPoolSize=workspace:{TRTEXEC_WORKSPACE}',
        '--skipInference',
    ]


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
    return [[*prefix, *trtexec_args(s, BUILD_STAND_IN_DIR)] for s in SPECS] + [
        [*prefix, *product_trtexec_args(p, BUILD_STAND_IN_DIR)] for p in PRODUCT_NULLS
    ]


def generate(
    out: Path, *, onnx: bool, detector_instances: int, encoder_instances: int
) -> list[str]:
    """Write the six models' configs, the capture models' code, the product nulls' configs and,
    with ``onnx``, the graphs of every null model. Returns what was written."""
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
    for p in PRODUCT_NULLS:
        put(out / p.null_name / 'config.pbtxt', product_null_config(p))
        (out / p.null_name / '1').mkdir(parents=True, exist_ok=True)
        if onnx:
            import onnx as onnx_lib

            graph = out / '_onnx' / f'{p.null_name}.onnx'
            graph.parent.mkdir(parents=True, exist_ok=True)
            onnx_lib.save(product_null_onnx(p), str(graph))
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
    names.add_argument('--kind', choices=['null', 'capture', 'product'], required=True)
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
        if args.kind == 'product':
            print(json.dumps(product_null_names()))
        else:
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
