"""GPU tier: the frame engine's ``faces`` and ``face_embed`` branches against the reference path.

The reference is InsightFace ``v0.7`` (``SCRFD.detect`` + ``norm_crop`` + ``ArcFaceONNX``) run on the
CPU around the served FP32 models: OpenCV decode, the letterbox (top-left, ``cv2.resize``), the
same anchor decode and NMS (``frame_engine.faces.scrfd``, numpy), ``cv2.warpAffine`` to the ArcFace
template, the recogniser input ``(x - 127.5) / 127.5``. The engine decodes with nvJPEG, resamples
and warps on the GPU and calls its uint8 builds. For a face reduced to 112 pixels the engine
averages ``n x n`` bilinear taps per output pixel (an area filter; the reference samples once and
aliases), so its oracle there is the same warp at ``n x 112`` reduced by ``INTER_AREA``.

The engine under test is either engine (the native one through ``OP_TEST_FRAME_ENGINE_GRPC``);
the last test also asks the Python engine (``OP_TEST_PY_FRAME_ENGINE_GRPC``) the same photos under
a tiled plan (coarse square and tiles) and holds the two answers to each other.

Run (needs the engine, its model server with the uint8 builds, and the served FP32 models):

    OP_TEST_TRITON_GRPC=127.0.0.1:4731 OP_TEST_FRAME_ENGINE_GRPC=127.0.0.1:4744 \\
    OP_TEST_PY_FRAME_ENGINE_GRPC=127.0.0.1:4745 \\
    OP_TEST_FACE_DETECTOR=<uint8 SCRFD> OP_TEST_FACE_RECOGNISER=<uint8 ArcFace> \\
    OP_TEST_FACE_IMAGES=<dir of public JPEGs with faces> \\
        .venv/bin/python -m pytest tests/gpu/test_gpu_faces.py -q --no-cov -m gpu

A missing setting or an unreachable server FAILS these tests; they are never skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from frame_engine.faces import alignment, scrfd
from src.domain import frame_engine_wire as wire


pytestmark = pytest.mark.gpu

SAMPLE = Path(__file__).resolve().parents[1] / 'fixtures' / 'sample_image.jpg'
N_IMAGES = 40
BOX_ATOL_PX = 1.0  # same detector at the same size: decode and resample differences only
COSINE_P01_MIN = 0.995  # the F1 gate, against the oracle of each face


def _env(name: str) -> str:
    value = os.environ.get(name, '')
    if not value:
        pytest.fail(f'{name} is not set (docstring)')
    return value


@pytest.fixture(scope='module')
def clients() -> tuple[Any, Any]:
    import tritonclient.grpc as grpcclient

    return (
        grpcclient.InferenceServerClient(_env('OP_TEST_TRITON_GRPC')),
        grpcclient.InferenceServerClient(_env('OP_TEST_FRAME_ENGINE_GRPC')),
    )


def _faces_plan(**extra: int) -> str:
    faces = {
        'detector': _env('OP_TEST_FACE_DETECTOR'),
        'input_size': 640,
        'recogniser': _env('OP_TEST_FACE_RECOGNISER'),
        'batch': 32,
        'score_min': 0.5,
        'nms': 0.4,
        'max_faces': 128,
        **extra,
    }
    return json.dumps({'version': 1, 'branches': ['faces', 'face_embed'], 'crops': 'none',
                       'faces': faces})  # fmt: skip


def _engine(client: Any, data: bytes, plan: str | None = None) -> dict[str, Any]:
    import tritonclient.grpc as grpcclient

    inputs = []
    for name, value in ((wire.IMAGE, data), (wire.PLAN, (plan or _faces_plan()).encode())):
        inp = grpcclient.InferInput(name, [1], 'BYTES')
        inp.set_data_from_numpy(np.array([value], dtype=np.object_))
        inputs.append(inp)
    res = client.infer(wire.MODEL_NAME, inputs, client_timeout=120)
    assert res.as_numpy(wire.STATUS)[0].decode() == 'ok'
    states = wire.decode_branch_status(res.as_numpy(wire.BRANCH_STATUS)[0])
    assert states == {'faces': ('ok', ''), 'face_embed': ('ok', '')}
    names = (*wire.BRANCH_OUTPUTS['faces'], *wire.BRANCH_OUTPUTS['face_embed'])
    return {n: res.as_numpy(n) for n in names}


def _aligned(img: np.ndarray, lmk: np.ndarray) -> np.ndarray:
    """The reference's ``norm_crop``, or for a reduced face the area-filtered warp: the warp at
    ``n x 112`` (each big pixel at the centre of one of the ``n x n`` taps) then ``INTER_AREA``."""
    fwd = alignment.estimate(lmk)
    _, taps, _ = alignment.warp_params(lmk[None])
    n = int(taps[0])
    if n == 1:
        return cv2.warpAffine(img, fwd, (112, 112), borderValue=0.0)
    big = n * fwd
    big[:, 2] += (n - 1) / 2
    wide = cv2.warpAffine(img, big, (112 * n, 112 * n), borderValue=0.0)
    return cv2.resize(wide, (112, 112), interpolation=cv2.INTER_AREA)


def _reference(client: Any, data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boxes, scores, landmarks and unit vectors of the reference path, best score first."""
    import tritonclient.grpc as grpcclient

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    h, w = img.shape[:2]
    nw, nh = scrfd.fit(w, h, 640)
    canvas = np.zeros((640, 640, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (640, 640), (127.5,) * 3, swapRB=True)
    inp = grpcclient.InferInput('input.1', list(blob.shape), 'FP32')
    inp.set_data_from_numpy(blob.astype(np.float32))
    res = client.infer('scrfd_10g_bnkps', [inp], client_timeout=60)
    outs = {n: res.as_numpy(n)[0] for n in scrfd.OUTPUT_NAMES}
    passes = [scrfd.whole_pass(w, h, 640)]
    faces = scrfd.merge(
        [scrfd.to_source(passes[0], w, h, *scrfd.decode(outs, 640, 640, 0.5))], 0.4, 128
    )
    if not len(faces.scores):
        return faces.boxes, faces.scores, faces.landmarks, np.zeros((0, 512), np.float32)
    crops = np.stack([_aligned(img, k) for k in faces.landmarks])
    x = (crops[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) - 127.5) / 127.5
    inp = grpcclient.InferInput('input', list(x.shape), 'FP32')
    inp.set_data_from_numpy(np.ascontiguousarray(x))
    v = client.infer('arcface_w600k_r50', [inp], client_timeout=60).as_numpy('output')
    return faces.boxes, faces.scores, faces.landmarks, v / np.linalg.norm(v, axis=1, keepdims=True)


def test_a_photo_without_a_face_answers_no_face(clients: tuple[Any, Any]) -> None:
    out = _engine(clients[1], SAMPLE.read_bytes())
    assert out[wire.FACE_BOXES].shape == (0, 4)
    assert out[wire.FACE_EMBEDDINGS].shape[0] == 0
    assert out[wire.FACE_OVERFLOW].tolist() == [0]


def test_faces_and_vectors_agree_with_the_reference(clients: tuple[Any, Any]) -> None:
    """Every reference face has the engine face over it within ``BOX_ATOL_PX`` and the same
    vector (cosine first percentile). Catches a colour swap, a normalisation off the reference, a
    shifted letterbox or warp, a filter that is not the area average, and a dropped face."""
    files = sorted(Path(_env('OP_TEST_FACE_IMAGES')).rglob('*.jpg'))[:N_IMAGES]
    assert files, 'OP_TEST_FACE_IMAGES has no JPEG'
    cosines: list[float] = []
    faces = 0
    for f in files:
        data = f.read_bytes()
        rb, _, _, rv = _reference(clients[0], data)
        out = _engine(clients[1], data)
        eb = out[wire.FACE_BOXES]
        ev = np.zeros((len(eb), 512), np.float32)
        if out[wire.FACE_EMBEDDED].any():
            ev[out[wire.FACE_EMBEDDED]] = out[wire.FACE_EMBEDDINGS]
        for b, v in zip(rb, rv, strict=True):
            faces += 1
            assert len(eb), f'{f.name}: no engine face for {b}'
            k = int(np.argmin(np.abs(eb - b).max(axis=1)))
            assert np.abs(eb[k] - b).max() <= BOX_ATOL_PX, f'{f.name}: {eb[k]} vs {b}'
            cosines.append(float(ev[k] @ v))
    assert faces >= 10, f'only {faces} reference faces in {len(files)} photos'
    assert np.percentile(cosines, 1) >= COSINE_P01_MIN, sorted(cosines)[:5]


#: Every pass kind: the whole frame, a coarse square and, above 1920 pixels, tiles.
TILED = {'coarse': 320, 'tile_side': 1920, 'overlap': 64}


def test_the_engine_answers_as_the_python_engine_with_tiles() -> None:
    """The engine under test and the Python engine on the same photos under a tiled plan: the same
    faces (count, boxes and landmarks within ``BOX_ATOL_PX``), the same overflow and embedded
    flags, and vectors whose cosine first percentile clears ``COSINE_P01_MIN``. Catches a tile grid,
    an edge drop, a per-axis scale, an NMS order or a warp that is not the Python engine's."""
    import tritonclient.grpc as grpcclient

    engine = grpcclient.InferenceServerClient(_env('OP_TEST_FRAME_ENGINE_GRPC'))
    python = grpcclient.InferenceServerClient(_env('OP_TEST_PY_FRAME_ENGINE_GRPC'))
    files = sorted(Path(_env('OP_TEST_FACE_IMAGES')).rglob('*.jpg'))[:N_IMAGES]
    assert files, 'OP_TEST_FACE_IMAGES has no JPEG'
    plan = _faces_plan(**TILED)
    cosines: list[float] = []
    for f in files:
        data = f.read_bytes()
        got, want = _engine(engine, data, plan), _engine(python, data, plan)
        assert got[wire.FACE_BOXES].shape == want[wire.FACE_BOXES].shape, f.name
        for name in (wire.FACE_BOXES, wire.FACE_LANDMARKS):
            diff = np.abs(got[name] - want[name]).max(initial=0.0)
            assert diff <= BOX_ATOL_PX, f'{f.name}: {name} {diff}'
        assert got[wire.FACE_OVERFLOW].tolist() == want[wire.FACE_OVERFLOW].tolist(), f.name
        assert got[wire.FACE_EMBEDDED].tolist() == want[wire.FACE_EMBEDDED].tolist(), f.name
        rows = (got[wire.FACE_EMBEDDINGS] * want[wire.FACE_EMBEDDINGS]).sum(axis=1)
        cosines.extend(float(c) for c in rows)
    assert len(cosines) >= 10, f'only {len(cosines)} faces in {len(files)} photos'
    assert np.percentile(cosines, 1) >= COSINE_P01_MIN, sorted(cosines)[:5]
