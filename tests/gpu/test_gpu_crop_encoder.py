"""The crop encoder through the frame engine, on the GPU: the engine's letterbox squares against the host copy of the rule.

GPU tier, like ``test_gpu_decode_parity.py``: ``OP_TEST_FRAME_ENGINE_GRPC`` names an engine, ``OP_TEST_TRITON_GRPC`` the
model server it calls, ``OP_TEST_CROP_MODEL`` the crop encoder's name there (default ``pe_crop_encoder``; the lane
stack loads it under a lane name) and ``OP_TEST_CROP_SIZE`` its input size (default 224).

The truth is the host copy of the rule, ``src.utils.pe_preprocess.letterbox_crop_rgb`` (OpenCV on a Pillow-decoded
photo), sent straight to the crop model. The engine cuts the same square with its CUDA kernels from an nvJPEG decode, so
the two vectors differ by the decoders and the filters' rounding only; the centre-crop square (the retired rule) is
the negative control: it must be farther from the engine's vector than the letterbox is, on a wide box.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.domain import frame_engine_wire as wire
from src.domain.image_geometry import crop_rect, whole_square_geometry
from src.utils.pe_preprocess import letterbox_crop_rgb


pytestmark = pytest.mark.gpu

SAMPLE = Path(__file__).resolve().parents[1] / 'fixtures' / 'sample_image.jpg'
COS_MIN = 0.998  # two JPEG decoders and two bilinear implementations on a natural photo
SIZE = int(os.environ.get('OP_TEST_CROP_SIZE', '224'))
MODEL = os.environ.get('OP_TEST_CROP_MODEL', 'pe_crop_encoder')


def _client(env: str) -> Any:
    from tritonclient.grpc import InferenceServerClient

    url = os.environ.get(env)
    if not url:
        pytest.fail(f'GPU tier needs {env}')
    c = InferenceServerClient(url=url)
    try:
        c.is_server_ready()
    except Exception as exc:
        pytest.fail(f'GPU tier needs a server at {url}: {exc}')
    return c


@pytest.fixture(scope='module')
def engine() -> Any:
    c = _client('OP_TEST_FRAME_ENGINE_GRPC')
    assert c.is_model_ready(wire.MODEL_NAME)
    return c


@pytest.fixture(scope='module')
def triton() -> Any:
    c = _client('OP_TEST_TRITON_GRPC')
    assert c.is_model_ready(MODEL), f'{MODEL} is not loaded in the model server'
    return c


def _frame() -> tuple[bytes, Image.Image]:
    data = SAMPLE.read_bytes()
    return data, Image.open(SAMPLE).convert('RGB')


def _engine_vectors(
    engine: Any, data: bytes, rects: list[tuple[int, int, int, int]], size: tuple[int, int]
) -> np.ndarray:
    from tritonclient.grpc import InferInput

    w, h = size
    plan = wire.EnginePlan(
        branches=('embed_crops',), crop_embed=wire.CropEmbedSpec(MODEL, SIZE, 32)
    )
    inputs = []
    for name, value in ((wire.IMAGE, data), (wire.PLAN, plan.to_json().encode())):
        inp = InferInput(name, [1], 'BYTES')
        inp.set_data_from_numpy(np.array([value], dtype=np.object_))
        inputs.append(inp)
    boxes = np.array(rects, np.float64) / np.array([w, h, w, h], np.float64)
    b = InferInput(wire.BOXES, list(boxes.shape), 'FP64')
    b.set_data_from_numpy(boxes)
    inputs.append(b)
    r = engine.infer(wire.MODEL_NAME, inputs, client_timeout=120)
    assert bytes(r.as_numpy(wire.STATUS)[0]).decode() == 'ok'
    assert wire.decode_branch_status(r.as_numpy(wire.BRANCH_STATUS)[0])['embed_crops'] == ('ok', '')
    return np.asarray(r.as_numpy(wire.CROP_EMBEDDINGS), np.float32)


def _host_vectors(triton: Any, squares: list[np.ndarray]) -> np.ndarray:
    from tritonclient.grpc import InferInput

    x = np.ascontiguousarray(np.stack(squares).transpose(0, 3, 1, 2))
    inp = InferInput('images', list(x.shape), 'UINT8')
    inp.set_data_from_numpy(x)
    out = triton.infer(MODEL, [inp], client_timeout=120).as_numpy('image_embeddings')
    v = np.asarray(out, np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _centre_square(crop: np.ndarray) -> np.ndarray:
    import cv2

    h, w = crop.shape[:2]
    g = whole_square_geometry(w, h, SIZE)
    r = cv2.resize(crop, (g.new_w, g.new_h), interpolation=cv2.INTER_LINEAR)
    return r[g.top : g.top + SIZE, g.left : g.left + SIZE]


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    return (a * b).sum(1)


def _rects(w: int, h: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        'wide': crop_rect((w * 0.02, h * 0.40, w * 0.98, h * 0.55), w, h),
        'tall': crop_rect((w * 0.45, h * 0.02, w * 0.55, h * 0.98), w, h),
        'tiny': crop_rect((w * 0.50, h * 0.50, w * 0.50 + 14, h * 0.50 + 11), w, h),
        'full_frame': (0, 0, w, h),
        'border': crop_rect((w * 0.60, h * 0.55, w, h), w, h),
        'middle': crop_rect((w * 0.25, h * 0.25, w * 0.70, h * 0.80), w, h),
    }


def test_the_engines_crop_vector_is_the_vector_of_the_hosts_letterbox_square(
    engine: Any, triton: Any
) -> None:
    data, img = _frame()
    w, h = img.size
    rects = _rects(w, h)
    arr = np.asarray(img)
    squares = [letterbox_crop_rgb(arr[y1:y2, x1:x2], SIZE) for x1, y1, x2, y2 in rects.values()]
    got = _engine_vectors(engine, data, list(rects.values()), (w, h))
    want = _host_vectors(triton, squares)
    cos = _cos(got, want)
    by_box = dict(zip(rects, np.round(cos, 5), strict=True))
    assert cos.min() >= COS_MIN, by_box


def test_a_wide_box_is_not_cut_to_its_centre_square(engine: Any, triton: Any) -> None:
    """The retired rule is the negative control: against the engine's vector of a wide box the centre square is
    farther than the letterbox by a clear margin, so a regression to the old rule fails here."""
    data, img = _frame()
    w, h = img.size
    x1, y1, x2, y2 = _rects(w, h)['wide']
    crop = np.asarray(img)[y1:y2, x1:x2]
    got = _engine_vectors(engine, data, [(x1, y1, x2, y2)], (w, h))
    letterbox, centre = _host_vectors(
        triton, [letterbox_crop_rgb(crop, SIZE), _centre_square(crop)]
    )
    assert _cos(got, letterbox[None])[0] - _cos(got, centre[None])[0] > 0.05
