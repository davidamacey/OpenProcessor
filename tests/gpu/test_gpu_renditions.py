"""Stored renditions from the resident frame against Pillow's thumbnail (ground truth).

Run like ``test_gpu_decode_parity.py`` (GPU tier; ``OP_TEST_FRAME_ENGINE_GRPC`` names an engine whose
Triton has ``pe_image_encoder``). The reference is ``ImageOps.exif_transpose`` + ``Image.thumbnail``
(LANCZOS) from the original file; the pixels transformed by the inverse of each EXIF op make the
upright source the second, independent truth. Greyscale, CMYK and progressive files are decoded on the
host (``frame_engine/host_decode.py``) and get the same renditions from the same kernels.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageOps

from src.domain import frame_engine_wire as wire


pytestmark = pytest.mark.gpu

SAMPLE = Path(__file__).resolve().parents[1] / 'fixtures' / 'sample_image.jpg'
PE_MODEL = 'pe_image_encoder'
SIDES = (128, 256)
PSNR_RAW_MIN = 40.0  # the resample alone, vs Pillow's LANCZOS thumbnail
PSNR_BYTES_GAP_MAX = (
    1.0  # dB the delivered q85 JPEG may fall below Pillow's own q85 of the reference
)
INVERSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}


@pytest.fixture(scope='module')
def engine() -> Any:
    from tritonclient.grpc import InferenceServerClient

    url = os.environ.get('OP_TEST_FRAME_ENGINE_GRPC')
    if not url:
        pytest.fail('GPU tier needs the frame engine gRPC address in OP_TEST_FRAME_ENGINE_GRPC')
    e = InferenceServerClient(url=url)
    try:
        ready = e.is_model_ready(wire.MODEL_NAME)
    except Exception as exc:
        pytest.fail(f'GPU tier needs a frame engine at {url}: {exc}')
    assert ready, f'the frame engine at {url} is not ready'
    return e


def _plan() -> wire.EnginePlan:
    return wire.EnginePlan(
        branches=('embed_frame',),
        embed=wire.EmbedSpec(PE_MODEL, 336, 32),
        derivatives=tuple(wire.RenditionSpec(f's{s}', s) for s in SIDES),
    )


def _ask(engine: Any, data: bytes) -> tuple[str, dict[str, Any], list[bytes]]:
    from tritonclient.grpc import InferInput

    inputs = []
    for name, value in ((wire.IMAGE, data), (wire.PLAN, _plan().to_json().encode())):
        inp = InferInput(name, [1], 'BYTES')
        inp.set_data_from_numpy(np.array([value], dtype=np.object_))
        inputs.append(inp)
    r = engine.infer(wire.MODEL_NAME, inputs, client_timeout=120)
    status = bytes(r.as_numpy('status')[0]).decode()
    branches = json.loads(r.as_numpy('branch_status')[0])
    jpegs = r.as_numpy(wire.DERIVATIVE_JPEGS)
    return status, branches, [bytes(j) for j in jpegs] if jpegs is not None else []


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * float(np.log10(255**2 / mse))


def _ref(img: Image.Image, side: int) -> np.ndarray:
    im = img.copy()
    im.thumbnail((side, side), Image.Resampling.LANCZOS)
    return np.asarray(im.convert('RGB'))


def _tagged(upright: Image.Image, orientation: int) -> bytes:
    stored = upright if orientation == 1 else upright.transpose(INVERSE[orientation])
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    stored.save(buf, 'JPEG', quality=95, exif=exif.tobytes())
    return buf.getvalue()


@pytest.mark.parametrize('orientation', range(1, 9))
def test_rendition_shows_the_upright_photo_like_pillow(engine: Any, orientation: int) -> None:
    upright = Image.open(SAMPLE).convert('RGB')
    data = _tagged(upright, orientation)
    status, branches, jpegs = _ask(engine, data)
    assert status == 'ok'
    assert branches['derivatives'] == {'status': 'ok'}
    assert len(jpegs) == len(SIDES)
    ref_img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    for side, jpeg in zip(SIDES, jpegs, strict=True):
        got = np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB'))
        ref = _ref(ref_img, side)
        assert got.shape == ref.shape
        pillow_q = io.BytesIO()
        Image.fromarray(ref).save(pillow_q, 'JPEG', quality=wire.DEFAULT_RENDITION_QUALITY)
        floor = _psnr(np.asarray(Image.open(pillow_q).convert('RGB')), ref)
        assert _psnr(got, ref) >= floor - PSNR_BYTES_GAP_MAX
        assert _psnr(got, _ref(upright, side)) > 28  # upright: a wrong rotation is far below this


@pytest.mark.parametrize('kind', ['grey', 'cmyk', 'progressive'])
def test_the_files_nvjpeg_does_not_take_get_their_renditions_from_the_host_decoded_frame(
    engine: Any, kind: str
) -> None:
    img = Image.open(SAMPLE).convert('RGB')
    buf = io.BytesIO()
    if kind == 'grey':
        img.convert('L').save(buf, 'JPEG')
    elif kind == 'cmyk':
        img.convert('CMYK').save(buf, 'JPEG')
    else:
        img.save(buf, 'JPEG', progressive=True)
    status, branches, jpegs = _ask(engine, buf.getvalue())
    assert status == 'ok'
    assert branches['derivatives'] == {'status': 'ok'}
    assert len(jpegs) == len(SIDES)
    ref_img = ImageOps.exif_transpose(Image.open(io.BytesIO(buf.getvalue())))
    for side, jpeg in zip(SIDES, jpegs, strict=True):
        got = np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB'))
        ref = _ref(ref_img, side)
        assert got.shape == ref.shape
        assert _psnr(got, ref) > 28
