"""Colour on the GPU route against littleCMS (ground truth), through the engine's renditions.

Run like ``test_gpu_renditions.py`` (GPU tier; ``OP_TEST_FRAME_ENGINE_GRPC`` names an engine whose
Triton has ``pe_image_encoder``). A Display P3 file is the public sample converted to P3 by
littleCMS with the profile embedded; the reference is Pillow's decode of that file converted back
to sRGB by ``ImageCms`` and, for a thumbnail, ``Image.thumbnail`` (LANCZOS). A rendition at the
photo's own size is the frame itself (no resample), so it shows the converted frame directly.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageCms

from src.domain import frame_engine_wire as wire


pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'fixtures' / 'sample_image.jpg'
P3 = (ROOT / 'fixtures' / 'icc' / 'DisplayP3-v4.icc').read_bytes()
COLORD_SRGB = (ROOT / 'fixtures' / 'icc' / 'colord-sRGB.icc').read_bytes()
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
P3_PROFILE = ImageCms.ImageCmsProfile(io.BytesIO(P3))
TO_P3 = ImageCms.buildTransform(SRGB, P3_PROFILE, 'RGB', 'RGB')
TO_SRGB = ImageCms.buildTransform(P3_PROFILE, SRGB, 'RGB', 'RGB')
THUMB = 256
FULL = wire.RENDITION_SIDE_RANGE[1]  # larger than the sample: the frame itself, never enlarged


@pytest.fixture(scope='module')
def engine() -> Any:
    from tritonclient.grpc import InferenceServerClient

    url = os.environ.get('OP_TEST_FRAME_ENGINE_GRPC')
    if not url:
        pytest.fail('GPU tier needs the frame engine gRPC address in OP_TEST_FRAME_ENGINE_GRPC')
    e = InferenceServerClient(url=url)
    assert e.is_model_ready(wire.MODEL_NAME), f'the frame engine at {url} is not ready'
    return e


def _ask(engine: Any, data: bytes) -> list[np.ndarray]:
    from tritonclient.grpc import InferInput

    plan = wire.EnginePlan(
        branches=('embed_frame',),
        embed=wire.EmbedSpec('pe_image_encoder', 336, 32),
        derivatives=(wire.RenditionSpec('t', THUMB, 95), wire.RenditionSpec('full', FULL, 100)),
    )
    inputs = []
    for name, value in ((wire.IMAGE, data), (wire.PLAN, plan.to_json().encode())):
        inp = InferInput(name, [1], 'BYTES')
        inp.set_data_from_numpy(np.array([value], dtype=np.object_))
        inputs.append(inp)
    r = engine.infer(wire.MODEL_NAME, inputs, client_timeout=120)
    assert bytes(r.as_numpy('status')[0]).decode() == 'ok'
    assert json.loads(r.as_numpy('branch_status')[0])['derivatives'] == {'status': 'ok'}
    return [np.asarray(Image.open(io.BytesIO(bytes(j))).convert('RGB'))
            for j in r.as_numpy(wire.DERIVATIVE_JPEGS)]  # fmt: skip


def _jpeg(img: Image.Image, icc: bytes | None) -> bytes:
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=95, **({'icc_profile': icc} if icc else {}))
    return buf.getvalue()


def _thumb(img: Image.Image) -> np.ndarray:
    im = img.copy()
    im.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
    return np.asarray(im)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * float(np.log10(255**2 / mse))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def test_a_p3_photo_is_converted_before_its_renditions(engine: Any) -> None:
    srgb = Image.open(SAMPLE).convert('RGB')
    p3_pixels = TO_P3.apply(srgb)
    data = _jpeg(p3_pixels, P3)
    decoded = Image.open(io.BytesIO(data)).convert('RGB')  # the samples, profile ignored
    converted = TO_SRGB.apply(decoded)
    thumb, full = _ask(engine, data)
    # the frame: within JPEG q100 noise of littleCMS, clearly away from the unconverted samples
    assert _mae(full, np.asarray(converted)) < 0.5
    assert _mae(full, np.asarray(decoded)) > 2 * _mae(full, np.asarray(converted))
    # the thumbnail: Pillow's thumbnail of the converted photo (A3: renditions are sRGB)
    assert _psnr(thumb, _thumb(converted)) > 40
    assert _psnr(thumb, _thumb(converted)) > _psnr(thumb, _thumb(decoded)) + 3


def test_an_srgb_tagged_photo_is_the_untagged_photo_bit_for_bit(engine: Any) -> None:
    img = Image.open(SAMPLE).convert('RGB')
    untagged = _ask(engine, _jpeg(img, None))
    tagged = _ask(engine, _jpeg(img, COLORD_SRGB))
    for a, b in zip(untagged, tagged, strict=True):
        assert np.array_equal(a, b)
