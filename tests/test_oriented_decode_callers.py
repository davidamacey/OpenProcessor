"""Every decode path returns the upright frame: sizes agree for EXIF tags 1-8.

The reference is Pillow's ``ImageOps.exif_transpose`` (the documented way to apply the tag).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageOps

from src.clients.triton_preprocess import preprocess_clip_cpu, preprocess_for_mobileclip
from src.schemas.analyze import DetectionResult
from src.services.analyze_pipeline import crop_batch_tensor
from src.utils.affine import get_jpeg_dimensions_fast
from src.utils.image_processing import decode_image as decode_bgr


def _tagged(tag: int, fmt: str = 'JPEG', size: tuple[int, int] = (64, 32)) -> bytes:
    rng = np.random.default_rng(tag)
    img = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0112] = tag
    buf = io.BytesIO()
    img.save(buf, format=fmt, exif=exif, **({'quality': 95} if fmt == 'JPEG' else {}))
    return buf.getvalue()


def _upright(data: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')


@pytest.mark.parametrize('tag', range(1, 9))
def test_header_size_is_the_upright_size(tag: int) -> None:
    data = _tagged(tag)
    assert get_jpeg_dimensions_fast(data) == _upright(data).size


@pytest.mark.parametrize('tag', range(1, 9))
def test_embed_preprocessing_sees_the_upright_frame(tag: int) -> None:
    data = _tagged(tag)
    want = preprocess_clip_cpu(np.asarray(_upright(data)))
    np.testing.assert_array_equal(preprocess_for_mobileclip(data), want)


@pytest.mark.parametrize('tag', range(1, 9))
def test_analyze_crops_are_cut_from_the_upright_frame(tag: int) -> None:
    data = _tagged(tag)
    w, h = _upright(data).size
    det = DetectionResult(box=[0.0, 0.0, 1.0, 1.0], confidence=0.9, class_id=0, class_name='x')
    got = crop_batch_tensor(data, [det], w, h)
    want = preprocess_clip_cpu(np.asarray(_upright(data)))[0]
    np.testing.assert_array_equal(got[0], want)


@pytest.mark.parametrize('tag', range(1, 9))
def test_cv2_path_and_pillow_fallback_both_orient(tag: int) -> None:
    data = _tagged(tag, 'WEBP')
    # WebP carries EXIF; OpenCV may or may not apply it, the fallback must give the same size.
    assert decode_bgr(data, 'x.webp').shape[:2] == _upright(data).size[::-1]


@pytest.mark.parametrize('tag', range(1, 9))
def test_pillow_fallback_of_the_cv2_decoder_orients(
    tag: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('src.utils.image_processing.cv2.imdecode', lambda *_a, **_k: None)
    data = _tagged(tag)
    assert decode_bgr(data, 'x.jpg').shape[:2] == _upright(data).size[::-1]
