"""The MobileCLIP image preprocessing has one implementation (``center_crop_cpu``)."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.clients.triton_preprocess import preprocess_clip_cpu, preprocess_for_mobileclip


def _png(size: tuple[int, int], value: int = 200) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', size, (value, value, value)).save(buf, format='PNG')
    return buf.getvalue()


def test_a_300x187_image_has_no_black_padded_row() -> None:
    out = preprocess_for_mobileclip(_png((300, 187)))
    assert out.shape == (1, 3, 256, 256)
    # A uniform grey image stays uniform; the old int() truncation left a row of zeros.
    assert out.min() > 0.7


def test_a_480_short_side_stays_close_to_the_old_pil_output() -> None:
    """The resampler changed (cv2 bilinear instead of Pillow); a smooth image barely moves."""
    x = np.linspace(0, 255, 640, dtype=np.uint8)
    arr = np.broadcast_to(x[None, :, None], (480, 640, 3)).copy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')

    img = Image.fromarray(arr).resize((341, 256), Image.Resampling.BILINEAR)
    left = (341 - 256) // 2
    old = np.transpose(np.array(img.crop((left, 0, left + 256, 256))) / 255.0, (2, 0, 1))
    new = preprocess_for_mobileclip(buf.getvalue())[0]
    assert np.abs(new - old).max() < 0.03


def test_the_array_and_bytes_paths_agree() -> None:
    arr = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    assert np.array_equal(preprocess_for_mobileclip(buf.getvalue()), preprocess_clip_cpu(arr))
