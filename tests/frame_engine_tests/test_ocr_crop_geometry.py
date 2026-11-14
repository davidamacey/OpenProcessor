"""``frame_engine/ocr/crop_geometry.py``: the composed crop map against PaddleOCR's two-step cut.

The reference is ``get_rotate_crop_image`` (``cv2.warpPerspective``, cubic, replicated border, a
quarter turn when tall) followed by ``cv2.resize`` to 48 high (PaddleX release/3.3). The map is
sampled here in numpy with one bilinear tap per pixel (what the kernel does with one tap): on a
smooth image the two must agree to within the bicubic-against-bilinear difference.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from frame_engine.ocr.batching import REC_HEIGHT
from frame_engine.ocr.crop_geometry import crop_dims, is_tall, rec_map, rec_ratio, taps_for


def _q(points: Any) -> np.ndarray:
    return np.array(points, np.float32)


def _image(h: int = 240, w: int = 320) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 127 + 100 * np.sin(x / 23.0) * np.cos(y / 31.0)
    g = 127 + 100 * np.sin((x + y) / 41.0)
    b = 60 + 0.5 * x + 0.3 * y
    return np.clip(np.stack([b, g, r], axis=2), 0, 255).astype(np.uint8)


def _reference(img: np.ndarray, quad: np.ndarray, width: int) -> np.ndarray:
    cw, ch = crop_dims(quad)
    std = _q([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    m = cv2.getPerspectiveTransform(_q(quad), std)
    dst = cv2.warpPerspective(
        img, m, (cw, ch), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )
    if dst.shape[0] / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return cv2.resize(np.ascontiguousarray(dst), (width, REC_HEIGHT))


def _sample(img: np.ndarray, g: np.ndarray, width: int) -> np.ndarray:
    v, u = np.mgrid[0:REC_HEIGHT, 0:width].astype(np.float64)
    p = np.einsum('ij,jhw->ihw', g, np.stack([u, v, np.ones_like(u)]))
    x, y = p[0] / p[2], p[1] / p[2]
    h, w = img.shape[:2]
    x0, y0 = np.floor(x), np.floor(y)
    ax, ay = (x - x0)[..., None], (y - y0)[..., None]
    xi, yi = x0.astype(int), y0.astype(int)

    def px(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        return img[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)].astype(np.float64)

    out = (
        px(yi, xi) * (1 - ax) * (1 - ay) + px(yi, xi + 1) * ax * (1 - ay)
        + px(yi + 1, xi) * (1 - ax) * ay + px(yi + 1, xi + 1) * ax * ay
    )  # fmt: skip
    return np.rint(out).astype(np.uint8)


QUADS = {
    'flat': [[40, 60], [200, 60], [200, 100], [40, 100]],
    'slanted': [[50, 40], [210, 70], [205, 105], [45, 75]],
    'tall': [[100, 20], [130, 22], [128, 180], [98, 178]],
    'perspective': [[30, 50], [230, 70], [215, 120], [45, 105]],
}


@pytest.mark.parametrize('name', QUADS)
def test_the_composed_map_reads_what_the_two_step_cut_reads(name: str) -> None:
    img, quad = _image(), _q(QUADS[name])
    ratio = rec_ratio(quad)
    assert ratio is not None
    width = int(np.ceil(REC_HEIGHT * ratio))
    got = _sample(img, rec_map(quad, width), width)
    want = _reference(img, quad, width)
    diff = np.abs(got.astype(int) - want.astype(int))
    assert diff.mean() < 1.5, diff.mean()
    assert np.percentile(diff, 99) < 8


def test_a_tall_quad_is_turned_and_a_flat_one_is_not() -> None:
    assert is_tall(*crop_dims(_q(QUADS['tall'])))
    assert not is_tall(*crop_dims(_q(QUADS['flat'])))
    assert rec_ratio(_q(QUADS['tall'])) == pytest.approx(158 / 30, rel=0.05)


def test_the_turn_is_counter_clockwise() -> None:
    """The top-left corner of a tall quad ends up at the bottom-left of the recogniser input."""
    quad = _q(QUADS['tall'])
    img = np.full((240, 320, 3), 128, np.uint8)
    img[15:25, 95:105] = (0, 0, 255)  # a red mark at the quad's top-left corner
    ratio = rec_ratio(quad)
    assert ratio is not None
    width = int(np.ceil(REC_HEIGHT * ratio))
    out = _sample(img, rec_map(quad, width), width)
    red = (out[..., 2] > 200) & (out[..., 1] < 80)
    ys, xs = np.nonzero(red)
    assert len(ys) > 0
    assert ys.mean() > REC_HEIGHT / 2
    assert xs.mean() < width / 4


def test_a_degenerate_quad_has_no_ratio() -> None:
    assert rec_ratio(_q([[5, 5], [5.4, 5], [5.4, 5.4], [5, 5.4]])) is None


def test_taps_follow_the_downscale_and_respect_the_cap() -> None:
    quad = _q([[0, 0], [1000, 0], [1000, 200], [0, 200]])  # 200 -> 48 high: 4.2 per pixel
    ratio = rec_ratio(quad)
    assert ratio is not None
    width = int(np.ceil(REC_HEIGHT * ratio))
    g = rec_map(quad, width)
    assert taps_for(g, width, 8) == (5, 5)
    assert taps_for(g, width, 3) == (3, 3)
    assert taps_for(g, width, 1) == (1, 1)
    small = _q([[0, 0], [60, 0], [60, 24], [0, 24]])
    assert taps_for(rec_map(small, 120), 120, 8) == (1, 1)  # upscaled: one tap
