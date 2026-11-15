"""The crop square of ``embed_crops`` is the whole box, letterboxed (``frame_engine/branches/embed.py``).

The branch runs against a NumPy stand-in of the resampling kernel (the same separable weights,
``prep_geometry.filter_weights``, widened or not as the call asks) and is compared with a Pillow
reference of the rule: the box scaled to fit the square with its aspect kept (Pillow ``BILINEAR``:
antialiased for a box larger than the square, plain bilinear for a smaller one), centred on a
square of the plan's fill. The centre-crop rule it replaced kept the middle of a wide box only; the
tests fail if any part of the box is cut away, the geometry moves, or a box above the square size
is resampled without the antialiasing.
"""

from __future__ import annotations

import contextlib
import types
from typing import Any

import numpy as np
import pytest
from PIL import Image

from frame_engine.branches.embed import EmbedCrops
from frame_engine.prep_geometry import filter_weights
from src.domain import frame_engine_wire as wire
from src.domain.image_geometry import letterbox_geometry


FRAME_W, FRAME_H = 120, 90
SIZE = 24


def _resample(
    frame: np.ndarray, roi: tuple[int, int, int, int], size: tuple[int, int], *, antialias: bool
) -> np.ndarray:
    """``frame[roi]`` to ``size`` (w, h) with the kernel's triangle weights, HWC float."""
    x0, y0, rw, rh = roi
    ow, oh = size
    sub = frame[y0 : y0 + rh, x0 : x0 + rw].astype(np.float64)
    for axis, n_in, n_out in ((1, rw, ow), (0, rh, oh)):
        first, count, w = filter_weights(n_in, n_out, antialias=antialias)
        taps = [
            np.tensordot(
                w[i, : count[i]],
                np.take(sub, range(first[i], first[i] + count[i]), axis),
                (0, axis),
            )
            for i in range(n_out)
        ]
        sub = np.stack(taps, axis=axis)
    return sub


class _Kernels:
    """What ``PrepKernels.resample_into`` does, on the host: write CHW uint8 into ``dst``."""

    def resample_into(self, frame: Any, roi: Any, size: Any, dst: Any, *_a: Any, **k: Any) -> None:
        hwc = _resample(frame, roi, size, antialias=k['antialias'])
        hwc = np.clip(np.rint(hwc), 0, 255).astype(np.uint8)
        assert dst.shape == (3, size[1], size[0])  # the view is the fitted rectangle, nothing more
        dst[...] = hwc.transpose(2, 0, 1)


class _Calls:
    def __init__(self) -> None:
        self.staged: list[Any] = []

    def stage(self, nbytes: int, _stage: str) -> Any:
        s = types.SimpleNamespace(
            device=np.full(nbytes, 7, np.uint8)
        )  # not the fill: a stale slice
        self.staged.append(s)
        return s

    def submit(
        self, _owner: Any, _stage: str, staged: Any, model: str, tensor: Any, _out: Any
    ) -> Any:
        return types.SimpleNamespace(staged=staged, model=model, tensor=tensor)


class _Clock:
    @contextlib.contextmanager
    def stage(self, _name: str) -> Any:
        yield


def _frame() -> np.ndarray:
    return np.random.default_rng(3).integers(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8)


def reference_square(
    frame: np.ndarray, rect: tuple[int, int, int, int], size: int, fill: int
) -> np.ndarray:
    """The rule in Pillow: CHW uint8, the whole rectangle scaled to fit, fill elsewhere."""
    x1, y1, x2, y2 = rect
    crop = frame[y1:y2, x1:x2]
    g = letterbox_geometry(x2 - x1, y2 - y1, size)
    out = np.full((size, size, 3), fill, np.uint8)
    fitted = Image.fromarray(np.ascontiguousarray(crop)).resize(
        (g.new_w, g.new_h), Image.Resampling.BILINEAR
    )
    out[g.top : g.top + g.new_h, g.left : g.left + g.new_w] = np.asarray(fitted)
    return out.transpose(2, 0, 1)


def _run(
    rects: list[tuple[int, int, int, int]], frame: np.ndarray, *, fill: int = 128
) -> np.ndarray:
    spec = wire.CropEmbedSpec('crop_model', SIZE, 8, 'letterbox', fill)
    branch = EmbedCrops(spec, None)
    calls = _Calls()
    photo = types.SimpleNamespace(
        rects=rects, classes=None, calls=calls, clock=_Clock(), kernels=_Kernels(),
        frame=frame, size=(FRAME_W, FRAME_H),
    )  # fmt: skip
    branch.round2(photo)  # type: ignore[arg-type]
    return np.concatenate([s.device.reshape(-1, 3, SIZE, SIZE) for s in calls.staged])[: len(rects)]


RECTS = {
    'wide': (4, 40, 100, 52),
    'tall': (50, 5, 58, 85),
    'tiny': (30, 30, 33, 32),
    'enlarged': (10, 60, 20, 68),
    'enlarged_wide': (20, 20, 40, 26),
    'at_the_size': (30, 30, 30 + SIZE, 30 + SIZE - 4),
    'just_over': (30, 30, 30 + SIZE + 1, 30 + SIZE),
    'one_pixel': (60, 60, 61, 61),
    'full_frame': (0, 0, FRAME_W, FRAME_H),
    'square': (10, 10, 50, 50),
    'left_border': (0, 10, 20, 70),
    'right_bottom_border': (FRAME_W - 30, FRAME_H - 12, FRAME_W, FRAME_H),
}


@pytest.mark.parametrize('name', list(RECTS))
def test_the_crop_square_is_the_whole_box_letterboxed(name: str) -> None:
    frame = _frame()
    got = _run([RECTS[name]], frame)[0]
    want = reference_square(frame, RECTS[name], SIZE, 128)
    assert got.shape == want.shape
    assert np.abs(got.astype(int) - want.astype(int)).max() <= 1


ABOVE = [n for n, (x1, y1, x2, y2) in RECTS.items() if max(x2 - x1, y2 - y1) > SIZE]
BELOW = [n for n, (x1, y1, x2, y2) in RECTS.items() if max(x2 - x1, y2 - y1) < SIZE]


def test_the_rects_cover_boxes_above_and_below_the_square_size() -> None:
    assert {'wide', 'tall', 'full_frame', 'just_over'} <= set(ABOVE)
    assert {'tiny', 'one_pixel', 'enlarged', 'enlarged_wide'} <= set(BELOW)


@pytest.mark.parametrize('name', ['wide', 'full_frame', 'square'])
def test_a_box_above_the_size_is_not_the_plain_bilinear(name: str) -> None:
    """Guards the guard: on these boxes the antialiased square and the plain one differ well beyond
    the one-grey-level agreement the engine and the reference are held to."""
    frame = _frame()
    x1, y1, x2, y2 = RECTS[name]
    g = letterbox_geometry(x2 - x1, y2 - y1, SIZE)
    plain = _resample(frame, (x1, y1, x2 - x1, y2 - y1), (g.new_w, g.new_h), antialias=False)
    anti = _resample(frame, (x1, y1, x2 - x1, y2 - y1), (g.new_w, g.new_h), antialias=True)
    assert np.abs(plain - anti).max() > 8


def test_every_crop_of_a_chunk_is_its_own_square_and_the_pad_is_the_plan_fill() -> None:
    frame = _frame()
    rects = list(RECTS.values())
    got = _run(rects, frame, fill=114)
    assert got.shape == (len(rects), 3, SIZE, SIZE)
    for k, rect in enumerate(rects):
        want = reference_square(frame, rect, SIZE, 114)
        assert np.abs(got[k].astype(int) - want.astype(int)).max() <= 1, k
    wide = got[list(RECTS).index('wide')]
    assert (wide[:, :3, :] == 114).all()  # the pad above a wide box, taken from the plan, not 7


def test_a_wide_box_keeps_both_of_its_ends() -> None:
    """The centre-crop rule kept the middle square of this box and dropped its two ends."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    frame[40:52, 4:12] = (255, 0, 0)  # the box's left end
    frame[40:52, 92:100] = (0, 255, 0)  # its right end
    sq = _run([RECTS['wide']], frame)[0]
    assert sq[0].max() == 255  # red, the box's left end, is in the square
    assert sq[1].max() == 255  # green, its right end, too
    g = letterbox_geometry(96, 12, SIZE)
    assert sq[0, g.top + 1, g.left] > 100  # the first fitted column is the box's first column
    assert sq[1, g.top + 1, g.left + g.new_w - 1] > 100  # and the last is its last


@pytest.mark.parametrize('name', list(RECTS))
def test_the_engine_square_is_the_squares_the_apis_host_copy_cuts(name: str) -> None:
    """One rule, two implementations (GPU kernels, Pillow on the host): the same pixels, for a box
    above the square size (antialiased) and below it (plain bilinear)."""
    from src.utils.pe_preprocess import letterbox_crop_rgb

    frame = _frame()
    x1, y1, x2, y2 = RECTS[name]
    host = letterbox_crop_rgb(frame[y1:y2, x1:x2], SIZE).transpose(2, 0, 1)
    engine = _run([RECTS[name]], frame)[0]
    assert np.abs(host.astype(int) - engine.astype(int)).max() <= 1
