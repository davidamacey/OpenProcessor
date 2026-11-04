"""``center_crop_cpu`` must always return exactly (3, target_size,
target_size), for any input crop size.

Root cause: ``new_w = int(orig_w * scale)`` truncates instead of rounds.
``scale = target_size / min(orig_h, orig_w)`` is constructed so that
``round(min(orig_h, orig_w) * scale) == target_size`` exactly, but float
rounding error can put the true product a hair under ``target_size``
(e.g. 255.999999...), and ``int()`` truncates that down to
``target_size - 1``. The subsequent center-crop then read past the
resized array on one axis, so ``start_y``/``start_x`` went negative and
numpy's slicing silently returned a smaller (sometimes near-empty) crop
instead of raising.

On a real ingest batch this showed up as ``np.stack`` failing with "all
input arrays must have the same shape" once two boxes on the same
image (or across images in a batch) happened to round differently --
e.g. a 640x480 COCO image with several small detection crops
(``000000008690.jpg``).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.cpu_preprocess import center_crop_cpu


# Exact (h, w) crop sizes that reproduced a non-(256, 256) output under the
# old int()-truncation math (found by scanning h, w in [1, 500)).
KNOWN_BAD_SIZES = [(49, 49), (49, 50), (49, 55), (97, 99), (145, 149)]


@pytest.mark.parametrize(('h', 'w'), KNOWN_BAD_SIZES)
def test_center_crop_cpu_always_returns_the_target_size(h: int, w: int) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    out = center_crop_cpu(img, target_size=256)
    assert out.shape == (3, 256, 256), f'{h}x{w} crop -> {out.shape}'


def test_center_crop_cpu_outputs_stack_across_mixed_small_box_sizes() -> None:
    """The actual failure mode: np.stack over several box crops of
    different (small) sizes must succeed because every crop is exactly
    target_size x target_size."""
    crops = [np.zeros((h, w, 3), dtype=np.uint8) for h, w in KNOWN_BAD_SIZES]
    preprocessed = [center_crop_cpu(c, target_size=256) for c in crops]
    batch = np.stack(preprocessed)  # raised ValueError before the fix
    assert batch.shape == (len(KNOWN_BAD_SIZES), 3, 256, 256)


@pytest.mark.parametrize(('h', 'w'), [(1, 1), (2, 3), (1, 500), (500, 1)])
def test_center_crop_cpu_handles_degenerate_and_extreme_aspect_ratios(h: int, w: int) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    out = center_crop_cpu(img, target_size=256)
    assert out.shape == (3, 256, 256)


def _reference(img: np.ndarray, target: int = 256) -> np.ndarray:
    """The full-resize-then-crop definition of the centre crop."""
    import cv2

    h, w = img.shape[:2]
    scale = target / min(h, w)
    nw, nh = max(target, round(w * scale)), max(target, round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    sx, sy = max(0, (nw - target) // 2), max(0, (nh - target) // 2)
    crop = resized[sy : sy + target, sx : sx + target]
    return np.transpose(crop.astype(np.float32) / 255.0, (2, 0, 1))


@pytest.mark.parametrize('shape', [(256, 256), (480, 640), (640, 480)])
def test_normal_crops_are_byte_identical_to_the_full_resize(shape: tuple[int, int]) -> None:
    img = np.random.default_rng(0).integers(0, 255, (*shape, 3), dtype=np.uint8)
    assert np.array_equal(center_crop_cpu(img), _reference(img))


def test_a_sliver_crop_does_not_allocate_the_full_resize() -> None:
    import tracemalloc

    img = np.random.default_rng(1).integers(0, 255, (1, 3000, 3), dtype=np.uint8)
    tracemalloc.start()
    out = center_crop_cpu(img)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert out.shape == (3, 256, 256)
    assert peak < 20 * 1024 * 1024


def test_a_sliver_crop_matches_the_full_resize_within_rounding() -> None:
    img = np.random.default_rng(2).integers(0, 255, (60, 900, 3), dtype=np.uint8)
    assert np.abs(center_crop_cpu(img) - _reference(img)).max() < 0.02
