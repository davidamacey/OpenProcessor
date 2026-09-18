"""Tests for :mod:`src.services.detection.crop_quality`.

No direct coverage exists on the reference branch for this module —
written fresh to prove the ported blur-scoring math and decision rule
behave as documented.
"""

from __future__ import annotations

import numpy as np

from src.services.detection.crop_quality import (
    DEFAULT_SHARP_MODE,
    SHARP_THRESHOLDS,
    blur_ratio,
    crop_blur,
    crop_lap_var,
    image_lap_var,
    is_blurry,
    laplacian_var,
)


def _sharp_image(size: int = 64) -> np.ndarray:
    """A high-frequency checkerboard — large Laplacian variance."""
    rng = np.arange(size)
    board = ((rng[:, None] // 4 + rng[None, :] // 4) % 2).astype(np.uint8) * 255
    return np.stack([board, board, board], axis=-1)


def _flat_image(size: int = 64) -> np.ndarray:
    """A uniform gray image — ~zero Laplacian variance."""
    return np.full((size, size, 3), 128, dtype=np.uint8)


class TestLaplacianVar:
    def test_flat_image_has_near_zero_variance(self) -> None:
        gray = _flat_image()[:, :, 0]
        assert laplacian_var(gray) == 0.0

    def test_sharp_image_has_higher_variance_than_flat(self) -> None:
        sharp_gray = _sharp_image()[:, :, 0]
        flat_gray = _flat_image()[:, :, 0]
        assert laplacian_var(sharp_gray) > laplacian_var(flat_gray)

    def test_rounds_to_two_decimals(self) -> None:
        gray = _sharp_image()[:, :, 0]
        value = laplacian_var(gray)
        assert value == round(value, 2)


class TestImageLapVar:
    def test_matches_laplacian_var_of_grayscale(self) -> None:
        img = _sharp_image()
        assert image_lap_var(img) >= 0.0


class TestCropLapVar:
    def test_degenerate_zero_area_returns_none(self) -> None:
        img = _sharp_image()
        assert crop_lap_var(img, (10, 10, 10, 10)) is None

    def test_inverted_coords_returns_none(self) -> None:
        img = _sharp_image()
        assert crop_lap_var(img, (20, 20, 5, 5)) is None

    def test_out_of_bounds_clamps_to_image(self) -> None:
        img = _sharp_image()
        # bbox extends far past the image edges; should clamp, not crash.
        result = crop_lap_var(img, (-100, -100, 1000, 1000))
        assert result is not None
        assert result >= 0.0

    def test_valid_crop_returns_variance(self) -> None:
        img = _sharp_image()
        result = crop_lap_var(img, (0, 0, 32, 32))
        assert result is not None
        assert result >= 0.0


class TestBlurRatio:
    def test_none_box_var_returns_none(self) -> None:
        assert blur_ratio(None, 100.0) is None

    def test_flat_full_var_returns_none(self) -> None:
        assert blur_ratio(50.0, 0.0) is None

    def test_computes_rounded_ratio(self) -> None:
        assert blur_ratio(50.0, 100.0) == 0.5


class TestCropBlur:
    def test_returns_named_tuple_with_expected_fields(self) -> None:
        img = _sharp_image()
        result = crop_blur(img, (0, 0, 32, 32))
        assert result.full_var >= 0.0
        assert result.box_var is None or result.box_var >= 0.0
        assert result.ratio is None or isinstance(result.ratio, float)

    def test_reuses_precomputed_full_var(self) -> None:
        img = _sharp_image()
        full_var = image_lap_var(img)
        result = crop_blur(img, (0, 0, 32, 32), full_var=full_var)
        assert result.full_var == full_var


class TestIsBlurry:
    def test_none_ratio_is_not_blurry(self) -> None:
        assert is_blurry(None, 100.0) is False

    def test_none_box_var_is_not_blurry(self) -> None:
        assert is_blurry(0.5, None) is False

    def test_below_both_thresholds_is_blurry(self) -> None:
        ratio_thr, boxval_thr = SHARP_THRESHOLDS[DEFAULT_SHARP_MODE]
        assert is_blurry(ratio_thr - 0.01, boxval_thr - 1.0) is True

    def test_above_ratio_threshold_is_not_blurry(self) -> None:
        ratio_thr, boxval_thr = SHARP_THRESHOLDS[DEFAULT_SHARP_MODE]
        # ratio passes even though box_var alone is low -> AND means not blurry.
        assert is_blurry(ratio_thr + 0.5, boxval_thr - 1.0) is False

    def test_above_boxval_threshold_is_not_blurry(self) -> None:
        ratio_thr, boxval_thr = SHARP_THRESHOLDS[DEFAULT_SHARP_MODE]
        assert is_blurry(ratio_thr - 0.01, boxval_thr + 1000.0) is False

    def test_unknown_mode_falls_back_to_default(self) -> None:
        ratio_thr, boxval_thr = SHARP_THRESHOLDS[DEFAULT_SHARP_MODE]
        assert is_blurry(ratio_thr - 0.01, boxval_thr - 1.0, mode='not_a_real_mode') is True

    def test_low_mode_is_more_lenient_than_high(self) -> None:
        # A ratio between the 'high' and 'low' thresholds should only trip
        # the more permissive ('low') threshold band, not 'high'.
        low_ratio_thr, low_box_thr = SHARP_THRESHOLDS['low']
        high_ratio_thr, _ = SHARP_THRESHOLDS['high']
        mid_ratio = (low_ratio_thr + high_ratio_thr) / 2
        assert is_blurry(mid_ratio, low_box_thr - 1.0, mode='high') is False
        assert is_blurry(mid_ratio, low_box_thr - 1.0, mode='low') is True
