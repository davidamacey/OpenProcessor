"""Tests for src.services.detection.pe_preprocess (T-4: restores three
dropped preprocessing cases from the pre-Wave-2 zero-coverage audit)."""

from __future__ import annotations

import numpy as np

from src.services.detection.pe_preprocess import (
    PE_IMAGENET_MEAN,
    PE_IMAGENET_STD,
    PE_SIZE,
    normalize_chw,
    resize_crop_rgb,
)


class TestNormalizeChwShapeAndDtype:
    def test_single_image_output_shape_and_dtype(self) -> None:
        rgb = np.random.default_rng(0).integers(0, 256, size=(PE_SIZE, PE_SIZE, 3), dtype=np.uint8)
        chw = normalize_chw(rgb)
        assert chw.shape == (3, PE_SIZE, PE_SIZE)
        assert chw.dtype == np.float32

    def test_batch_output_shape_and_dtype(self) -> None:
        rgb = np.random.default_rng(0).integers(
            0, 256, size=(4, PE_SIZE, PE_SIZE, 3), dtype=np.uint8
        )
        chw = normalize_chw(rgb)
        assert chw.shape == (4, 3, PE_SIZE, PE_SIZE)
        assert chw.dtype == np.float32


class TestNormalizeChwImagenetMean:
    def test_mean_pixel_normalizes_to_near_zero(self) -> None:
        # A flat image at exactly the ImageNet mean (as a 0-255 uint8
        # value) should normalize to ~0 in every channel.
        mean_rgb = np.round(PE_IMAGENET_MEAN.reshape(3) * 255).astype(np.uint8)
        flat = np.tile(mean_rgb.reshape(1, 1, 3), (PE_SIZE, PE_SIZE, 1))
        chw = normalize_chw(flat)
        assert np.abs(chw.mean()) < 0.05

    def test_normalization_matches_manual_computation(self) -> None:
        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        chw = normalize_chw(rgb)
        expected = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        expected = (expected - PE_IMAGENET_MEAN) / PE_IMAGENET_STD
        np.testing.assert_allclose(chw, expected, atol=1e-6)


class TestResizeCropRgbZeroSize:
    def test_zero_height_returns_all_zeros(self) -> None:
        rgb = np.zeros((0, 10, 3), dtype=np.uint8)
        out = resize_crop_rgb(rgb, target=64)
        assert out.shape == (64, 64, 3)
        assert np.all(out == 0)

    def test_zero_width_returns_all_zeros(self) -> None:
        rgb = np.zeros((10, 0, 3), dtype=np.uint8)
        out = resize_crop_rgb(rgb, target=64)
        assert out.shape == (64, 64, 3)
        assert np.all(out == 0)

    def test_zero_size_then_normalize_is_all_zero_mean_shifted(self) -> None:
        """A degenerate crop should preprocess to a deterministic
        constant (the negative-mean/std offset), never NaN/inf."""
        rgb = np.zeros((0, 0, 3), dtype=np.uint8)
        resized = resize_crop_rgb(rgb, target=PE_SIZE)
        chw = normalize_chw(resized)
        assert chw.shape == (3, PE_SIZE, PE_SIZE)
        assert np.isfinite(chw).all()


class TestResizeCropRgbNonDegenerate:
    def test_resizes_and_center_crops_to_target_square(self) -> None:
        rgb = np.random.default_rng(0).integers(0, 256, size=(100, 200, 3), dtype=np.uint8)
        out = resize_crop_rgb(rgb, target=64)
        assert out.shape == (64, 64, 3)

    def test_output_never_smaller_than_target(self) -> None:
        rgb = np.random.default_rng(0).integers(0, 256, size=(10, 500, 3), dtype=np.uint8)
        out = resize_crop_rgb(rgb, target=32)
        assert out.shape == (32, 32, 3)
