"""Tests for src.services.detection.pe_preprocess (restores three
dropped preprocessing cases from an earlier zero-coverage audit)."""

from __future__ import annotations

import numpy as np

from src.services.detection.pe_preprocess import (
    PE_MEAN,
    PE_SIZE,
    PE_STD,
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
        # A flat image at exactly the PE mean (as a 0-255 uint8
        # value) should normalize to ~0 in every channel.
        mean_rgb = np.round(PE_MEAN.reshape(3) * 255).astype(np.uint8)
        flat = np.tile(mean_rgb.reshape(1, 1, 3), (PE_SIZE, PE_SIZE, 1))
        chw = normalize_chw(flat)
        assert np.abs(chw.mean()) < 0.05

    def test_normalization_matches_manual_computation(self) -> None:
        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        chw = normalize_chw(rgb)
        expected = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        expected = (expected - PE_MEAN) / PE_STD
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


class TestWholeFrameFromBytes:
    """Byte-upload ingest embeds from memory; it must land in the same space
    as the from-disk path for the same file."""

    def test_bytes_path_matches_file_path(self, tmp_path) -> None:
        import io

        from PIL import Image

        from src.services.detection.pe_preprocess import whole_frame_chw, whole_frame_chw_from_bytes

        arr = np.random.default_rng(3).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='JPEG', quality=90)
        data = buf.getvalue()
        path = tmp_path / 'frame.jpg'
        path.write_bytes(data)

        from_disk = whole_frame_chw(str(path))
        from_bytes = whole_frame_chw_from_bytes(data)
        assert from_disk is not None
        assert from_bytes is not None
        assert from_bytes.shape == (3, PE_SIZE, PE_SIZE)
        np.testing.assert_array_equal(from_bytes, from_disk)

    def test_undecodable_and_empty_bytes_return_none(self) -> None:
        from src.services.detection.pe_preprocess import whole_frame_chw_from_bytes

        assert whole_frame_chw_from_bytes(b'') is None
        assert whole_frame_chw_from_bytes(b'not an image') is None


def test_normalization_matches_upstream_pe_core() -> None:
    """PE-Core's own transform is Normalize([0.5]*3, [0.5]*3), not ImageNet."""
    assert np.allclose(PE_MEAN.reshape(3), 0.5)
    assert np.allclose(PE_STD.reshape(3), 0.5)


def test_bakeoff_sampler_uses_the_same_normalization() -> None:
    from scripts.curation.bakeoff import sample

    assert np.array_equal(sample._PE_MEAN, PE_MEAN)
    assert np.array_equal(sample._PE_STD, PE_STD)


def _legacy_normalize_chw(rgb_uint8: np.ndarray) -> np.ndarray:
    """The per-pixel arithmetic normalize_chw ran before the lookup table (the oracle)."""
    arr = rgb_uint8.astype(np.float32) / 255.0
    if arr.ndim == 4:
        chw = np.transpose(arr, (0, 3, 1, 2))
        return ((chw - PE_MEAN[None]) / PE_STD[None]).astype(np.float32, copy=False)
    chw = np.transpose(arr, (2, 0, 1))
    return ((chw - PE_MEAN) / PE_STD).astype(np.float32, copy=False)


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


class TestNormalizeLookupIsBitIdentical:
    """P-pe: the lookup table returns the exact float32 the arithmetic produced."""

    def test_every_uint8_value_on_every_channel(self) -> None:
        values = np.arange(256, dtype=np.uint8)
        rgb = np.stack([values, values[::-1], np.roll(values, 7)], axis=-1).reshape(16, 16, 3)
        got = np.ascontiguousarray(normalize_chw(rgb))
        assert _same_bits(got, np.ascontiguousarray(_legacy_normalize_chw(rgb)))

    def test_batch_path(self) -> None:
        rgb = np.random.default_rng(2).integers(0, 256, size=(3, 40, 30, 3), dtype=np.uint8)
        got = np.ascontiguousarray(normalize_chw(rgb))
        assert _same_bits(got, np.ascontiguousarray(_legacy_normalize_chw(rgb)))

    def test_normalize_into_matches_for_a_non_contiguous_crop(self) -> None:
        from src.services.detection.pe_preprocess import normalize_into

        big = np.random.default_rng(3).integers(0, 256, size=(90, 80, 3), dtype=np.uint8)
        view = big[10:50, 20:70]  # a resize_crop_rgb result is such a view
        out = np.empty((3, 40, 50), dtype=np.float32)
        normalize_into(out, view)
        assert _same_bits(out, np.ascontiguousarray(_legacy_normalize_chw(view)))

    def test_embed_crops_batch_equals_the_old_stack(self) -> None:
        from src.clients.pe_encoder import _preprocess_crops

        rng = np.random.default_rng(4)
        crops = [
            rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            for h, w in ((20, 30), (500, 410), (7, 9))
        ] + [np.zeros((0, 5, 3), dtype=np.uint8)]
        old = np.stack([_legacy_normalize_chw(resize_crop_rgb(c)) for c in crops], axis=0)
        assert _same_bits(_preprocess_crops(crops), old.astype(np.float32, copy=False))
