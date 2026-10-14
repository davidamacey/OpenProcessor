"""Tests for src.utils.pe_preprocess.

The PE engine takes uint8 pixels and normalizes in its graph (``export/uint8_input.py``), so
preprocessing here is resize, center-crop and a layout change, with no arithmetic.
"""

from __future__ import annotations

import numpy as np

from src.utils.pe_preprocess import PE_SIZE, resize_crop_rgb, to_chw, to_chw_into


class TestToChw:
    def test_single_image_is_chw_uint8_with_the_same_pixels(self) -> None:
        rgb = np.random.default_rng(0).integers(0, 256, size=(PE_SIZE, PE_SIZE, 3), dtype=np.uint8)
        chw = to_chw(rgb)
        assert chw.shape == (3, PE_SIZE, PE_SIZE)
        assert chw.dtype == np.uint8
        assert chw.flags['C_CONTIGUOUS']
        np.testing.assert_array_equal(chw[1], rgb[..., 1])

    def test_batch_is_nchw(self) -> None:
        rgb = np.random.default_rng(0).integers(0, 256, size=(4, 20, 30, 3), dtype=np.uint8)
        chw = to_chw(rgb)
        assert chw.shape == (4, 3, 20, 30)
        np.testing.assert_array_equal(chw[2, 0], rgb[2, ..., 0])

    def test_into_a_slot_from_a_non_contiguous_crop(self) -> None:
        big = np.random.default_rng(3).integers(0, 256, size=(90, 80, 3), dtype=np.uint8)
        view = big[10:50, 20:70]  # a resize_crop_rgb result is such a view
        out = np.empty((3, 40, 50), dtype=np.uint8)
        to_chw_into(out, view)
        np.testing.assert_array_equal(out, to_chw(view))


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

    def test_zero_size_crop_becomes_a_black_frame(self) -> None:
        """A degenerate crop is a deterministic all-zero frame the engine normalizes to a constant."""
        rgb = np.zeros((0, 0, 3), dtype=np.uint8)
        chw = to_chw(resize_crop_rgb(rgb, target=PE_SIZE))
        assert chw.shape == (3, PE_SIZE, PE_SIZE)
        assert chw.dtype == np.uint8
        assert not chw.any()


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

        from src.utils.pe_preprocess import whole_frame_chw, whole_frame_chw_from_bytes

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
        from src.utils.pe_preprocess import whole_frame_chw_from_bytes

        assert whole_frame_chw_from_bytes(b'') is None
        assert whole_frame_chw_from_bytes(b'not an image') is None


def test_embed_crops_batch_is_the_stacked_uint8_layout() -> None:
    from src.clients.pe_encoder import _preprocess_crops

    rng = np.random.default_rng(4)
    crops = [
        rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        for h, w in ((20, 30), (500, 410), (7, 9))
    ] + [np.zeros((0, 5, 3), dtype=np.uint8)]
    expected = np.stack([to_chw(resize_crop_rgb(c)) for c in crops], axis=0)
    batch = _preprocess_crops(crops)
    assert batch.dtype == np.uint8
    np.testing.assert_array_equal(batch, expected)


def test_build_script_normalizes_with_the_pe_core_mean_and_std() -> None:
    """PE-Core's own transform is Normalize([0.5]*3, [0.5]*3), not ImageNet; the engine owns it."""
    from pathlib import Path

    script = (Path(__file__).resolve().parents[2] / 'export' / 'build_pe_trt.sh').read_text()
    assert '--mean 0.5 0.5 0.5 --std 0.5 0.5 0.5' in script
