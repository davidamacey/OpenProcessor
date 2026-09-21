"""Tests for src.services.detection.geometry — shared bbox/crop math."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.services.detection.geometry import (
    bbox_norm,
    crop_id,
    crop_to_jpeg,
    iou,
    letterbox_to_square,
    roi_pool_sppf,
    undo_letterbox,
)


class TestBboxNorm:
    def test_basic(self) -> None:
        assert bbox_norm((10.0, 20.0, 30.0, 40.0), 100, 200) == pytest.approx([0.1, 0.1, 0.3, 0.2])

    def test_clamped_to_bounds(self) -> None:
        result = bbox_norm((-10.0, -5.0, 150.0, 250.0), 100, 200)
        assert result == pytest.approx([0.0, 0.0, 1.0, 1.0])

    def test_zero_size_image_does_not_divide_by_zero(self) -> None:
        result = bbox_norm((0.0, 0.0, 10.0, 10.0), 0, 0)
        assert result == pytest.approx([0.0, 0.0, 1.0, 1.0])


class TestCropId:
    def test_deterministic(self) -> None:
        a = crop_id('img1', [0.1, 0.2, 0.3, 0.4])
        b = crop_id('img1', [0.1, 0.2, 0.3, 0.4])
        assert a == b
        assert len(a) == 32

    def test_differs_by_image_id(self) -> None:
        bbox = [0.1, 0.2, 0.3, 0.4]
        assert crop_id('img1', bbox) != crop_id('img2', bbox)

    def test_differs_by_bbox(self) -> None:
        assert crop_id('img1', [0.1, 0.2, 0.3, 0.4]) != crop_id('img1', [0.1, 0.2, 0.3, 0.5])

    def test_matches_across_list_and_tuple_input(self) -> None:
        assert crop_id('img1', [0.1, 0.2, 0.3, 0.4]) == crop_id('img1', (0.1, 0.2, 0.3, 0.4))


class TestIou:
    def test_identical_boxes(self) -> None:
        box = (0.0, 0.0, 10.0, 10.0)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0

    def test_partial_overlap(self) -> None:
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        # intersection = 5x5=25, union = 100+100-25=175
        assert iou(a, b) == pytest.approx(25.0 / 175.0)

    def test_degenerate_box_is_zero_area_not_error(self) -> None:
        assert iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 10.0, 10.0)) == 0.0

    def test_touching_edges_no_overlap(self) -> None:
        assert iou((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0)) == 0.0


class TestLetterboxRoundTrip:
    def test_shape_and_dtype(self) -> None:
        img = Image.new('RGB', (200, 100), (50, 100, 150))
        chw, scale, pad = letterbox_to_square(img, target=64)
        assert chw.shape == (1, 3, 64, 64)
        assert chw.dtype == np.float32
        assert chw.min() >= 0.0
        assert chw.max() <= 1.0
        assert scale == pytest.approx(64 / 200)
        assert pad[0] == pytest.approx(0.0)
        assert pad[1] > 0.0

    def test_undo_letterbox_round_trip_sub_pixel(self) -> None:
        img = Image.new('RGB', (300, 150), (0, 0, 0))
        target = 128
        _, scale, pad = letterbox_to_square(img, target=target)

        # A box in original-image pixel space -> letterbox space -> undo
        # should recover the original within sub-pixel tolerance.
        orig_box = (30.0, 15.0, 200.0, 100.0)
        x1, y1, x2, y2 = orig_box
        letter_box = (
            x1 * scale + pad[0],
            y1 * scale + pad[1],
            x2 * scale + pad[0],
            y2 * scale + pad[1],
        )
        recovered = undo_letterbox(letter_box, scale, pad)
        for a, b in zip(orig_box, recovered, strict=True):
            assert a == pytest.approx(b, abs=1e-6)

    def test_zero_size_raises(self) -> None:
        img = Image.new('RGB', (0, 0))
        with pytest.raises(ValueError, match='degenerate'):
            letterbox_to_square(img, target=64)


class TestCropToJpeg:
    def test_round_trips_as_valid_jpeg(self) -> None:
        img = Image.new('RGB', (32, 32), (10, 20, 30))
        data = crop_to_jpeg(img, quality=80)
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert data[:2] == b'\xff\xd8'  # JPEG SOI marker


class TestRoiPoolSppf:
    def test_shape_and_normalization(self) -> None:
        sppf = np.random.default_rng(0).normal(size=(768, 40, 40)).astype(np.float32)
        pooled = roi_pool_sppf(sppf, (100.0, 100.0, 300.0, 300.0), input_size=1280, target_dim=1024)
        assert pooled.shape == (1024,)
        assert pooled.dtype == np.float32
        norm = float(np.linalg.norm(pooled))
        assert norm == pytest.approx(1.0, abs=1e-4)

    def test_degenerate_bbox_falls_back_to_single_cell(self) -> None:
        sppf = np.ones((16, 10, 10), dtype=np.float32)
        pooled = roi_pool_sppf(sppf, (5.0, 5.0, 5.0, 5.0), input_size=320, target_dim=16)
        assert pooled.shape == (16,)
        assert np.isfinite(pooled).all()
