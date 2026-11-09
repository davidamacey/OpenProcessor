"""``_decode_yolo_output`` and the confidence-floor filter of the region detector, over the fused
``[300, 6]`` tensor (``x1, y1, x2, y2, score, class`` in network pixels)."""

from __future__ import annotations

import numpy as np
import pytest
from _region_profile_fixture import EXAMPLE_LICENSE_PLATE_PROFILE as REFERENCE_LICENSE_PLATE_PROFILE
from PIL import Image

from src.services.detection.cascade_detect.preprocess import _letterbox
from src.services.detection.cascade_detect.region_detector import (
    _decode_yolo_output,
    _decode_yolo_output_multi,
)


def _fused(*rows: tuple[float, float, float, float, float]) -> np.ndarray:
    """A fused ``[1, 300, 6]`` tensor holding ``rows`` (``x1, y1, x2, y2, score``), class 0."""
    raw = np.zeros((1, 300, 6), dtype=np.float32)
    for i, row in enumerate(rows):
        raw[0, i, :5] = row
    return raw


def _cxcywh(cx: float, cy: float, w: float, h: float, score: float) -> np.ndarray:
    """One planted detection given as center and size."""
    return _fused((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score))


# =============================================================================
# Decoder tests
# =============================================================================


class TestDecoder:
    """Pure ``_decode_yolo_output`` behavior — no Triton involved."""

    def test_decoder_round_trips_through_letterbox(self) -> None:
        """A planted box at the network center decodes to crop center."""
        crop_w, crop_h = 200, 100
        # 200x100 letterboxed onto 640x640 → scale=3.2, pad_h=(640-320)/2=160
        img = Image.new('RGB', (crop_w, crop_h), (10, 20, 30))
        _, scale, pad = _letterbox(img)

        # A tight 20x10 crop-px box at the center of the crop; the center of the crop in
        # network space is (320, 320).
        raw = _cxcywh(320.0, 320.0, 20.0 * scale, 10.0 * scale, 0.92)

        result = _decode_yolo_output(raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h)
        assert result is not None
        assert result.score == pytest.approx(0.92)
        x1, y1, x2, y2 = result.bbox_norm
        # Box is centered → midpoint should be ~0.5 in each dim.
        assert (x1 + x2) / 2 == pytest.approx(0.5, abs=0.01)
        assert (y1 + y2) / 2 == pytest.approx(0.5, abs=0.01)
        # Width 20/200 = 0.1; height 10/100 = 0.1.
        assert (x2 - x1) == pytest.approx(0.1, abs=0.01)
        assert (y2 - y1) == pytest.approx(0.1, abs=0.01)

    def test_xyxy_box_is_not_read_as_cxcywh(self) -> None:
        crop_w, crop_h = 200, 100
        _, scale, pad = _letterbox(Image.new('RGB', (crop_w, crop_h)))
        # Crop box (50, 25)-(150, 75) in network pixels.
        raw = _fused(
            (
                50 * scale + pad[0],
                25 * scale + pad[1],
                150 * scale + pad[0],
                75 * scale + pad[1],
                0.9,
            )
        )
        result = _decode_yolo_output(raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h)
        assert result is not None
        assert result.bbox_norm == pytest.approx((0.25, 0.25, 0.75, 0.75), abs=0.01)
        assert result.score == pytest.approx(0.9)

    def test_decoder_clamps_out_of_bounds(self) -> None:
        """A box that pokes past the crop edge gets clamped to [0, 1]."""
        crop_w, crop_h = 100, 100
        _, scale, pad = _letterbox(Image.new('RGB', (crop_w, crop_h)))

        # Plant a huge box that's bigger than the network input.
        raw = _cxcywh(320.0, 320.0, 10000.0, 10000.0, 0.95)

        result = _decode_yolo_output(raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h)
        assert result is not None
        assert result.bbox_norm == (0.0, 0.0, 1.0, 1.0)

    def test_decoder_drops_collapsed_box(self) -> None:
        """Boxes that collapse to a point after clamping return ``None``."""
        crop_w, crop_h = 100, 100
        _, scale, pad = _letterbox(Image.new('RGB', (crop_w, crop_h)))

        # Box well outside the crop's letterbox area → after clamp == (0,0,0,0)
        raw = _cxcywh(-500.0, -500.0, 1.0, 1.0, 0.95)

        assert _decode_yolo_output(raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h) is None

    def test_decoder_accepts_normalized_export(self) -> None:
        """Some exports emit boxes in [0, 1] of network input rather than pixels."""
        crop_w, crop_h = 200, 100
        _, scale, pad = _letterbox(Image.new('RGB', (crop_w, crop_h)))

        # Same logical detection, but emitted in normalized 0-1 space.
        raw = _cxcywh(0.5, 0.5, 0.1, 0.05, 0.9)

        result = _decode_yolo_output(raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h)
        assert result is not None
        assert result.score == pytest.approx(0.9)
        x1, y1, x2, y2 = result.bbox_norm
        assert 0.4 < (x1 + x2) / 2 < 0.6
        assert 0.4 < (y1 + y2) / 2 < 0.6

    def test_scale_is_decided_per_tensor_not_per_row(self) -> None:
        # A pixel-space tensor with one tiny box near the origin: that row alone looks
        # "normalized", but the tensor is pixels, so it must not be multiplied by 640.
        raw = _fused((0.2, 0.2, 1.0, 1.0, 0.9), (100, 100, 300, 300, 0.8))
        got = _decode_yolo_output_multi(raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640)
        tiny = min(got, key=lambda c: c.bbox_norm[2])
        assert tiny.bbox_norm[2] == pytest.approx(1.0 / 640, abs=1e-4)

    def test_normalized_tensor_is_scaled_to_pixels(self) -> None:
        raw = _fused((0.25, 0.25, 0.75, 0.75, 0.9))
        got = _decode_yolo_output_multi(raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640)
        assert got[0].bbox_norm == pytest.approx((0.25, 0.25, 0.75, 0.75))


# =============================================================================
# Confidence-floor filter
# =============================================================================


class TestConfidenceFloor:
    def test_drops_below_floor(self) -> None:
        """A detection below ``confidence_floor`` returns ``None``."""
        _, scale, pad = _letterbox(Image.new('RGB', (100, 100)))
        raw = _cxcywh(320.0, 320.0, 32.0, 16.0, 0.30)  # below default floor 0.4

        assert (
            _decode_yolo_output(
                raw,
                scale=scale,
                pad=pad,
                crop_w=100,
                crop_h=100,
                confidence_floor=REFERENCE_LICENSE_PLATE_PROFILE.confidence_floor,
            )
            is None
        )

    def test_keeps_at_floor(self) -> None:
        """Score exactly at the floor counts as a hit (>=, not >)."""
        _, scale, pad = _letterbox(Image.new('RGB', (100, 100)))
        floor = REFERENCE_LICENSE_PLATE_PROFILE.confidence_floor
        raw = _cxcywh(320.0, 320.0, 32.0, 16.0, floor)

        result = _decode_yolo_output(
            raw, scale=scale, pad=pad, crop_w=100, crop_h=100, confidence_floor=floor
        )
        assert result is not None

    def test_custom_floor_overrides_default(self) -> None:
        """Caller-supplied floor wins over the default."""
        _, scale, pad = _letterbox(Image.new('RGB', (100, 100)))
        raw = _cxcywh(320.0, 320.0, 32.0, 16.0, 0.55)

        # Strict floor at 0.6 → drop.
        assert (
            _decode_yolo_output(
                raw, scale=scale, pad=pad, crop_w=100, crop_h=100, confidence_floor=0.6
            )
            is None
        )
        # Looser floor at 0.5 → keep.
        result = _decode_yolo_output(
            raw, scale=scale, pad=pad, crop_w=100, crop_h=100, confidence_floor=0.5
        )
        assert result is not None
        assert result.score == pytest.approx(0.55)


# =============================================================================
# Multi-candidate decode and padding rows
# =============================================================================


class TestMultiAndPadding:
    def test_multi_keeps_every_row_above_floor(self) -> None:
        crop_w, crop_h = 200, 100
        _, scale, pad = _letterbox(Image.new('RGB', (crop_w, crop_h)))
        rows = []
        for i, score in enumerate((0.9, 0.7, 0.1)):
            x = 20 * i * scale + pad[0]
            rows.append((x, pad[1] + 10, x + 10 * scale, pad[1] + 20 * scale, score))
        got = _decode_yolo_output_multi(
            _fused(*rows), scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h
        )
        assert sorted(round(c.score, 2) for c in got) == [0.7, 0.9]
        assert got[0].bbox_norm[2] - got[0].bbox_norm[0] == pytest.approx(0.05, abs=0.01)

    def test_padding_rows_are_dropped(self) -> None:
        # A positive score on a degenerate box, and a real box with score 0: both padding.
        raw = _fused((100, 100, 200, 200, 0.9), (50, 50, 50, 60, 0.95), (10, 10, 90, 90, 0.0))
        got = _decode_yolo_output_multi(
            raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640, confidence_floor=0.0
        )
        assert [round(c.score, 2) for c in got] == [0.9]
        best = _decode_yolo_output(
            raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640, confidence_floor=0.0
        )
        assert best is not None
        assert best.score == pytest.approx(0.9)

    def test_a_tensor_that_is_not_n_by_six_is_rejected(self) -> None:
        # The old raw head [5, N]: decoded as nothing, never as garbage rows.
        raw = np.zeros((1, 5, 8), dtype=np.float32)
        raw[0, :, 3] = (320, 320, 64, 64, 0.8)
        assert (
            _decode_yolo_output_multi(raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640) == []
        )
        assert _decode_yolo_output(raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640) is None
