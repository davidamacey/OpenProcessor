"""``_decode_yolo_output`` and the confidence-floor filter of the region detector."""

from __future__ import annotations

import numpy as np
import pytest
from _region_profile_fixture import EXAMPLE_LICENSE_PLATE_PROFILE as REFERENCE_LICENSE_PLATE_PROFILE
from PIL import Image

from src.services.detection.cascade_detect.preprocess import _letterbox
from src.services.detection.cascade_detect.region_detector import _decode_yolo_output


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

        # Place a tight 20x10 region-shaped box at the center of the crop
        # (in fraction of network input). Center of crop in network
        # space is (cx, cy) = (320, 320 (160 + 100/2*3.2 = 320)).
        cx_net = 320.0
        cy_net = 320.0
        w_net = 20.0 * scale  # 20 crop-px wide
        h_net = 10.0 * scale  # 10 crop-px tall
        raw = np.zeros((1, 5, 100), dtype=np.float32)
        raw[0, 0, 0] = cx_net
        raw[0, 1, 0] = cy_net
        raw[0, 2, 0] = w_net
        raw[0, 3, 0] = h_net
        raw[0, 4, 0] = 0.92

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

    def test_decoder_clamps_out_of_bounds(self) -> None:
        """A box that pokes past the crop edge gets clamped to [0, 1]."""
        crop_w, crop_h = 100, 100
        img = Image.new('RGB', (crop_w, crop_h))
        _, scale, pad = _letterbox(img)

        # Plant a huge box that's bigger than the network input.
        raw = np.zeros((1, 5, 10), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 10000.0
        raw[0, 3, 0] = 10000.0
        raw[0, 4, 0] = 0.95

        result = _decode_yolo_output(raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h)
        assert result is not None
        assert result.bbox_norm == (0.0, 0.0, 1.0, 1.0)

    def test_decoder_drops_collapsed_box(self) -> None:
        """Boxes that collapse to a point after clamping return ``None``."""
        crop_w, crop_h = 100, 100
        img = Image.new('RGB', (crop_w, crop_h))
        _, scale, pad = _letterbox(img)

        raw = np.zeros((1, 5, 10), dtype=np.float32)
        # Box well outside the crop's letterbox area → after clamp == (0,0,0,0)
        raw[0, 0, 0] = -500.0
        raw[0, 1, 0] = -500.0
        raw[0, 2, 0] = 1.0
        raw[0, 3, 0] = 1.0
        raw[0, 4, 0] = 0.95

        result = _decode_yolo_output(raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h)
        assert result is None

    def test_decoder_accepts_normalized_export(self) -> None:
        """Some exports emit boxes in [0, 1] of network input rather than pixels."""
        crop_w, crop_h = 200, 100
        img = Image.new('RGB', (crop_w, crop_h))
        _, scale, pad = _letterbox(img)

        # Same logical detection, but emitted in normalized 0-1 space.
        raw = np.zeros((1, 5, 10), dtype=np.float32)
        raw[0, 0, 0] = 0.5  # center
        raw[0, 1, 0] = 0.5
        raw[0, 2, 0] = 0.1
        raw[0, 3, 0] = 0.05
        raw[0, 4, 0] = 0.9

        result = _decode_yolo_output(raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h)
        assert result is not None
        assert result.score == pytest.approx(0.9)
        x1, y1, x2, y2 = result.bbox_norm
        assert 0.4 < (x1 + x2) / 2 < 0.6
        assert 0.4 < (y1 + y2) / 2 < 0.6

    def test_decoder_handles_n5_layout(self) -> None:
        """Some exports ship ``[1, N, 5]`` instead of ``[1, 5, N]``."""
        crop_w, crop_h = 100, 100
        img = Image.new('RGB', (crop_w, crop_h))
        _, scale, pad = _letterbox(img)

        # [1, N, 5] layout
        raw = np.zeros((1, 10, 5), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 0, 1] = 320.0
        raw[0, 0, 2] = 32.0
        raw[0, 0, 3] = 16.0
        raw[0, 0, 4] = 0.85

        result = _decode_yolo_output(raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h)
        assert result is not None
        assert result.score == pytest.approx(0.85)


# =============================================================================
# Confidence-floor filter
# =============================================================================


class TestConfidenceFloor:
    def test_drops_below_floor(self) -> None:
        """A detection below ``confidence_floor`` returns ``None``."""
        crop_w, crop_h = 100, 100
        img = Image.new('RGB', (crop_w, crop_h))
        _, scale, pad = _letterbox(img)

        raw = np.zeros((1, 5, 10), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 32.0
        raw[0, 3, 0] = 16.0
        raw[0, 4, 0] = 0.30  # below default floor 0.4

        assert (
            _decode_yolo_output(
                raw,
                scale=scale,
                pad=pad,
                crop_w=crop_w,
                crop_h=crop_h,
                confidence_floor=REFERENCE_LICENSE_PLATE_PROFILE.confidence_floor,
            )
            is None
        )

    def test_keeps_at_floor(self) -> None:
        """Score exactly at the floor counts as a hit (>=, not >)."""
        crop_w, crop_h = 100, 100
        img = Image.new('RGB', (crop_w, crop_h))
        _, scale, pad = _letterbox(img)

        raw = np.zeros((1, 5, 10), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 32.0
        raw[0, 3, 0] = 16.0
        raw[0, 4, 0] = REFERENCE_LICENSE_PLATE_PROFILE.confidence_floor

        result = _decode_yolo_output(
            raw,
            scale=scale,
            pad=pad,
            crop_w=crop_w,
            crop_h=crop_h,
            confidence_floor=REFERENCE_LICENSE_PLATE_PROFILE.confidence_floor,
        )
        assert result is not None

    def test_custom_floor_overrides_default(self) -> None:
        """Caller-supplied floor wins over the default."""
        crop_w, crop_h = 100, 100
        img = Image.new('RGB', (crop_w, crop_h))
        _, scale, pad = _letterbox(img)

        raw = np.zeros((1, 5, 10), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 32.0
        raw[0, 3, 0] = 16.0
        raw[0, 4, 0] = 0.55

        # Strict floor at 0.6 → drop.
        assert (
            _decode_yolo_output(
                raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h, confidence_floor=0.6
            )
            is None
        )
        # Looser floor at 0.5 → keep.
        result = _decode_yolo_output(
            raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h, confidence_floor=0.5
        )
        assert result is not None
        assert result.score == pytest.approx(0.55)
