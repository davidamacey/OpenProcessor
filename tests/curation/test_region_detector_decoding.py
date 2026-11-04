"""``_decode_yolo_output`` and the confidence-floor filter of the region detector."""

from __future__ import annotations

import numpy as np
import pytest
from _region_profile_fixture import EXAMPLE_LICENSE_PLATE_PROFILE as REFERENCE_LICENSE_PLATE_PROFILE
from PIL import Image

from src.services.detection.cascade_detect.preprocess import _letterbox
from src.services.detection.cascade_detect.region_detector import (
    _decode_yolo_output,
    _decode_yolo_output_multi,
    head_layout,
)


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

        result = _decode_yolo_output(
            raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h, layout='anchors'
        )
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

        result = _decode_yolo_output(
            raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h, layout='anchors'
        )
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

        result = _decode_yolo_output(
            raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h, layout='anchors'
        )
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

        result = _decode_yolo_output(
            raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h, layout='anchors'
        )
        assert result is not None
        assert result.score == pytest.approx(0.9)
        x1, y1, x2, y2 = result.bbox_norm
        assert 0.4 < (x1 + x2) / 2 < 0.6
        assert 0.4 < (y1 + y2) / 2 < 0.6


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
                layout='anchors',
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
            layout='anchors',
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
                raw,
                scale=scale,
                pad=pad,
                crop_w=crop_w,
                crop_h=crop_h,
                confidence_floor=0.6,
                layout='anchors',
            )
            is None
        )
        # Looser floor at 0.5 → keep.
        result = _decode_yolo_output(
            raw,
            scale=scale,
            pad=pad,
            crop_w=crop_w,
            crop_h=crop_h,
            confidence_floor=0.5,
            layout='anchors',
        )
        assert result is not None
        assert result.score == pytest.approx(0.55)


# =============================================================================
# End2end head ([300, 6]: x1, y1, x2, y2, score, class), #437
# =============================================================================


def _end2end_raw(box_xyxy: tuple[float, float, float, float], score: float) -> np.ndarray:
    raw = np.zeros((1, 300, 6), dtype=np.float32)
    raw[0, 0, :4] = box_xyxy
    raw[0, 0, 4] = score
    raw[0, 0, 5] = 2  # class index is not used by the single-region decoder
    return raw


class TestEnd2EndHead:
    def test_xyxy_box_is_not_read_as_cxcywh(self) -> None:
        crop_w, crop_h = 200, 100
        _, scale, pad = _letterbox(Image.new('RGB', (crop_w, crop_h)))
        # Crop box (50, 25)-(150, 75) in network pixels.
        net = (
            50 * scale + pad[0],
            25 * scale + pad[1],
            150 * scale + pad[0],
            75 * scale + pad[1],
        )
        result = _decode_yolo_output(
            _end2end_raw(net, 0.9),
            scale=scale,
            pad=pad,
            crop_w=crop_w,
            crop_h=crop_h,
            layout='end2end',
        )
        assert result is not None
        assert result.bbox_norm == pytest.approx((0.25, 0.25, 0.75, 0.75), abs=0.01)
        assert result.score == pytest.approx(0.9)

    def test_multi_keeps_every_row_above_floor(self) -> None:
        crop_w, crop_h = 200, 100
        _, scale, pad = _letterbox(Image.new('RGB', (crop_w, crop_h)))
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        for i, score in enumerate((0.9, 0.7, 0.1)):
            x = 20 * i * scale + pad[0]
            raw[0, i] = (x, pad[1] + 10, x + 10 * scale, pad[1] + 20 * scale, score, 0)
        got = _decode_yolo_output_multi(
            raw, scale=scale, pad=pad, crop_w=crop_w, crop_h=crop_h, layout='end2end'
        )
        assert sorted(round(c.score, 2) for c in got) == [0.7, 0.9]
        assert got[0].bbox_norm[2] - got[0].bbox_norm[0] == pytest.approx(0.05, abs=0.01)

    def test_five_row_head_is_unchanged(self) -> None:
        raw = np.zeros((1, 5, 8), dtype=np.float32)
        raw[0, :, 3] = (320, 320, 64, 64, 0.8)
        got = _decode_yolo_output_multi(
            raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640, layout='anchors'
        )
        assert len(got) == 1
        assert got[0].bbox_norm == pytest.approx((0.45, 0.45, 0.55, 0.55))


class TestLayoutFromDeclaredOutput:
    """The head is chosen from the model's DECLARED output0 shape, not guessed from the array."""

    @pytest.mark.parametrize(
        ('declared', 'expected'),
        [
            ((-1, 5, -1), 'anchors'),
            ((1, 5, 8400), 'anchors'),
            ((-1, 300, 6), 'end2end'),
            ((-1, -1, 6), 'end2end'),
            ((300, 6), 'end2end'),
            ((-1, 6, -1), None),
            ((-1, 6, 300), None),
            ((-1, 300, 5), None),
            ((-1, 5, 5), None),
            ((6,), None),
            ((), None),
        ],
    )
    def test_head_layout(self, declared: tuple[int, ...], expected: str | None) -> None:
        assert head_layout(declared) == expected

    def test_array_that_contradicts_the_declared_layout_is_rejected(self) -> None:
        # Declared end2end [N, 6] but the runtime tensor is transposed [6, N]: the old shape
        # guess decoded this as anchors garbage; it must yield nothing.
        raw = np.zeros((1, 6, 300), dtype=np.float32)
        raw[0, :, 0] = (100, 100, 200, 200, 0.9, 0)
        assert (
            _decode_yolo_output_multi(
                raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640, layout='end2end'
            )
            == []
        )
        assert (
            _decode_yolo_output(
                raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640, layout='end2end'
            )
            is None
        )

    def test_six_anchor_head_is_not_mistaken_for_end2end(self) -> None:
        # N == 6 anchors: [1, 5, 6] must stay an anchors head.
        raw = np.zeros((1, 5, 6), dtype=np.float32)
        raw[0, :, 2] = (320, 320, 64, 64, 0.8)
        got = _decode_yolo_output_multi(
            raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640, layout='anchors'
        )
        assert len(got) == 1

    def test_end2end_padding_rows_are_dropped(self) -> None:
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = (100, 100, 200, 200, 0.9, 0)
        # Positive score but a degenerate box, and a real box with score 0: both padding.
        raw[0, 1] = (50, 50, 50, 60, 0.95, 0)
        raw[0, 2] = (10, 10, 90, 90, 0.0, 0)
        got = _decode_yolo_output_multi(
            raw,
            scale=1.0,
            pad=(0.0, 0.0),
            crop_w=640,
            crop_h=640,
            confidence_floor=0.0,
            layout='end2end',
        )
        assert [round(c.score, 2) for c in got] == [0.9]
        best = _decode_yolo_output(
            raw,
            scale=1.0,
            pad=(0.0, 0.0),
            crop_w=640,
            crop_h=640,
            confidence_floor=0.0,
            layout='end2end',
        )
        assert best is not None
        assert best.score == pytest.approx(0.9)

    def test_scale_is_decided_per_tensor_not_per_row(self) -> None:
        # A pixel-space tensor with one tiny box near the origin: that row alone looks
        # "normalized", but the tensor is pixels, so it must not be multiplied by 640.
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = (0.2, 0.2, 1.0, 1.0, 0.9, 0)
        raw[0, 1] = (100, 100, 300, 300, 0.8, 0)
        got = _decode_yolo_output_multi(
            raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640, layout='end2end'
        )
        tiny = min(got, key=lambda c: c.bbox_norm[2])
        assert tiny.bbox_norm[2] == pytest.approx(1.0 / 640, abs=1e-4)

    def test_normalized_tensor_is_scaled_to_pixels(self) -> None:
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = (0.25, 0.25, 0.75, 0.75, 0.9, 0)
        got = _decode_yolo_output_multi(
            raw, scale=1.0, pad=(0.0, 0.0), crop_w=640, crop_h=640, layout='end2end'
        )
        assert got[0].bbox_norm == pytest.approx((0.25, 0.25, 0.75, 0.75))
