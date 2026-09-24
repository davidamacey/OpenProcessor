"""Unit tests for ``src.services.detection.cascade_detect``.

The Triton client is mocked end-to-end — we never hit a real detector
model.

Test coverage:
* Decoder produces sane crop-frame bboxes (round-trip through letterbox).
* Confidence floor filters low-score detections.
* ``detect_batch`` aligns results 1:1 with inputs and survives partial
  failures.
* ``RegionDetector.detect`` returns ``None`` for empty / corrupt JPEG
  bytes rather than raising.
* Triton errors get logged and degrade gracefully (return ``None``).
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from src.services.detection.cascade_detect import (
    REFERENCE_LICENSE_PLATE_PROFILE,
    RegionCandidate,
    RegionDetector,
    _decode_yolo_output,
    _letterbox,
    crop_norm_to_source_norm,
)


# =============================================================================
# Fixtures and helpers
# =============================================================================


def _make_jpeg(
    width: int = 320, height: int = 240, color: tuple[int, int, int] = (50, 80, 120)
) -> bytes:
    """Synthesize a small RGB JPEG suitable for the region detector input path."""
    buf = io.BytesIO()
    Image.new('RGB', (width, height), color).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def _make_raw_output(
    *,
    cx_norm: float,
    cy_norm: float,
    w_norm: float,
    h_norm: float,
    score: float,
    n_anchors: int = 8400,
    pixel_space: bool = True,
    extra: tuple[float, float, float, float, float] | None = None,
) -> np.ndarray:
    """Build a synthetic ``[1, 5, N]`` raw YOLOv11-shaped output.

    ``cx_norm`` etc are in fraction-of-network-input (so ``cx_norm=0.5``
    is the center of the 640x640 letterbox). When ``pixel_space=True``
    we multiply by 640 to match the actual model export. When False we
    leave normalized to exercise the decoder's "looks normalized?"
    fallback branch.
    """
    arr = np.zeros((5, n_anchors), dtype=np.float32)
    multiplier = float(REFERENCE_LICENSE_PLATE_PROFILE.input_size) if pixel_space else 1.0
    # Anchor 0 = our planted detection at the requested score.
    arr[0, 0] = cx_norm * multiplier
    arr[1, 0] = cy_norm * multiplier
    arr[2, 0] = w_norm * multiplier
    arr[3, 0] = h_norm * multiplier
    arr[4, 0] = score
    # Anchor 1 = a low-score distractor so argmax has work to do.
    if n_anchors > 1:
        arr[0, 1] = 0.5 * multiplier
        arr[1, 1] = 0.5 * multiplier
        arr[2, 1] = 0.05 * multiplier
        arr[3, 1] = 0.02 * multiplier
        arr[4, 1] = 0.01
    if extra is not None and n_anchors > 2:
        arr[0, 2] = extra[0] * multiplier
        arr[1, 2] = extra[1] * multiplier
        arr[2, 2] = extra[2] * multiplier
        arr[3, 2] = extra[3] * multiplier
        arr[4, 2] = extra[4]
    return arr[None, ...]  # add batch dim → [1, 5, N]


class _FakeInferResult:
    """Mimic the subset of ``InferResult`` the detector uses."""

    def __init__(self, output: np.ndarray) -> None:
        self._output = output

    def as_numpy(self, name: str) -> np.ndarray:  # noqa: ARG002 — single output
        return self._output


def _make_mock_pool(
    raw_output: np.ndarray | None = None, *, raises: BaseException | None = None
) -> MagicMock:
    """Build an :class:`AsyncTritonPool` mock that returns one canned output."""
    pool = MagicMock()
    if raises is not None:
        pool.infer = AsyncMock(side_effect=raises)
    else:
        pool.infer = AsyncMock(return_value=_FakeInferResult(raw_output))
    return pool


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


# =============================================================================
# RegionDetector — single-crop API
# =============================================================================


class TestRegionDetectorSingle:
    @pytest.mark.asyncio
    async def test_detect_returns_candidate(self) -> None:
        raw = _make_raw_output(cx_norm=0.5, cy_norm=0.5, w_norm=0.1, h_norm=0.05, score=0.85)
        pool = _make_mock_pool(raw)
        detector = RegionDetector(pool)

        result = await detector.detect(_make_jpeg())
        assert result is not None
        assert isinstance(result, RegionCandidate)
        assert result.source == 'lpr_nanov11_640'
        assert result.rectangularity is None
        assert 0.0 <= result.bbox_norm[0] < result.bbox_norm[2] <= 1.0
        assert 0.0 <= result.bbox_norm[1] < result.bbox_norm[3] <= 1.0

    @pytest.mark.asyncio
    async def test_detect_below_floor_returns_none(self) -> None:
        raw = _make_raw_output(cx_norm=0.5, cy_norm=0.5, w_norm=0.1, h_norm=0.05, score=0.05)
        pool = _make_mock_pool(raw)
        detector = RegionDetector(pool)

        assert await detector.detect(_make_jpeg()) is None

    @pytest.mark.asyncio
    async def test_detect_empty_bytes_returns_none(self) -> None:
        pool = _make_mock_pool(np.zeros((1, 5, 10), dtype=np.float32))
        detector = RegionDetector(pool)

        assert await detector.detect(b'') is None
        # Also: corrupt bytes don't raise, just degrade.
        assert await detector.detect(b'\x00\x01\x02not-a-jpeg') is None

    @pytest.mark.asyncio
    async def test_detect_swallows_triton_error(self) -> None:
        pool = _make_mock_pool(raises=RuntimeError('triton blew up'))
        detector = RegionDetector(pool)

        # Ingest must not crash on Triton failures.
        assert await detector.detect(_make_jpeg()) is None

    @pytest.mark.asyncio
    async def test_detect_handles_missing_output(self) -> None:
        class _NullResult:
            def as_numpy(self, name: str) -> Any:  # noqa: ARG002
                return None

        pool = MagicMock()
        pool.infer = AsyncMock(return_value=_NullResult())
        detector = RegionDetector(pool)

        assert await detector.detect(_make_jpeg()) is None


# =============================================================================
# RegionDetector — batched API
# =============================================================================


class TestRegionDetectorBatch:
    @pytest.mark.asyncio
    async def test_detect_batch_aligns_results(self) -> None:
        raw = _make_raw_output(cx_norm=0.5, cy_norm=0.5, w_norm=0.1, h_norm=0.05, score=0.7)
        pool = _make_mock_pool(raw)
        detector = RegionDetector(pool)

        crops = [_make_jpeg() for _ in range(5)]
        results = await detector.detect_batch(crops)
        assert len(results) == 5
        for res in results:
            assert isinstance(res, RegionCandidate)

    @pytest.mark.asyncio
    async def test_detect_batch_empty_input(self) -> None:
        pool = _make_mock_pool(np.zeros((1, 5, 10), dtype=np.float32))
        detector = RegionDetector(pool)
        assert await detector.detect_batch([]) == []
        # Triton was never called.
        pool.infer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_detect_batch_skips_corrupt_bytes(self) -> None:
        raw = _make_raw_output(cx_norm=0.5, cy_norm=0.5, w_norm=0.1, h_norm=0.05, score=0.7)
        pool = _make_mock_pool(raw)
        detector = RegionDetector(pool)

        crops: list[bytes] = [_make_jpeg(), b'', b'not-a-jpeg', _make_jpeg()]
        results = await detector.detect_batch(crops)
        assert len(results) == 4
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is None
        assert results[3] is not None

    @pytest.mark.asyncio
    async def test_detect_batch_chunks_large_input(self) -> None:
        """More than ``profile.batch_limit`` crops still all get a result."""
        raw = _make_raw_output(cx_norm=0.5, cy_norm=0.5, w_norm=0.1, h_norm=0.05, score=0.6)
        pool = _make_mock_pool(raw)
        detector = RegionDetector(pool)

        # 40 > 16 (the chunk limit) — exercises multiple chunks.
        crops = [_make_jpeg() for _ in range(40)]
        results = await detector.detect_batch(crops)
        assert len(results) == 40
        assert all(isinstance(r, RegionCandidate) for r in results)
        # Each crop yielded exactly one Triton call (Triton does the
        # actual GPU batching server-side via dynamic batching).
        assert pool.infer.await_count == 40


# =============================================================================
# crop_norm_to_source_norm — coordinate frame transform
# =============================================================================


class TestCropNormToSourceNorm:
    """Re-projects region bboxes from crop frame to source-image frame.

    YOLO training labels need the region's source-image coordinates,
    regardless of which crop the detector was actually run against.
    """

    def test_centered_vehicle_centered_plate(self) -> None:
        """A region at the center of a centered crop lands at the image center."""
        # Parent occupies the middle 50% of the source image.
        vehicle = (0.25, 0.25, 0.75, 0.75)
        # Region is dead center of the crop.
        plate_in_crop = (0.4, 0.4, 0.6, 0.6)
        sx1, sy1, sx2, sy2 = crop_norm_to_source_norm(plate_in_crop, vehicle)
        # Crop is 0.5 wide, region is 0.2 wide in crop → 0.1 wide in source.
        # Region starts at 0.4 of crop → 0.4*0.5 = 0.2 from crop start →
        # 0.25 + 0.2 = 0.45 in source.
        assert sx1 == pytest.approx(0.45, abs=1e-9)
        assert sy1 == pytest.approx(0.45, abs=1e-9)
        assert sx2 == pytest.approx(0.55, abs=1e-9)
        assert sy2 == pytest.approx(0.55, abs=1e-9)

    def test_full_crop_plate_returns_vehicle_box(self) -> None:
        """A region that fills the entire crop is the parent box."""
        vehicle = (0.10, 0.20, 0.40, 0.60)
        plate_in_crop = (0.0, 0.0, 1.0, 1.0)
        result = crop_norm_to_source_norm(plate_in_crop, vehicle)
        assert result == pytest.approx(vehicle)

    def test_offset_vehicle_offset_plate(self) -> None:
        """Generic case: parent in lower-right, region in upper-left of crop."""
        vehicle = (0.5, 0.5, 1.0, 1.0)  # bottom-right quadrant
        plate_in_crop = (0.1, 0.0, 0.3, 0.2)  # upper-left of the crop
        sx1, sy1, sx2, sy2 = crop_norm_to_source_norm(plate_in_crop, vehicle)
        # Parent is 0.5 wide x 0.5 tall starting at (0.5, 0.5).
        assert sx1 == pytest.approx(0.5 + 0.1 * 0.5, abs=1e-9)
        assert sy1 == pytest.approx(0.5 + 0.0 * 0.5, abs=1e-9)
        assert sx2 == pytest.approx(0.5 + 0.3 * 0.5, abs=1e-9)
        assert sy2 == pytest.approx(0.5 + 0.2 * 0.5, abs=1e-9)

    def test_clamps_to_unit_interval(self) -> None:
        """Numerical noise that pushes a coord just past 1.0 is clipped."""
        vehicle = (0.0, 0.0, 1.0, 1.0)
        plate_in_crop = (-0.05, -0.05, 1.05, 1.05)
        sx1, sy1, sx2, sy2 = crop_norm_to_source_norm(plate_in_crop, vehicle)
        assert sx1 == 0.0
        assert sy1 == 0.0
        assert sx2 == 1.0
        assert sy2 == 1.0

    def test_zero_width_vehicle_collapses_plate(self) -> None:
        """Degenerate parent (width=0) maps any region to the parent's left edge.

        Realistically the ingest will reject a zero-width parent bbox
        before the detector ever sees it, but the helper should not
        divide-by-zero or raise.
        """
        vehicle = (0.5, 0.2, 0.5, 0.8)
        plate_in_crop = (0.1, 0.2, 0.9, 0.8)
        sx1, _, sx2, _ = crop_norm_to_source_norm(plate_in_crop, vehicle)
        # Width is 0 → region has no horizontal extent in source.
        assert sx1 == pytest.approx(0.5)
        assert sx2 == pytest.approx(0.5)


# =============================================================================
# PaddleOcrTextRecognizer (ocr_pipeline BLS wrapper)
# =============================================================================


def _ocr_pipeline_mock_result(
    *,
    n: int,
    boxes: list[list[float]],
    texts: list[str],
    det_scores: list[float],
    rec_scores: list[float],
) -> Any:
    """Build a fake Triton result matching the ocr_pipeline output spec.

    The wrapper pads to 128 entries the same way the BLS does; only the
    first ``n`` slots carry real data.
    """
    pad_box = [0.0, 0.0, 0.0, 0.0]
    boxes_full = boxes + [pad_box] * (128 - len(boxes))
    texts_full = texts + [''] * (128 - len(texts))
    det_full = det_scores + [0.0] * (128 - len(det_scores))
    rec_full = rec_scores + [0.0] * (128 - len(rec_scores))

    table = {
        'num_texts': np.asarray([n], dtype=np.int32),
        'text_boxes_normalized': np.asarray(boxes_full, dtype=np.float32),
        'texts': np.asarray([t.encode('utf-8') for t in texts_full], dtype=object),
        'text_scores': np.asarray(det_full, dtype=np.float32),
        'rec_scores': np.asarray(rec_full, dtype=np.float32),
    }
    m = MagicMock()
    m.as_numpy = lambda name: table.get(name)
    return m


def _jpeg_bytes(w: int = 64, h: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (w, h), (128, 128, 128)).save(buf, 'JPEG')
    return buf.getvalue()


class TestPaddleOcrTextRecognizer:
    @pytest.mark.asyncio
    async def test_canonicalizes_and_filters(self) -> None:
        from src.services.detection.cascade_detect import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=3,
                boxes=[
                    [0.10, 0.40, 0.50, 0.50],  # plate-shaped, plate text
                    [0.05, 0.05, 0.95, 0.10],  # bumper sticker, plate-shaped, junk
                    [0.20, 0.60, 0.22, 0.62],  # tiny, non-plate-shape
                ],
                texts=['Abc-1234', 'COOLBUMPER!!!', '??'],
                det_scores=[0.90, 0.85, 0.70],
                rec_scores=[0.95, 0.80, 0.60],
            )
        )
        rec = PaddleOcrTextRecognizer(pool)
        regions = await rec.detect_regions(_jpeg_bytes())
        # Third entry's '??' canonicalizes to an empty string (no
        # alphanumerics survive the filter), so it's dropped — left
        # with the plate row and the bumper sticker.
        assert len(regions) == 2
        # Canonicalization: uppercased, single-spaced, kept dash.
        assert regions[0].text == 'ABC-1234'
        # text_raw preserves the exact pipeline string.
        assert regions[0].text_raw == 'Abc-1234'

    @pytest.mark.asyncio
    async def test_pick_best_plate_region(self) -> None:
        from src.services.detection.cascade_detect import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=2,
                boxes=[
                    [0.10, 0.40, 0.30, 0.50],  # plate-shaped, small
                    [0.10, 0.40, 0.50, 0.50],  # plate-shaped, larger
                ],
                texts=['XY-12', 'ABC-1234'],
                det_scores=[0.90, 0.90],
                rec_scores=[0.99, 0.80],
            )
        )
        rec = PaddleOcrTextRecognizer(pool)
        regions = await rec.detect_regions(_jpeg_bytes())
        pick = rec.pick_best_plate_region(regions)
        # Larger plate-shaped region wins despite lower rec_score.
        assert pick is not None
        assert pick.text == 'ABC-1234'

    @pytest.mark.asyncio
    async def test_f4_rejects_letters_only_text(self) -> None:
        """Bumper-sticker / dealer-frame text without digits must not promote."""
        from src.services.detection.cascade_detect import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=3,
                boxes=[
                    [0.10, 0.40, 0.50, 0.50],  # plate-shaped, all-letters
                    [0.10, 0.55, 0.50, 0.65],  # plate-shaped, all-letters
                    [0.10, 0.70, 0.45, 0.78],  # plate-shaped, all-letters
                ],
                texts=['FORD', 'DEALER', 'TURBO'],
                det_scores=[0.95, 0.95, 0.95],
                rec_scores=[0.98, 0.95, 0.92],
            )
        )
        rec = PaddleOcrTextRecognizer(pool)
        regions = await rec.detect_regions(_jpeg_bytes())
        # Loose plate-shape filter accepts them, but the stricter
        # is_plate_text_candidate rejects all three because they lack digits.
        assert all(r.is_plate_shaped for r in regions)
        assert all(not r.is_plate_text_candidate for r in regions)
        assert rec.pick_best_plate_region(regions) is None

    @pytest.mark.asyncio
    async def test_f4_rejects_low_rec_score(self) -> None:
        """Even a perfect-looking plate text below the rec-score floor is rejected."""
        from src.services.detection.cascade_detect import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=1,
                boxes=[[0.10, 0.40, 0.50, 0.50]],
                texts=['ABC1234'],
                det_scores=[0.95],
                rec_scores=[0.55],  # below profile.text_hint_rec_floor=0.70
            )
        )
        rec = PaddleOcrTextRecognizer(pool)
        regions = await rec.detect_regions(_jpeg_bytes())
        assert len(regions) == 1
        assert not regions[0].is_plate_text_candidate
        assert rec.pick_best_plate_region(regions) is None

    @pytest.mark.asyncio
    async def test_f4_accepts_vanity_plate(self) -> None:
        """Vanity plates with letter+digit mix promote (e.g. 'LUV2DRV')."""
        from src.services.detection.cascade_detect import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=1,
                boxes=[[0.10, 0.40, 0.45, 0.50]],  # ar=3.5
                texts=['LUV2DRV'],
                det_scores=[0.92],
                rec_scores=[0.88],
            )
        )
        rec = PaddleOcrTextRecognizer(pool)
        regions = await rec.detect_regions(_jpeg_bytes())
        assert regions[0].is_plate_text_candidate
        assert rec.pick_best_plate_region(regions) is regions[0]

    @pytest.mark.asyncio
    async def test_rejects_non_plate_shaped_region(self) -> None:
        from src.services.detection.cascade_detect import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=1,
                # Aspect 1.0 — square, not plate-shaped.
                boxes=[[0.10, 0.40, 0.30, 0.60]],
                texts=['ABC-1234'],
                det_scores=[0.95],
                rec_scores=[0.95],
            )
        )
        rec = PaddleOcrTextRecognizer(pool)
        regions = await rec.detect_regions(_jpeg_bytes())
        assert len(regions) == 1
        assert not regions[0].is_plate_shaped
        assert rec.pick_best_plate_region(regions) is None

    @pytest.mark.asyncio
    async def test_read_plate_region_joins_lines(self) -> None:
        from src.services.detection.cascade_detect import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(
            return_value=_ocr_pipeline_mock_result(
                n=2,
                boxes=[
                    [0.10, 0.40, 0.80, 0.55],  # bottom line
                    [0.10, 0.10, 0.80, 0.30],  # top line
                ],
                texts=['1234', 'MOTO'],
                det_scores=[0.95, 0.95],
                rec_scores=[0.95, 0.90],
            )
        )
        rec = PaddleOcrTextRecognizer(pool)
        result = await rec.read_plate_region(_jpeg_bytes())
        assert result is not None
        text, conf = result
        # Top-then-bottom reading order: MOTO before 1234.
        assert text == 'MOTO 1234'
        assert conf == pytest.approx(0.925, abs=1e-3)

    @pytest.mark.asyncio
    async def test_infer_failure_returns_empty(self) -> None:
        from src.services.detection.cascade_detect import PaddleOcrTextRecognizer

        pool = MagicMock()
        pool.infer = AsyncMock(side_effect=RuntimeError('triton down'))
        rec = PaddleOcrTextRecognizer(pool)
        regions = await rec.detect_regions(_jpeg_bytes())
        assert regions == []


def test_reference_profile_secondary_shape_groups_match_registry_group_names() -> None:
    """The reference profile's secondary-shape groups must be real class
    registry ``group`` values -- a stale name (e.g. a bare 'dirtbikes')
    silently never matches and routes those crops down the wrong path."""
    assert REFERENCE_LICENSE_PLATE_PROFILE.secondary_shape_groups == frozenset(
        {
            'sportbikes',
            'cruisers',
            'touring-adventurebikes',
            'trikes-dirtbikes-motards-scooters-bicycles',
        }
    )
