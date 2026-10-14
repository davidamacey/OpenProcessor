"""``RegionDetector.detect`` over a mocked Triton pool and ``crop_norm_to_source_norm``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from _region_profile_fixture import (
    EXAMPLE_LICENSE_PLATE_PROFILE as REFERENCE_LICENSE_PLATE_PROFILE,
    NEUTRAL_REGION_PROFILE,
)

from curation.cascade_detect_helpers import _make_jpeg, _make_mock_pool, _make_raw_output
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.detection.cascade_detect.region_detector import RegionDetector
from src.services.detection.cascade_detect.sanity import crop_norm_to_source_norm


# =============================================================================
# RegionDetector — single-crop API
# =============================================================================


class TestRegionDetectorSingle:
    @pytest.mark.asyncio
    async def test_detect_returns_candidate(self) -> None:
        raw = _make_raw_output(cx_norm=0.5, cy_norm=0.5, w_norm=0.1, h_norm=0.05, score=0.85)
        pool = _make_mock_pool(raw)
        detector = RegionDetector(pool, NEUTRAL_REGION_PROFILE)

        result = await detector.detect(_make_jpeg())
        assert result is not None
        assert isinstance(result, RegionCandidate)
        assert result.source == NEUTRAL_REGION_PROFILE.detector_model
        assert result.rectangularity is None
        assert 0.0 <= result.bbox_norm[0] < result.bbox_norm[2] <= 1.0
        assert 0.0 <= result.bbox_norm[1] < result.bbox_norm[3] <= 1.0

    @pytest.mark.asyncio
    async def test_detect_below_floor_returns_none(self) -> None:
        raw = _make_raw_output(cx_norm=0.5, cy_norm=0.5, w_norm=0.1, h_norm=0.05, score=0.05)
        pool = _make_mock_pool(raw)
        detector = RegionDetector(pool, NEUTRAL_REGION_PROFILE)

        assert await detector.detect(_make_jpeg()) is None

    @pytest.mark.asyncio
    async def test_detect_empty_bytes_returns_none(self) -> None:
        pool = _make_mock_pool(np.zeros((1, 5, 10), dtype=np.float32))
        detector = RegionDetector(pool, NEUTRAL_REGION_PROFILE)

        assert await detector.detect(b'') is None
        # Also: corrupt bytes don't raise, just degrade.
        assert await detector.detect(b'\x00\x01\x02not-a-jpeg') is None

    @pytest.mark.asyncio
    async def test_detect_swallows_triton_error(self) -> None:
        pool = _make_mock_pool(raises=RuntimeError('triton blew up'))
        detector = RegionDetector(pool, NEUTRAL_REGION_PROFILE)

        # Ingest must not crash on Triton failures.
        assert await detector.detect(_make_jpeg()) is None

    @pytest.mark.asyncio
    async def test_detect_handles_missing_output(self) -> None:
        class _NullResult:
            def as_numpy(self, name: str) -> Any:  # noqa: ARG002
                return None

        pool = MagicMock()
        pool.infer = AsyncMock(return_value=_NullResult())
        detector = RegionDetector(pool, NEUTRAL_REGION_PROFILE)

        assert await detector.detect(_make_jpeg()) is None


# =============================================================================
# RegionDetector — batched API
# =============================================================================


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
        region_in_crop = (0.4, 0.4, 0.6, 0.6)
        sx1, sy1, sx2, sy2 = crop_norm_to_source_norm(region_in_crop, vehicle)
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
        region_in_crop = (0.0, 0.0, 1.0, 1.0)
        result = crop_norm_to_source_norm(region_in_crop, vehicle)
        assert result == pytest.approx(vehicle)

    def test_offset_vehicle_offset_plate(self) -> None:
        """Generic case: parent in lower-right, region in upper-left of crop."""
        vehicle = (0.5, 0.5, 1.0, 1.0)  # bottom-right quadrant
        region_in_crop = (0.1, 0.0, 0.3, 0.2)  # upper-left of the crop
        sx1, sy1, sx2, sy2 = crop_norm_to_source_norm(region_in_crop, vehicle)
        # Parent is 0.5 wide x 0.5 tall starting at (0.5, 0.5).
        assert sx1 == pytest.approx(0.5 + 0.1 * 0.5, abs=1e-9)
        assert sy1 == pytest.approx(0.5 + 0.0 * 0.5, abs=1e-9)
        assert sx2 == pytest.approx(0.5 + 0.3 * 0.5, abs=1e-9)
        assert sy2 == pytest.approx(0.5 + 0.2 * 0.5, abs=1e-9)

    def test_clamps_to_unit_interval(self) -> None:
        """Numerical noise that pushes a coord just past 1.0 is clipped."""
        vehicle = (0.0, 0.0, 1.0, 1.0)
        region_in_crop = (-0.05, -0.05, 1.05, 1.05)
        sx1, sy1, sx2, sy2 = crop_norm_to_source_norm(region_in_crop, vehicle)
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
        region_in_crop = (0.1, 0.2, 0.9, 0.8)
        sx1, _, sx2, _ = crop_norm_to_source_norm(region_in_crop, vehicle)
        # Width is 0 → region has no horizontal extent in source.
        assert sx1 == pytest.approx(0.5)
        assert sx2 == pytest.approx(0.5)


def test_reference_profile_secondary_shape_groups_empty_by_default() -> None:
    """The shipped example profile leaves ``secondary_shape_groups`` empty
    (G-20): a private deployment's motorcycle-subtype taxonomy isn't
    meaningful to a new user and doesn't match any group in the public
    class registry example. Empty routes nothing to the secondary-shape
    path (see ``scripts/curation/worker/state.py::_is_secondary_shape``);
    a deployment opts in by populating it with its own registry ``group``
    values -- a stale name would otherwise silently never match and route
    those crops down the wrong path."""
    assert REFERENCE_LICENSE_PLATE_PROFILE.secondary_shape_groups == frozenset()
