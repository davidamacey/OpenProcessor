"""Proof that the detection cascade is actually generic, not just renamed.

Builds a SECOND, non-plate ``DetectionProfile`` (a hypothetical 'box'
region type with a tighter, more square-ish aspect band than the
license-plate default) and drives it through the same cascade
primitives that ``REFERENCE_LICENSE_PLATE_PROFILE`` uses. If a heuristic were still
hardcoded to the license-plate constants, this profile's tighter aspect
band would have no effect and this test would fail to demonstrate a
behavioral difference.

Per ``docs/design/curation_design_rationale.md`` §2.3, this is the real
proof the genericization happened rather than a rename.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from src.config import DetectionProfile
from src.services.detection.cascade_detect import (
    REFERENCE_LICENSE_PLATE_PROFILE,
    OcrRegion,
    RegionCandidate,
    RegionDetector,
    _letterbox,
)


BOX_PROFILE = DetectionProfile(
    name='box',
    detector_model='box_detector_v1',
    detector_version='2',
    input_size=512,
    confidence_floor=0.6,
    batch_limit=8,
    letterbox_fill=(0, 0, 0),
    aspect_min=0.7,
    aspect_max=1.4,
    text_hint_aspect_min=0.8,
    text_hint_aspect_max=1.2,
    text_hint_rec_floor=0.9,
    text_hint_len_min=2,
    text_hint_len_max=6,
)


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), (50, 80, 120)).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


class _FakeInferResult:
    def __init__(self, output: np.ndarray) -> None:
        self._output = output

    def as_numpy(self, name: str) -> np.ndarray:  # noqa: ARG002
        return self._output


def test_profiles_are_distinct_instances() -> None:
    """Sanity: the two profiles really do carry different values."""
    assert BOX_PROFILE.name != REFERENCE_LICENSE_PLATE_PROFILE.name
    assert BOX_PROFILE.detector_model != REFERENCE_LICENSE_PLATE_PROFILE.detector_model
    assert BOX_PROFILE.aspect_min != REFERENCE_LICENSE_PLATE_PROFILE.aspect_min
    assert BOX_PROFILE.confidence_floor != REFERENCE_LICENSE_PLATE_PROFILE.confidence_floor
    assert BOX_PROFILE.input_size != REFERENCE_LICENSE_PLATE_PROFILE.input_size


def test_letterbox_uses_profile_input_size_and_fill() -> None:
    """A non-default profile's ``input_size``/``letterbox_fill`` actually apply."""
    img = Image.new('RGB', (100, 50), (10, 20, 30))
    chw, _scale, _pad = _letterbox(
        img, target=BOX_PROFILE.input_size, fill=BOX_PROFILE.letterbox_fill
    )
    assert chw.shape == (1, 3, BOX_PROFILE.input_size, BOX_PROFILE.input_size)
    # Black letterbox fill (0, 0, 0) → the padded corner pixel is 0.0,
    # not the reference profile's gray (114, 114, 114) / 255.
    assert chw[0, :, 0, 0].max() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_region_detector_uses_profile_model_name_and_floor() -> None:
    """``RegionDetector`` sources its Triton model name + floor from the profile."""
    raw = np.zeros((1, 5, 10), dtype=np.float32)
    raw[0, 0, 0] = BOX_PROFILE.input_size / 2.0
    raw[0, 1, 0] = BOX_PROFILE.input_size / 2.0
    raw[0, 2, 0] = 40.0
    raw[0, 3, 0] = 38.0
    raw[0, 4, 0] = 0.65  # clears the box profile's 0.6 floor

    pool = MagicMock()
    pool.infer = AsyncMock(return_value=_FakeInferResult(raw))
    detector = RegionDetector(pool, BOX_PROFILE)
    assert detector.model_name == 'box_detector_v1'
    assert detector.confidence_floor == pytest.approx(0.6)

    result = await detector.detect(_make_jpeg())
    assert result is not None
    assert isinstance(result, RegionCandidate)
    # Source is stamped with the profile's own model name, not the
    # reference LPR default.
    assert result.source == 'box_detector_v1'

    # A score that clears the LPR default floor (0.4) but not the box
    # profile's stricter 0.6 floor must be dropped — proves the floor
    # really is profile-driven, not the old hardcoded constant.
    raw_low = raw.copy()
    raw_low[0, 4, 0] = 0.45
    pool_low = MagicMock()
    pool_low.infer = AsyncMock(return_value=_FakeInferResult(raw_low))
    detector_low = RegionDetector(pool_low, BOX_PROFILE)
    assert await detector_low.detect(_make_jpeg()) is None


def test_ocr_region_shape_checks_use_the_bound_profile() -> None:
    """``OcrRegion``'s aspect/length checks read the profile it was built with.

    A near-square region (aspect ~1.0) fails the license-plate profile's
    aspect band (1.2-8.0) but passes the box profile's (0.7-1.4) — same
    bbox, different verdict, purely because of the bound profile.
    """
    square_region_default = OcrRegion(
        bbox_norm=(0.1, 0.1, 0.3, 0.3),  # aspect 1.0
        text='AB12',
        text_raw='AB12',
        det_score=0.9,
        rec_score=0.95,
        profile=REFERENCE_LICENSE_PLATE_PROFILE,
    )
    square_region_box = OcrRegion(
        bbox_norm=(0.1, 0.1, 0.3, 0.3),
        text='AB12',
        text_raw='AB12',
        det_score=0.9,
        rec_score=0.95,
        profile=BOX_PROFILE,
    )
    assert not square_region_default.is_plate_shaped
    assert square_region_box.is_plate_shaped


def test_ocr_region_text_candidate_gate_is_profile_scoped() -> None:
    """The stricter text-hint promotion gate also reads the bound profile."""
    region = OcrRegion(
        bbox_norm=(0.1, 0.1, 0.3, 0.3),  # aspect 1.0 — inside box's 0.8-1.2 band
        text='A1',  # length 2 — clears box's text_hint_len_min=2
        text_raw='A1',
        det_score=0.95,
        rec_score=0.95,
        profile=BOX_PROFILE,
    )
    assert region.is_plate_text_candidate

    # Same bbox + text bound to the LPR default profile fails on both
    # counts: aspect 1.0 is outside the default's 1.5-7.0 band, and
    # length 2 is below the default's text_hint_len_min=4. Proves the
    # gate reads the bound profile rather than a module constant.
    default_equivalent = OcrRegion(
        bbox_norm=region.bbox_norm,
        text=region.text,
        text_raw=region.text_raw,
        det_score=region.det_score,
        rec_score=region.rec_score,
        profile=REFERENCE_LICENSE_PLATE_PROFILE,
    )
    assert not default_equivalent.is_plate_text_candidate
