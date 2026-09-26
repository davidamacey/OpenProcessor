"""Unit tests for :func:`is_plausible_region_bbox` and provenance helpers.

These tests exercise ``src.services.detection.cascade_detect``.

``is_plausible_region_bbox`` is a GEOMETRY GUARD only — it rejects
non-finite / degenerate boxes but applies no aspect or
region-vs-parent size heuristics (a shape prior built from one
reference domain silently drops legitimate detections from another).
The tests below pin that contract: degenerate geometry rejects; any
well-formed box — including angled, square, wide, and large shapes a
naive shape-prior gate would reject — passes.
"""

from __future__ import annotations

import math

import pytest
from _region_profile_fixture import NEUTRAL_REGION_PROFILE

from src.config import get_region_fields
from src.services.detection.cascade_detect import (
    class_provenance,
    is_plausible_region_bbox,
    region_provenance,
)


class TestIsPlausibleRegionBbox:
    def test_typical_us_plate_passes(self) -> None:
        # 2:1 US plate occupying ~3% of the crop.
        ok, reason = is_plausible_region_bbox((0.4, 0.6, 0.5, 0.65))
        assert ok, reason

    def test_square_motorcycle_plate_passes(self) -> None:
        # Aspect = 1.0 — square moto/EU plate. A naive shape-prior gate
        # rejects this (aspect < 1.2); the geometry-only gate must admit it.
        ok, reason = is_plausible_region_bbox((0.4, 0.6, 0.5, 0.7))
        assert ok, reason

    def test_angled_plate_below_old_floor_passes(self) -> None:
        # Aspect ~0.7 — a 2:1 plate foreshortened on a leaning bike. A
        # naive gate -> aspect_out_of_range; must now pass (the verifier
        # is the arbiter of shape, not this gate).
        ok, reason = is_plausible_region_bbox((0.4, 0.6, 0.47, 0.7))
        assert ok, reason

    def test_wide_eu_plate_passes(self) -> None:
        # Aspect = 10 — wide EU strip / loose box. A naive gate rejects
        # (> 8.0 ceiling); now passes.
        ok, reason = is_plausible_region_bbox((0.1, 0.5, 0.6, 0.55))
        assert ok, reason

    def test_large_plate_relative_to_bike_passes(self) -> None:
        # Region covering >50% width and >15% area of a (tight motorcycle)
        # crop. A naive gate -> too_wide / too_large; now passes.
        ok, reason = is_plausible_region_bbox(
            (0.1, 0.3, 0.85, 0.8),
            parent_in_source=(0.0, 0.0, 1.0, 1.0),
        )
        assert ok, reason

    def test_degenerate_zero_size_rejects(self) -> None:
        ok, reason = is_plausible_region_bbox((0.4, 0.6, 0.4, 0.6))
        assert not ok
        assert reason == 'degenerate_zero_size'

    def test_inverted_coords_rejects(self) -> None:
        ok, reason = is_plausible_region_bbox((0.5, 0.7, 0.4, 0.6))
        assert not ok
        assert reason == 'degenerate_zero_size'

    def test_non_finite_rejects(self) -> None:
        ok, reason = is_plausible_region_bbox((0.4, 0.6, math.nan, 0.65))
        assert not ok
        assert 'non_finite' in reason

    def test_vehicle_bbox_optional(self) -> None:
        # Geometry-only gate passes any well-formed box with or without
        # the parent bbox.
        ok, _ = is_plausible_region_bbox((0.4, 0.6, 0.5, 0.65))
        assert ok
        ok2, _ = is_plausible_region_bbox((0.1, 0.5, 0.7, 0.6))
        assert ok2

    def test_degenerate_parent_bbox_rejects(self) -> None:
        ok, reason = is_plausible_region_bbox(
            (0.4, 0.6, 0.5, 0.65),
            parent_in_source=(0.3, 0.3, 0.3, 0.3),
        )
        assert not ok
        assert reason == 'parent_bbox_degenerate'


class TestRegionProvenance:
    def test_minimal_fields(self) -> None:
        F = get_region_fields()
        doc = region_provenance(
            NEUTRAL_REGION_PROFILE.detector_model,
            NEUTRAL_REGION_PROFILE.detector_version,
        )
        assert doc[F.detector] == NEUTRAL_REGION_PROFILE.detector_model
        assert doc[F.detector_version] == NEUTRAL_REGION_PROFILE.detector_version
        assert doc[F.bbox_frame] == 'source'
        assert F.detected_at in doc
        assert F.verifier not in doc

    def test_with_verifier(self) -> None:
        F = get_region_fields()
        doc = region_provenance(
            NEUTRAL_REGION_PROFILE.human_detector_name,
            NEUTRAL_REGION_PROFILE.human_detector_version,
            verifier=NEUTRAL_REGION_PROFILE.human_detector_name,
            verifier_version=NEUTRAL_REGION_PROFILE.human_detector_version,
        )
        assert doc[F.verifier] == NEUTRAL_REGION_PROFILE.human_detector_name
        assert doc[F.verifier_version] == NEUTRAL_REGION_PROFILE.human_detector_version
        assert F.verified_at in doc


class TestClassProvenance:
    def test_basic(self) -> None:
        doc = class_provenance(
            detector='item_classifier_trt',
            detector_version='1',
            labeler='ingest_classifier',
        )
        assert doc['class_detector'] == 'item_classifier_trt'
        assert doc['class_labeler'] == 'ingest_classifier'
        assert 'class_labeled_at' in doc


@pytest.mark.parametrize(
    'bbox',
    [
        (0.4, 0.6, 0.5, 0.65),  # 2:1 us plate
        (0.45, 0.55, 0.52, 0.6),  # tighter EU-ish plate
        (0.4, 0.6, 0.52, 0.7),  # square moto, aspect = 1.2 boundary
    ],
)
def test_known_good_plates_pass(bbox: tuple[float, float, float, float]) -> None:
    ok, reason = is_plausible_region_bbox(bbox)
    assert ok, f'expected pass, got reject: {reason} for {bbox}'
