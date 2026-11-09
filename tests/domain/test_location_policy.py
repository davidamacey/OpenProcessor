"""apply_location_policy: the one rule for what a project may keep of a photo's position (CM-4)."""

from __future__ import annotations

import pytest

from src.domain.capture_metadata import CaptureLocation, CaptureMetadata
from src.domain.location_policy import apply_location_policy


def _meta(lat: float = 47.123456, lon: float = 8.987654) -> CaptureMetadata:
    return CaptureMetadata(
        exif_status='ok',
        taken_local='2024-03-05T14:22:11',
        camera_make='Acme',
        location=CaptureLocation(lat, lon, altitude_m=410.0, accuracy_m=4.0, heading_deg=90.0),
        has_location=True,
    )


def test_store_keeps_the_fix_as_parsed() -> None:
    assert apply_location_policy(_meta(), 'store') == _meta()


def test_store_coarse_rounds_to_a_hundredth_of_a_degree_and_drops_the_detail() -> None:
    out = apply_location_policy(_meta(), 'store_coarse')
    assert out.location == CaptureLocation(47.12, 8.99, precision='coarse')
    assert out.has_location
    assert out.camera_make == 'Acme'


def test_drop_keeps_no_coordinate_but_the_fix_is_still_recorded_as_present() -> None:
    out = apply_location_policy(_meta(), 'drop')
    assert out.location is None
    assert out.has_location
    assert out.taken_local == '2024-03-05T14:22:11'


def test_coarse_rounding_to_zero_zero_is_no_location() -> None:
    assert apply_location_policy(_meta(0.003, 0.004), 'store_coarse').location is None


@pytest.mark.parametrize('policy', ['', 'STORE', 'keep', 'none'])
def test_an_unknown_policy_fails_closed(policy: str) -> None:
    with pytest.raises(ValueError, match='location_policy'):
        apply_location_policy(_meta(), policy)


def test_a_photo_without_a_fix_is_unchanged_by_every_policy() -> None:
    bare = CaptureMetadata(exif_status='absent')
    for policy in ('drop', 'store_coarse', 'store'):
        assert apply_location_policy(bare, policy) == bare
