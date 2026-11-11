"""The area, location and near-duplicate-group predicates of the shared item filter."""

from __future__ import annotations

import pytest

from src.domain.geo_box import GeoBox
from src.domain.location_policy import snap_box, snap_polygon


def test_coarse_policy_rounds_a_box_outward() -> None:
    box = GeoBox(west=8.1234, south=47.1234, east=8.5678, north=47.5678)
    assert snap_box(box, 'store_coarse') == GeoBox(west=8.12, south=47.12, east=8.57, north=47.57)
    assert snap_box(box, 'store') == box


def test_drop_policy_matches_nothing() -> None:
    box = GeoBox(west=8.0, south=47.0, east=9.0, north=48.0)
    assert snap_box(box, 'drop') is None
    assert snap_polygon([(8.0, 47.0), (9.0, 47.0), (8.5, 48.0)], 'drop') is None


def test_coarse_policy_rounds_polygon_vertices() -> None:
    got = snap_polygon([(8.1234, 47.1234), (9.5678, 47.0), (8.5, 48.0)], 'store_coarse')
    assert got == [(8.12, 47.12), (9.57, 47.0), (8.5, 48.0)]


def test_unknown_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match='location_policy'):
        snap_box(GeoBox(west=0, south=0, east=1, north=1), 'bogus')
