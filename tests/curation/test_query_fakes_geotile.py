"""The shared query fake answers the map's geotile_grid aggregation."""

from __future__ import annotations

import pytest

from curation.query_fakes import _aggregate


def test_geotile_grid_groups_points_into_cells_with_a_centroid() -> None:
    docs = [
        {'capture.location': {'lat': 48.8584, 'lon': 2.2945}},
        {'capture.location': {'lat': 48.8606, 'lon': 2.3376}},
        {'capture.location': {'lat': -33.8568, 'lon': 151.2153}},
        {'other': 1},
    ]
    spec = {
        'cells': {
            'geotile_grid': {'field': 'capture.location', 'precision': 3, 'size': 10},
            'aggs': {'centre': {'geo_centroid': {'field': 'capture.location'}}},
        }
    }
    buckets = _aggregate(docs, spec)['cells']['buckets']
    assert [(b['key'], b['doc_count']) for b in buckets] == [('3/4/2', 2), ('3/7/4', 1)]
    assert buckets[0]['centre']['location']['lat'] == pytest.approx(48.8595)


def test_geotile_grid_over_no_points_has_no_buckets() -> None:
    spec = {'cells': {'geotile_grid': {'field': 'capture.location', 'precision': 5, 'size': 3}}}
    assert _aggregate([{'x': 1}], spec) == {'cells': {'buckets': []}}
