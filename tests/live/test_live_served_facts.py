"""Live: facts the backend serves so the frontend computes nothing."""

from __future__ import annotations

from typing import Any

import pytest


pytestmark = pytest.mark.live


def test_export_status_serves_readiness(api_client: Any) -> None:
    body = api_client.get('/export/status').json()
    assert isinstance(body['can_export'], bool) or body['can_export'] is None
    assert isinstance(body['blocking_reasons'], list)
    if body['can_export'] is False:
        assert body['blocking_reasons']


def test_methods_serve_dedup_range_and_diverse_limits(api_client: Any) -> None:
    entries = {s['id']: s for s in api_client.get('/methods').json()['strategies']}
    for export in ('yolo', 'single_class'):
        e = entries[export]
        assert e['dedup_threshold_min'] <= e['dedup_threshold_default'] <= e['dedup_threshold_max']
    assert entries['diverse']['max_k'] >= 1
    assert entries['diverse']['select_max_k'] >= entries['diverse']['max_k']


def test_settings_serve_monitoring_links(api_client: Any) -> None:
    links = api_client.get('/settings').json()['monitoring_links']
    assert set(links) == {'grafana', 'prometheus', 'opensearch_dashboards'}
    assert all(v is None or v.startswith('http') for v in links.values())


def test_viz_projection_distinguishes_not_built_from_unavailable(api_client: Any) -> None:
    r = api_client.get('/viz/projection')
    if r.status_code == 200:
        assert 'points' in r.json()
    elif r.status_code == 400:
        pytest.skip('projection disabled (OP_VIZ_PROJECTION_ENABLED unset)')
    else:
        assert (r.status_code, r.json()['error']) in {
            (404, 'projection_not_built'),
            (503, 'projection_unavailable'),
        }


def test_region_rows_carry_unique_row_keys(api_client: Any) -> None:
    r = api_client.get('/regions', params={'page_size': 100})
    assert r.status_code == 200, r.text
    keys = [row['row_key'] for row in r.json()['items']]
    assert all(keys)
    assert len(keys) == len(set(keys))
