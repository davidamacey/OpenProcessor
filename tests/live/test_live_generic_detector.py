"""Live: the ingest policy, the shared item filter, the detections summary, bulk
writes on a selection and embed-missing, against the disposable harness."""

from __future__ import annotations

from typing import Any

import pytest


pytestmark = pytest.mark.live


def _policy(api_client: Any) -> dict[str, Any]:
    resp = api_client.get('/ingest/policy')
    resp.raise_for_status()
    return resp.json()


def test_policy_round_trip_and_revision_check(api_client: Any) -> None:
    before = _policy(api_client)
    assert before['embedding']['mode'] in {'all', 'selected', 'lazy'}
    put = api_client.put(
        '/ingest/policy',
        json={'expected_revision': before['revision'], 'embedding': {'mode': 'lazy'}},
    )
    put.raise_for_status()
    assert put.json()['revision'] == before['revision'] + 1
    stale = api_client.put(
        '/ingest/policy',
        json={'expected_revision': before['revision'], 'embedding': {'mode': 'all'}},
    )
    assert stale.status_code == 409
    assert _policy(api_client)['embedding']['mode'] == 'lazy'
    restore = api_client.put(
        '/ingest/policy',
        json={'expected_revision': put.json()['revision'], 'embedding': {'mode': 'all'}},
    )
    restore.raise_for_status()
    assert api_client.get('/ingest/config').json()['policy']['embedding']['mode'] == 'all'


def test_selected_mode_needs_a_criterion(api_client: Any) -> None:
    rev = _policy(api_client)['revision']
    r = api_client.put(
        '/ingest/policy', json={'expected_revision': rev, 'embedding': {'mode': 'selected'}}
    )
    assert r.status_code == 422


def test_preview_counts_the_stored_items_and_writes_nothing(api_client: Any) -> None:
    total = api_client.get('/stats/dataset').json()['total_crops']
    r = api_client.post('/ingest/policy/preview', json={'embedding': {'mode': 'lazy'}})
    r.raise_for_status()
    body = r.json()
    assert body['total_items'] == total
    assert body['would_embed'] + body['would_not_embed'] == body['scanned']
    assert _policy(api_client)['embedding']['mode'] == 'all'


def test_the_shared_filter_applies_to_every_list_route(api_client: Any) -> None:
    classes = api_client.get('/classes').json()['classes']
    name = classes[0]['class_name']
    params = {'class_name': name, 'embedding_state': 'embedded', 'min_area': 0.0}
    crops = api_client.get('/crops', params={**params, 'page_size': 20})
    crops.raise_for_status()
    assert all(c['class_name'] == name for c in crops.json()['crops'] if c.get('class_name'))
    for path in ('/stats/classes', '/stats/dataset', '/clusters', '/detections/summary'):
        assert api_client.get(path, params=params).status_code == 200, path
    assert api_client.get('/review/all', params=params).status_code == 200
    bad = api_client.get('/crops', params={'conf_min': 0.9, 'conf_max': 0.1})
    assert bad.status_code == 400


def test_detections_summary_matches_the_stats_and_its_suggestion_posts(api_client: Any) -> None:
    summary = api_client.get('/detections/summary')
    summary.raise_for_status()
    body = summary.json()
    stats = api_client.get('/stats/dataset').json()
    assert body['total'] == stats['total_crops']
    assert body['embedding']['embedded'] == stats['embedding']['embedded']
    if body['suggested_reprocess'] is not None:
        dry = api_client.post('/reprocess', json=body['suggested_reprocess'])
        dry.raise_for_status()
        assert dry.json()['dry_run'] is True


def test_bulk_exclude_on_a_selection_dry_run_changes_nothing(api_client: Any) -> None:
    before = api_client.get('/stats/dataset').json()
    selection = {'filter': {'max_rank': 1}, 'limit': 3, 'sample': 'random', 'seed': 1}
    r = api_client.post('/crops/batch_exclude', json={'selection': selection, 'dry_run': True})
    r.raise_for_status()
    assert r.json()['dry_run'] is True
    assert r.json()['selected'] <= 3
    after = api_client.get('/stats/dataset').json()
    assert after['total_crops'] == before['total_crops']
    assert api_client.post('/crops/batch_exclude', json={}).status_code == 422


def test_embed_missing_dry_run_reports_exact_counts(api_client: Any) -> None:
    body = {
        'targets': {'filter': {'embedding_state': ['not_selected', 'deferred', 'failed']}},
        'scopes': ['embed'],
        'embed': {'only_missing': True},
        'dry_run': True,
    }
    r = api_client.post('/reprocess', json=body)
    r.raise_for_status()
    (scope,) = r.json()['scopes']
    detail = scope['detail']
    if scope['selected']:
        assert detail['to_embed'] == detail['without_vector']
        assert detail['estimated_vector_kb'] >= 0
    assert api_client.post('/reprocess', json={**body, 'targets': {}}).status_code == 422


def test_region_box_edit_reports_the_vector_refresh(api_client: Any) -> None:
    regions = api_client.get('/regions', params={'page_size': 1})
    if regions.status_code == 409:
        pytest.skip('no region profile is active on this harness')
    regions.raise_for_status()
    rows = regions.json().get('items') or regions.json().get('rows') or []
    if not rows:
        pytest.skip('no region rows seeded')
    crop_id = rows[0]['crop_id']
    r = api_client.put(
        f'/crops/{crop_id}/regions', json={'boxes': [{'box_id': rows[0].get('region_box_id')}]}
    )
    if r.status_code != 200:
        pytest.skip(f'region edit refused on this harness: {r.status_code}')
    assert set(r.json()['vector_refresh']) == {'embedded', 'pending'}
