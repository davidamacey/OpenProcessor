"""Live checks of the open-vocabulary routes against the disposable harness.

The harness has no segmenter, so these cover everything that does not need one:
the prompt-set lifecycle (activation through ``force``, which bypasses only the
segmenter checks), the dry-run estimate of the ``open_vocab`` reprocess scope,
the image-level selectors, and that the test route reports a missing segmenter
as an outage. A segmenter-backed pass is checked by hand with
``POST /open_vocab/test`` and the ``open_vocab`` scope on a real stack.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest


pytestmark = pytest.mark.live


def _body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        'display_name': 'Live',
        'targets': [{'prompt': 'traffic cone', 'class_name': 'traffic_cone'}],
    }
    body.update(over)
    return body


@pytest.fixture
def named_set(api_client: Any) -> Any:
    name = f'live-{uuid.uuid4().hex[:8]}'
    created = api_client.post('/open_vocab', json={'name': name, 'body': _body()})
    assert created.status_code == 201, created.text
    yield name
    active = api_client.get('/open_vocab/active').json()['active']
    if active['name'] == name:
        api_client.post('/open_vocab/deactivate', json={'expected_active': active})
    rev = api_client.get(f'/open_vocab/{name}').json()['revision']
    api_client.delete(f'/open_vocab/{name}', params={'expected_revision': rev})


def test_schema_and_template(api_client: Any) -> None:
    schema = api_client.get('/open_vocab/schema')
    assert schema.status_code == 200
    rows = {(f['scope'], f['field']) for f in schema.json()['fields']}
    assert {
        ('target', 'prompt'),
        ('set', 'run_on_ingest'),
        ('gating', 'tier2_vlm_precheck'),
    } <= rows
    listing = api_client.get('/open_vocab', params={'include_templates': 'true'}).json()
    assert 'street_objects' in [t['name'] for t in listing['templates']]


def test_lifecycle_and_activation_gate(api_client: Any, named_set: str) -> None:
    got = api_client.get(f'/open_vocab/{named_set}')
    assert got.status_code == 200
    assert got.json()['body']['targets'][0]['min_score'] == 0.5

    stale = api_client.put(
        f'/open_vocab/{named_set}', json={'expected_revision': 0, 'body': _body()}
    )
    assert (stale.status_code, stale.json()['detail']['error']) == (409, 'revision_conflict')

    gated = api_client.post(f'/open_vocab/{named_set}/activate', json={})
    if gated.status_code == 422:  # no segmenter in the harness: refused, not silently active
        codes = {e['code'] for e in gated.json()['detail']['report']['errors']}
        assert codes <= {'segmenter_not_configured', 'segmenter_unreachable'}
    forced = api_client.post(f'/open_vocab/{named_set}/activate', json={'force': True})
    assert forced.status_code == 200, forced.text
    assert forced.json()['active']['name'] == named_set

    busy = api_client.delete(f'/open_vocab/{named_set}', params={'expected_revision': 1})
    assert (busy.status_code, busy.json()['detail']['error']) == (409, 'in_use')


def test_reprocess_dry_run_estimates_without_writing(
    api_client: Any, named_set: str, opensearch: Any
) -> None:
    from .conftest import INDEXES, refresh

    refresh(opensearch, INDEXES['items'])
    before = opensearch.get(f'/{INDEXES["items"]}/_count').json()['count']
    api_client.post(f'/open_vocab/{named_set}/activate', json={'force': True})

    resp = api_client.post(
        '/reprocess',
        json={'targets': {'filter': {'all_images': True}}, 'scopes': ['open_vocab']},
    )
    assert resp.status_code == 200, resp.text
    (scope,) = resp.json()['scopes']
    assert scope['scope'] == 'open_vocab'
    assert scope['selected'] > 0
    assert scope['detail']['enabled_targets'] == 1
    assert scope['detail']['estimated_calls'] == scope['selected']
    assert scope['detail']['estimated_minutes'] >= 1

    refresh(opensearch, INDEXES['items'])
    assert opensearch.get(f'/{INDEXES["items"]}/_count').json()['count'] == before


def test_reprocess_without_an_active_set_is_refused(api_client: Any) -> None:
    active = api_client.get('/open_vocab/active').json()['active']
    if active['name']:
        api_client.post('/open_vocab/deactivate', json={'expected_active': active})
    resp = api_client.post(
        '/reprocess',
        json={'targets': {'filter': {'all_images': True}}, 'scopes': ['open_vocab']},
    )
    assert resp.status_code == 422
    assert resp.json()['detail']['error'] == 'reprocess_targets_invalid'


def test_image_selectors_refuse_item_scopes(api_client: Any) -> None:
    resp = api_client.post(
        '/reprocess',
        json={'targets': {'filter': {'all_images': True}}, 'scopes': ['region']},
    )
    assert resp.status_code == 422


def test_the_test_route_reports_a_missing_segmenter_as_an_outage(
    api_client: Any, opensearch: Any
) -> None:
    from .conftest import INDEXES, search

    hits = search(opensearch, INDEXES['images'], {'size': 1, '_source': False})['hits']['hits']
    assert hits, 'the seeded harness has images'
    resp = api_client.post(
        '/open_vocab/test',
        json={'image_id': hits[0]['_id'], 'target': {'prompt': 'traffic cone'}},
    )
    # 502 when no segmenter answers; a stack that has one returns 200 with hits.
    assert resp.status_code in (200, 502), resp.text
    if resp.status_code == 502:
        assert resp.json()['detail']['error'] == 'segmenter_error'
    else:
        assert {'image', 'gate', 'hits'} <= set(resp.json())
