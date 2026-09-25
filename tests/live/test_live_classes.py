"""Live cohort-(a) scenarios: class registry, holdout freeze, class merge.

Order inside this module is load-bearing and intentional: the holdout
freeze must run *before* the merge scenarios, because the merge-refusal
test's whole point is that a merge which would relabel frozen holdout rows
is rejected. pytest preserves in-file definition order, which is why these
three live together rather than being spread across modules.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import (
    FROZEN_CLASS_ID,
    INDEXES,
    MERGE_SOURCE_CLASS_ID,
    MERGE_TARGET_CLASS_ID,
    crop_ids_in,
    get_doc,
    refresh,
    search,
)


pytestmark = pytest.mark.live


def _classes(api_client: Any) -> dict[int, dict[str, Any]]:
    resp = api_client.get('/classes')
    resp.raise_for_status()
    return {c['class_id']: c for c in resp.json()['classes']}


def test_registry_lists_the_seeded_classes_with_live_counts(api_client: Any) -> None:
    classes = _classes(api_client)
    assert len(classes) >= 8, classes
    assert classes[0]['class_name'] == 'box'
    # Counts come from a live aggregation over the items index, not from
    # the registry file's stored sample_count.
    assert classes[0]['sample_count'] > 0
    assert classes[0]['cluster_size'] > 0


def test_create_class_appends_to_the_registry(api_client: Any) -> None:
    resp = api_client.post('/classes', json={'name': 'live_probe_class', 'group': 'harness'})
    assert resp.status_code == 201, resp.text
    new_id = resp.json()['class_id']
    assert new_id >= 8

    classes = _classes(api_client)
    assert classes[new_id]['class_name'] == 'live_probe_class'
    assert classes[new_id]['group'] == 'harness'

    # Append-only: a duplicate name is a conflict, not a silent no-op.
    dup = api_client.post('/classes', json={'name': 'live_probe_class'})
    assert dup.status_code == 409, dup.text


def test_rename_and_hotkey_update_persist(api_client: Any) -> None:
    classes = _classes(api_client)
    target = next(c for c in classes.values() if c['class_name'] == 'live_probe_class')
    class_id = target['class_id']

    resp = api_client.put(f'/classes/{class_id}', json={'name': 'live_probe_renamed'})
    assert resp.status_code == 200, resp.text
    assert resp.json()['class_name'] == 'live_probe_renamed'

    resp = api_client.put(f'/classes/{class_id}', json={'hotkey_letter': 'q'})
    assert resp.status_code == 200, resp.text
    assert resp.json()['hotkey_letter'] == 'q'

    # A reserved action letter must be refused rather than silently bound.
    resp = api_client.put(f'/classes/{class_id}', json={'hotkey_letter': 'g'})
    assert resp.status_code == 422, resp.text

    assert _classes(api_client)[class_id]['class_name'] == 'live_probe_renamed'


def test_sync_to_opensearch_mirrors_the_registry(api_client: Any, opensearch: Any) -> None:
    resp = api_client.post('/classes/sync_to_opensearch')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['upserted'] == body['n_classes'] >= 9

    refresh(opensearch, INDEXES['classes'])
    count = opensearch.get(f'/{INDEXES["classes"]}/_count').json()['count']
    assert count == body['n_classes']
    doc = get_doc(opensearch, INDEXES['classes'], '0')['_source']
    assert doc['class_name'] == 'box'


def test_settings_put_then_get_round_trips(api_client: Any) -> None:
    methods = api_client.get('/methods')
    methods.raise_for_status()
    settable = {'cluster', 'sort', 'detection_profile', 'prompt_pack'}
    by_axis: dict[str, list[str]] = {}
    for entry in methods.json()['strategies']:
        by_axis.setdefault(entry['axis'], []).append(entry['id'])
    axis = next(a for a in sorted(by_axis) if a in settable)
    chosen = sorted(by_axis[axis])[0]

    resp = api_client.put('/settings', json={'defaults': {axis: chosen}})
    assert resp.status_code == 200, resp.text
    assert resp.json()['defaults'][axis] == chosen

    assert api_client.get('/settings').json()['defaults'][axis] == chosen

    # An id that is not advertised for that axis must 422, not persist.
    bad = api_client.put('/settings', json={'defaults': {axis: 'definitely_not_a_strategy'}})
    assert bad.status_code == 422, bad.text
    assert api_client.get('/settings').json()['defaults'][axis] == chosen


def test_freeze_test_holdout_selects_a_deterministic_cohort(
    api_client: Any, opensearch: Any
) -> None:
    resp = api_client.post('/test_holdout/freeze', json={'percent': 10})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['n_frozen'] > 0
    assert body['n_classes_covered'] >= 5
    assert len(body['test_holdout_sha']) == 64

    refresh(opensearch, INDEXES['items'])
    frozen = opensearch.post(
        f'/{INDEXES["items"]}/_count',
        json={'query': {'term': {'test_holdout': True}}},
    ).json()['count']
    assert frozen == body['n_frozen']

    stats = api_client.get('/test_holdout/stats')
    stats.raise_for_status()
    assert stats.json()['total'] == body['n_frozen']

    # A second freeze without ?force must refuse rather than re-roll the
    # frozen set under a running experiment.
    again = api_client.post('/test_holdout/freeze', json={'percent': 10})
    assert again.status_code == 409, again.text


def test_merge_is_refused_when_it_would_relabel_frozen_holdout_rows(
    api_client: Any, opensearch: Any
) -> None:
    frozen_in_class = search(
        opensearch,
        INDEXES['items'],
        {
            'size': 0,
            'query': {
                'bool': {
                    'must': [
                        {'term': {'class_id': FROZEN_CLASS_ID}},
                        {'term': {'test_holdout': True}},
                    ]
                }
            },
        },
    )['hits']['total']['value']
    assert frozen_in_class > 0, 'precondition: the freeze must have covered this class'

    resp = api_client.post('/classes/merge', json={'source_id': FROZEN_CLASS_ID, 'target_id': 3})
    assert resp.status_code == 409, resp.text
    assert 'test_holdout' in resp.text

    # Refusal must be total: the registry entry stays active.
    assert _classes(api_client)[FROZEN_CLASS_ID]['deprecated'] is False


def test_merge_success_relabels_items_and_confirmed_labels(
    api_client: Any, opensearch: Any
) -> None:
    source_ids = crop_ids_in(opensearch, 'cls7', limit=40)
    assert source_ids, 'precondition: the mergeable cohort must exist'

    resp = api_client.post(
        '/classes/merge',
        json={'source_id': MERGE_SOURCE_CLASS_ID, 'target_id': MERGE_TARGET_CLASS_ID},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['deprecated'] is True
    assert body['target_id'] == MERGE_TARGET_CLASS_ID

    classes = _classes(api_client)
    assert classes[MERGE_SOURCE_CLASS_ID]['deprecated'] is True

    refresh(opensearch, INDEXES['items'])
    refresh(opensearch, INDEXES['labels_confirmed'])

    still_source = opensearch.post(
        f'/{INDEXES["items"]}/_count',
        json={'query': {'term': {'class_id': MERGE_SOURCE_CLASS_ID}}},
    ).json()['count']
    assert still_source == 0, 'every non-holdout item should have been relabeled'

    sample = get_doc(opensearch, INDEXES['items'], source_ids[0])['_source']
    assert sample['class_id'] == MERGE_TARGET_CLASS_ID
    assert sample['class_source'] == 'class_merge'
    # A merged crop must not keep a stale human-validated flag.
    assert sample['class_validated'] is False
    assert sample['cluster_id'] == MERGE_TARGET_CLASS_ID
    assert sample['class_id_history'][-1]['writer'] == 'class_merge'

    confirmed = get_doc(opensearch, INDEXES['labels_confirmed'], f'lbl_{source_ids[0]}')['_source']
    assert confirmed['class_id'] == MERGE_TARGET_CLASS_ID
    assert confirmed['class_source'] == 'class_merge'


def test_labeling_against_a_deprecated_class_is_refused(api_client: Any, opensearch: Any) -> None:
    crop_id = crop_ids_in(opensearch, 'noise', limit=1)[0]
    resp = api_client.put(f'/crops/{crop_id}/label', json={'class_id': MERGE_SOURCE_CLASS_ID})
    assert resp.status_code == 400, resp.text
    assert 'unknown class_id' in resp.text
