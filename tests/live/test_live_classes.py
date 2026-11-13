"""Live cohort-(a) scenarios: class registry, holdout freeze, class merge.

Order inside this module is load-bearing and intentional: the holdout
freeze must run *before* the merge scenarios, because the merge-refusal
test's whole point is that a merge which would relabel frozen holdout rows
is rejected. pytest preserves in-file definition order, which is why these
three live together rather than being spread across modules.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.services.curation.keymap import reserved_hotkeys

from .conftest import (
    FROZEN_CLASS,
    INDEXES,
    MERGE_SOURCE_CLASS,
    MERGE_TARGET_CLASS,
    crop_ids_in,
    get_doc,
    refresh,
    request_when_free,
    run_near_duplicates,
    search,
    wait_job,
)


pytestmark = pytest.mark.live


def _classes(api_client: Any) -> dict[int, dict[str, Any]]:
    resp = api_client.get('/classes')
    resp.raise_for_status()
    return {c['class_id']: c for c in resp.json()['classes']}


def test_registry_lists_the_seeded_classes_with_live_counts(
    api_client: Any, class_ids: dict[str, int]
) -> None:
    classes = _classes(api_client)
    assert len(classes) >= 8, classes
    box = classes[class_ids['box']]
    # Counts come from a live aggregation over the items index, not from
    # the registry's stored sample_count.
    assert box['sample_count'] > 0


# The registry lives in PostgreSQL and outlives a harness reseed, so a probe class is named per
# session.
_PROBE = f'live_probe_{uuid.uuid4().hex[:8]}'


def test_create_class_appends_to_the_registry(api_client: Any) -> None:
    resp = api_client.post('/classes', json={'name': _PROBE, 'group': 'harness'})
    assert resp.status_code == 201, resp.text
    new_id = resp.json()['class_id']
    assert new_id >= 8

    classes = _classes(api_client)
    assert classes[new_id]['class_name'] == _PROBE
    assert classes[new_id]['group'] == 'harness'

    # Append-only: a duplicate name is a conflict, not a silent no-op.
    dup = api_client.post('/classes', json={'name': _PROBE})
    assert dup.status_code == 409, dup.text


def test_rename_persists(api_client: Any) -> None:
    classes = _classes(api_client)
    target = next(c for c in classes.values() if c['class_name'] == _PROBE)
    class_id = target['class_id']
    renamed = f'{_PROBE}_renamed'

    resp = api_client.put(f'/classes/{class_id}', json={'name': renamed})
    assert resp.status_code == 200, resp.text
    assert resp.json()['class_name'] == renamed

    assert _classes(api_client)[class_id]['class_name'] == renamed


def test_hotkey_set_clear_and_refusals_in_a_registry_of_its_own(own_registry: Any) -> None:
    def classes() -> dict[int, dict[str, Any]]:
        resp = own_registry.get('/classes')
        resp.raise_for_status()
        return {c['class_id']: c for c in resp.json()['classes']}

    first = own_registry.post('/classes', json={'name': 'alpha'})
    second = own_registry.post('/classes', json={'name': 'bravo'})
    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    first_id, second_id = first.json()['class_id'], second.json()['class_id']
    letter = classes()[first_id]['hotkey_letter']
    assert letter, 'a new class takes the first free letter'
    assert classes()[second_id]['hotkey_letter'] != letter

    # Release the class's own letter, read the release back, set it again, read that back.
    cleared = own_registry.put(f'/classes/{first_id}', json={'hotkey_letter': ''})
    assert cleared.status_code == 200, cleared.text
    assert classes()[first_id]['hotkey_letter'] is None
    again = own_registry.put(f'/classes/{first_id}', json={'hotkey_letter': letter})
    assert again.status_code == 200, again.text
    assert again.json()['hotkey_letter'] == letter
    assert classes()[first_id]['hotkey_letter'] == letter

    # A letter another class holds is a conflict; a reserved action letter is refused.
    taken = own_registry.put(f'/classes/{second_id}', json={'hotkey_letter': letter})
    assert taken.status_code == 409, taken.text
    reserved = own_registry.put(
        f'/classes/{second_id}', json={'hotkey_letter': reserved_hotkeys({})[0]}
    )
    assert reserved.status_code == 422, reserved.text
    assert classes()[second_id]['hotkey_letter'] != letter


def test_settings_put_then_get_round_trips(api_client: Any) -> None:
    methods = api_client.get('/methods')
    methods.raise_for_status()
    by_axis: dict[str, list[str]] = {}
    for entry in methods.json()['strategies']:
        by_axis.setdefault(entry['axis'], []).append(entry['id'])
    # A deployment default is gated: a sort must order by a field items carry, a detection
    # profile must be servable by the harness Triton. `recent` orders by a field every item has.
    axis = 'sort'
    chosen = 'recent'
    assert chosen in by_axis[axis], by_axis

    # The PUT carries the revision it was made against (docs/design/curation_api_contract.md,
    # "expected_revision"); a write answers with the next one.
    revision = api_client.get('/settings').json()['revision']
    resp = api_client.put(
        '/settings', json={'defaults': {axis: chosen}, 'expected_revision': revision}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['defaults'][axis] == chosen
    revision = resp.json()['revision']
    assert revision > 0

    assert api_client.get('/settings').json()['defaults'][axis] == chosen

    # An id that is not advertised for that axis must 422, not persist.
    bad = api_client.put(
        '/settings',
        json={'defaults': {axis: 'definitely_not_a_strategy'}, 'expected_revision': revision},
    )
    assert bad.status_code == 422, bad.text
    assert api_client.get('/settings').json()['defaults'][axis] == chosen

    # A stale revision is a conflict, whatever the value.
    stale = api_client.put(
        '/settings', json={'defaults': {axis: chosen}, 'expected_revision': revision - 1}
    )
    assert stale.status_code == 409, stale.text


def test_freeze_test_holdout_selects_a_deterministic_cohort(
    api: Any, api_client: Any, opensearch: Any
) -> None:
    run_near_duplicates(api)

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
    api_client: Any, opensearch: Any, class_ids: dict[str, int]
) -> None:
    frozen_id = class_ids[FROZEN_CLASS]
    frozen_in_class = search(
        opensearch,
        INDEXES['items'],
        {
            'size': 0,
            'query': {
                'bool': {
                    'must': [
                        {'term': {'class_id': frozen_id}},
                        {'term': {'test_holdout': True}},
                    ]
                }
            },
        },
    )['hits']['total']['value']
    assert frozen_in_class > 0, 'precondition: the freeze must have covered this class'

    resp = api_client.post(
        '/classes/merge', json={'source_id': frozen_id, 'target_id': class_ids['crate']}
    )
    assert resp.status_code == 409, resp.text
    assert 'test_holdout' in resp.text

    # Refusal must be total: the registry entry stays active.
    assert _classes(api_client)[frozen_id]['deprecated'] is False


def test_merge_success_relabels_items(
    api_client: Any, opensearch: Any, class_ids: dict[str, int]
) -> None:
    source_id = class_ids[MERGE_SOURCE_CLASS]
    target_id = class_ids[MERGE_TARGET_CLASS]
    source_ids = crop_ids_in(opensearch, 'cls7', limit=40)
    assert source_ids, 'precondition: the mergeable cohort must exist'

    # The merge is a queued job (docs/design/curation_api_contract.md, class merge): 202 with the
    # job, then the relabel lands when the worker finishes it.
    # A rename earlier in this module queues a projection sync in the merge's exclusivity group
    # (class_registry_write); the merge is answered 409 job_busy until that served job ends.
    resp = request_when_free(
        lambda: api_client.post(
            '/classes/merge', json={'source_id': source_id, 'target_id': target_id}
        ),
        busy='job_busy',
        on_active=lambda job_id: wait_job(api_client, job_id),
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body['kind'] == 'class_merge'
    assert body['params'] == {'source_id': source_id, 'target_id': target_id}
    job = wait_job(api_client, body['id'])
    assert job['status'] == 'completed', job

    classes = _classes(api_client)
    assert classes[source_id]['deprecated'] is True
    assert classes[source_id]['merged_into'] == target_id

    refresh(opensearch, INDEXES['items'])

    still_source = opensearch.post(
        f'/{INDEXES["items"]}/_count',
        json={'query': {'term': {'class_id': source_id}}},
    ).json()['count']
    assert still_source == 0, 'every non-holdout item should have been relabeled'

    sample = get_doc(opensearch, INDEXES['items'], source_ids[0])['_source']
    assert sample['class_id'] == target_id
    assert sample['class_source'] == 'class_merge'
    # A merge keeps class_validated (contract: class merge); the cls7 cohort is unvalidated.
    assert sample['class_validated'] is False
    assert 'cluster_id' not in sample
    assert sample['class_id_history'][-1]['writer'] == 'class_merge'


def test_labeling_against_a_deprecated_class_is_refused(
    api_client: Any, opensearch: Any, class_ids: dict[str, int]
) -> None:
    crop_id = crop_ids_in(opensearch, 'noise', limit=1)[0]
    resp = api_client.put(
        f'/crops/{crop_id}/label', json={'class_id': class_ids[MERGE_SOURCE_CLASS]}
    )
    assert resp.status_code == 400, resp.text
    assert 'unknown class_id' in resp.text
