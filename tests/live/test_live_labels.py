"""Live cohort-(a) scenarios: the item label/move/exclude write paths.

Every test here drives a real HTTP endpoint against the harness API and
then asserts on what actually landed in OpenSearch — never on the
endpoint's own response alone, which is what makes these different from
the offline router tests.

Cohort: ``cls1_*`` (v6-sourced, never human-validated), chosen so these
writes can never collide with the human-validated holdout cohort the
class-merge scenarios freeze.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

import pytest

from .conftest import CLASS_NAMES, INDEXES, crop_ids_in, get_doc, refresh


pytestmark = pytest.mark.live


def _source(opensearch: Any, crop_id: str) -> dict[str, Any]:
    return get_doc(opensearch, INDEXES['items'], crop_id)['_source']


@pytest.fixture(scope='module')
def label_cohort(opensearch: Any) -> list[str]:
    ids = crop_ids_in(opensearch, 'cls1', limit=40)
    assert len(ids) >= 30, f'expected the seeded cls1 cohort, got {len(ids)} ids'
    return ids


def test_single_label_writes_class_and_grows_history(
    client: Any, opensearch: Any, label_cohort: list[str]
) -> None:
    crop_id = label_cohort[0]
    before = _source(opensearch, crop_id)
    history_before = len(before.get('class_id_history') or [])

    resp = client.put(f'/crops/{crop_id}/label', json={'class_id': 3, 'label_source': 'human'})
    assert resp.status_code == 200, resp.text
    assert resp.json()['class_name'] == CLASS_NAMES[3]

    after = _source(opensearch, crop_id)
    assert after['class_id'] == 3
    assert after['class_name'] == CLASS_NAMES[3]
    assert after['class_validated'] is True
    assert after['class_source'] == 'human'
    # cluster_id mirrors class_id for class clusters, and the AHC sub-id is
    # cluster-local so a class change must clear it.
    assert after['cluster_id'] == 3
    assert after.get('cluster_subid') is None
    assert after['class_labeler'] == 'human'
    history_after = after.get('class_id_history') or []
    assert len(history_after) == history_before + 1, history_after
    assert history_after[-1]['class_id'] == before['class_id']
    assert history_after[-1]['writer'] == 'human:label_crop'


def test_second_label_appends_another_history_entry(
    client: Any, opensearch: Any, label_cohort: list[str]
) -> None:
    crop_id = label_cohort[0]
    before = _source(opensearch, crop_id)
    client.put(f'/crops/{crop_id}/label', json={'class_id': 4}).raise_for_status()
    after = _source(opensearch, crop_id)
    assert after['class_id'] == 4
    assert len(after['class_id_history']) == len(before['class_id_history']) + 1
    assert after['class_id_history'][-1]['class_id'] == 3


def test_unknown_class_is_rejected_and_leaves_the_doc_untouched(
    client: Any, opensearch: Any, label_cohort: list[str]
) -> None:
    crop_id = label_cohort[1]
    doc_before = get_doc(opensearch, INDEXES['items'], crop_id)

    resp = client.put(f'/crops/{crop_id}/label', json={'class_id': 9999})
    assert resp.status_code == 400, resp.text
    assert 'unknown class_id' in resp.text

    doc_after = get_doc(opensearch, INDEXES['items'], crop_id)
    assert doc_after['_version'] == doc_before['_version']
    assert doc_after['_seq_no'] == doc_before['_seq_no']


def test_batch_label_updates_every_crop(client: Any, opensearch: Any, label_cohort: list[str]) -> None:
    batch = label_cohort[2:8]
    resp = client.put('/crops/batch_label', json={'crop_ids': batch, 'class_id': 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['updated'] == len(batch), body
    assert body['conflicts'] == []

    for crop_id in batch:
        src = _source(opensearch, crop_id)
        assert src['class_id'] == 5
        assert src['class_name'] == CLASS_NAMES[5]
        assert src['class_validated'] is True
        assert src['cluster_id'] == 5


def test_unlabel_resets_class_provenance(client: Any, opensearch: Any, label_cohort: list[str]) -> None:
    crop_id = label_cohort[2]
    resp = client.delete(f'/crops/{crop_id}/label')
    assert resp.status_code == 200, resp.text

    src = _source(opensearch, crop_id)
    assert src['class_validated'] is False
    assert src['label_source'] == ''
    assert src['class_source'] is None
    assert src['class_labeler'] is None
    # The reset is itself an audit event.
    assert src['class_id_history'][-1]['writer'] == 'human:unlabel_crop'


def test_review_dismiss_is_recorded(client: Any, opensearch: Any, label_cohort: list[str]) -> None:
    crop_id = label_cohort[9]
    resp = client.post(f'/crops/{crop_id}/review_dismiss')
    assert resp.status_code == 200, resp.text
    src = _source(opensearch, crop_id)
    assert src['review_dismissed_by'] == 'human'
    assert src['review_dismissed_at']


def test_exclude_then_unexclude_round_trip(
    client: Any, opensearch: Any, label_cohort: list[str]
) -> None:
    batch = label_cohort[10:13]
    resp = client.post('/crops/batch_exclude', json={'crop_ids': batch, 'reason': 'blurry'})
    assert resp.status_code == 200, resp.text
    assert resp.json()['excluded'] == len(batch)
    for crop_id in batch:
        src = _source(opensearch, crop_id)
        assert src['class_excluded'] is True
        assert src['excluded_reason'] == 'blurry'
        # Leaves its candidate bucket immediately (the excluded sentinel).
        assert src['cluster_id'] == -2
        assert src['class_validated'] is False

    resp = client.post('/crops/batch_unexclude', json={'crop_ids': batch})
    assert resp.status_code == 200, resp.text
    assert resp.json()['unexcluded'] == len(batch)
    for crop_id in batch:
        src = _source(opensearch, crop_id)
        assert src['class_excluded'] is False
        assert src['excluded_reason'] is None
        assert src['cluster_id'] is None


def test_move_relabels_into_the_destination_cluster(
    client: Any, opensearch: Any, label_cohort: list[str]
) -> None:
    batch = label_cohort[14:17]
    resp = client.post('/crops/move', json={'crop_ids': batch, 'cluster_id': 6})
    assert resp.status_code == 200, resp.text
    assert resp.json()['updated'] == len(batch)
    for crop_id in batch:
        src = _source(opensearch, crop_id)
        assert src['cluster_id'] == 6
        assert src['class_id'] == 6
        assert src['class_name'] == CLASS_NAMES[6]
        assert src['class_source'] == 'human_move'
        assert src['class_validated'] is True
        assert src['cluster_subid'] is None


def test_flag_new_class_queues_for_the_curator(
    client: Any, opensearch: Any, label_cohort: list[str]
) -> None:
    batch = label_cohort[18:20]
    resp = client.post('/crops/flag_new_class', json={'crop_ids': batch, 'note': 'live-harness probe'})
    assert resp.status_code == 200, resp.text
    assert resp.json()['flagged'] == len(batch)
    refresh(opensearch, INDEXES['items'])
    for crop_id in batch:
        src = _source(opensearch, crop_id)
        assert src['needs_new_class'] is True
        assert src['needs_new_class_note'] == 'live-harness probe'


def test_parallel_labels_of_one_item_do_not_tear_the_history(
    client: Any, opensearch: Any, label_cohort: list[str]
) -> None:
    """Two concurrent human labels on the same doc.

    OCC must serialize them: one wins outright, the other either retries
    onto the winner's version or surfaces the final-conflict error. Either
    way the stored history must be internally consistent — no duplicated
    or half-written entry.
    """
    crop_id = label_cohort[21]
    before = _source(opensearch, crop_id)
    n_history_before = len(before.get('class_id_history') or [])

    def label(class_id: int) -> int:
        return client.put(f'/crops/{crop_id}/label', json={'class_id': class_id}).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(label, (2, 3)))

    assert 200 in statuses, statuses
    # A losing writer is allowed to 409/500 out, but never to corrupt state.
    assert all(s in (200, 409, 500) for s in statuses), statuses

    after = _source(opensearch, crop_id)
    assert after['class_id'] in (2, 3)
    history = after['class_id_history']
    n_wins = sum(1 for s in statuses if s == 200)
    assert len(history) == n_history_before + n_wins, history
    for entry in history:
        # Every entry is a complete snapshot, not a partial write.
        assert set(entry) >= {'class_id', 'class_source', 'writer', 'at'}
        assert isinstance(entry['class_id'], int)
