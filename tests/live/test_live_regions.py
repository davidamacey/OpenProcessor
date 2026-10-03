"""Live cohort-(a) scenarios: the region-of-interest human edit surface.

The HTTP wire model spells these fields ``region_*``; the OpenSearch
documents also spell them ``region_*`` via ``RegionFields``, but a
deployment can override the *storage* key via ``RegionFields`` while the
wire shape stays fixed. Every assertion below deliberately checks the
*stored* name, because the whole point of the indirection is that the two
can differ without anything silently falling through.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import FP_REGION_CLUSTER_ID, INDEXES, crop_ids_in, get_doc, refresh


pytestmark = pytest.mark.live

HUMAN_DETECTOR = 'human'


def _source(opensearch: Any, crop_id: str) -> dict[str, Any]:
    return get_doc(opensearch, INDEXES['items'], crop_id)['_source']


@pytest.fixture(scope='module')
def region_cohort(opensearch: Any) -> list[str]:
    ids = crop_ids_in(opensearch, 'rgn', limit=40)
    assert len(ids) >= 20, f'expected the seeded rgn cohort, got {len(ids)}'
    return ids


def test_put_regions_new_box_writes_human_provenance(
    api_client: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[0]
    bbox = [0.11, 0.22, 0.33, 0.44]

    resp = api_client.put(
        f'/crops/{crop_id}/regions',
        json={'boxes': [{'box_id': None, 'bbox_norm': bbox}], 'region_label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['item']['region_status'] == 'detected'

    src = _source(opensearch, crop_id)
    (box,) = src['region_boxes']
    assert box['bbox_norm'] == pytest.approx(bbox)
    assert src['region_status'] == 'detected'
    # A human-drawn box is ground truth (score 1.0) and terminal.
    assert box['state'] == 'accepted'
    assert box['score'] == 1.0
    assert box['detector'] == HUMAN_DETECTOR
    assert src['region_verified'] is True
    assert src['region_validated'] is True
    assert src['region_label_source'] == 'human'
    # A region edit must never touch the class side.
    assert 'class_validated' in src


def test_put_regions_empty_list_records_a_deliberate_negative(
    api_client: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[1]
    resp = api_client.put(f'/crops/{crop_id}/regions', json={'boxes': []})
    assert resp.status_code == 200, resp.text
    assert resp.json()['item']['region_status'] == 'no_region_visible'

    src = _source(opensearch, crop_id)
    assert src['region_boxes'] == []
    assert src['region_status'] == 'no_region_visible'
    assert src['region_validated'] is True


@pytest.mark.parametrize(
    'bbox',
    [
        [0.5, 0.5, 0.4, 0.6],  # x2 <= x1 -> degenerate
        [0.1, 0.1, 1.4, 0.6],  # outside [0, 1]
    ],
)
def test_malformed_region_bbox_is_rejected(
    api_client: Any, opensearch: Any, region_cohort: list[str], bbox: list[float]
) -> None:
    crop_id = region_cohort[2]
    before = get_doc(opensearch, INDEXES['items'], crop_id)
    resp = api_client.put(
        f'/crops/{crop_id}/regions', json={'boxes': [{'box_id': None, 'bbox_norm': bbox}]}
    )
    assert resp.status_code == 422, resp.text
    after = get_doc(opensearch, INDEXES['items'], crop_id)
    assert after['_seq_no'] == before['_seq_no']


def test_patch_box_text_writes_text_and_source_on_that_box(
    api_client: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[3]
    before = _source(opensearch, crop_id)['region_boxes'][0]
    resp = api_client.patch(
        f'/crops/{crop_id}/regions/{before["box_id"]}', json={'text': 'LIVE-HARNESS-7'}
    )
    assert resp.status_code == 200, resp.text

    box = _source(opensearch, crop_id)['region_boxes'][0]
    assert box['text'] == 'LIVE-HARNESS-7'
    assert box['text_source'] == 'human'
    # A text-only edit must not touch the box's geometry or cluster bucket.
    assert box['bbox_norm'] == before['bbox_norm']
    assert box['cluster_id'] == before['cluster_id']


def test_patch_region_status_routes_false_positives_to_the_fp_bucket(
    api_client: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[4]
    resp = api_client.patch(
        f'/crops/{crop_id}/region_meta',
        json={'region_status': 'false_positive', 'region_label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text

    src = _source(opensearch, crop_id)
    assert src['region_status'] == 'false_positive'
    assert {b['cluster_id'] for b in src['region_boxes']} == {FP_REGION_CLUSTER_ID}
    assert {b['cluster_subid'] for b in src['region_boxes']} == {None}
    assert src['region_validated'] is True

    # A false-positive item has no accepted box, so whole-set confirm is a
    # 422 (`no_accepted_box`); un-marking is per box.
    confirm = api_client.patch(f'/crops/{crop_id}/region_meta', json={'region_status': 'detected'})
    assert confirm.status_code == 422, confirm.text

    # Accepting each box releases it so the next re-cluster re-absorbs it.
    for box in src['region_boxes']:
        resp = api_client.patch(
            f'/crops/{crop_id}/regions/{box["box_id"]}', json={'state': 'accepted'}
        )
        assert resp.status_code == 200, resp.text
    src = _source(opensearch, crop_id)
    assert src['region_status'] == 'detected'
    assert {b['cluster_id'] for b in src['region_boxes']} == {None}


def test_patch_region_rejects_a_non_human_status_and_an_empty_body(
    api_client: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[5]
    bad_status = api_client.patch(
        f'/crops/{crop_id}/region_meta', json={'region_status': 'pending_detection'}
    )
    assert bad_status.status_code == 400, bad_status.text

    empty = api_client.patch(f'/crops/{crop_id}/region_meta', json={'region_label_source': 'human'})
    assert empty.status_code == 400, empty.text


def test_batch_regions_clear(api_client: Any, opensearch: Any, region_cohort: list[str]) -> None:
    batch = region_cohort[6:9]
    resp = api_client.put('/crops/batch_regions', json={'crop_ids': batch, 'boxes': []})
    assert resp.status_code == 200, resp.text
    assert resp.json()['updated'] == len(batch)
    assert resp.json()['conflicts'] == []
    for crop_id in batch:
        src = _source(opensearch, crop_id)
        assert src['region_status'] == 'no_region_visible'
        assert src['region_boxes'] == []


def test_bulk_region_status_confirms_many_regions_at_once(
    api_client: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    batch = region_cohort[10:15]
    resp = api_client.post(
        '/regions/batch_status',
        json={
            'crop_ids': batch,
            'region_status': 'detected',
            'region_label_source': 'human',
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['updated'] == len(batch)

    refresh(opensearch, INDEXES['items'])
    for crop_id in batch:
        src = _source(opensearch, crop_id)
        assert src['region_status'] == 'detected'
        assert src['region_verified'] is True
        assert src['region_validated'] is True


def test_bulk_region_status_rejects_a_pipeline_only_status(
    api_client: Any, region_cohort: list[str]
) -> None:
    resp = api_client.post(
        '/regions/batch_status',
        json={'crop_ids': region_cohort[16:17], 'region_status': 'detection_failed'},
    )
    assert resp.status_code == 400, resp.text


def test_region_browse_reflects_the_human_edits(api_client: Any) -> None:
    resp = api_client.get('/regions', params={'page_size': 50, 'detector': HUMAN_DETECTOR})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['total'] >= 1, body
    row = body['items'][0]
    box = next(b for b in row['region_boxes'] if b['box_id'] == row['region_box_id'])
    assert box['detector'] == HUMAN_DETECTOR
    assert box['thumbnail_url'].endswith(f'/region_thumbnail?box_id={box["box_id"]}')


def test_training_candidate_cohorts_are_queryable(api_client: Any) -> None:
    resp = api_client.get('/regions/training_candidates', params={'mode': 'human_corrected'})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['mode'] == 'human_corrected'
    assert body['total'] >= 1, body
    assert body['items'][0]['selection_reason']

    unknown = api_client.get('/regions/training_candidates', params={'mode': 'not_a_mode'})
    assert unknown.status_code == 400, unknown.text


def test_region_thumbnail_serves_real_pixels(
    api_client: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[20]
    box_id = _source(opensearch, crop_id)['region_boxes'][0]['box_id']
    resp = api_client.get(f'/crops/{crop_id}/region_thumbnail', params={'box_id': box_id})
    assert resp.status_code == 200, resp.text
    assert api_client.get(f'/crops/{crop_id}/region_thumbnail').status_code == 422
    assert resp.headers['content-type'].startswith('image/')
    assert len(resp.content) > 100


def test_region_stage_pause_resume_round_trip(api_client: Any) -> None:
    state = api_client.get('/region_stage')
    assert state.status_code == 200, state.text
    body = state.json()
    assert set(body['counts']) == {'pending_detection', 'pending_verification', 'gate_skipped'}
    assert body['rerun_skipped']['scopes'] == ['region']
    assert body['rerun_skipped']['dry_run'] is True

    try:
        paused = api_client.post('/region_stage/pause')
        assert paused.status_code == 200, paused.text
        assert paused.json()['paused'] is True
        assert paused.json()['paused_since']
        assert api_client.get('/region_stage').json()['paused'] is True
    finally:
        resumed = api_client.post('/region_stage/resume')
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()['paused'] is False

    dry = api_client.post('/reprocess', json=body['rerun_skipped'])
    assert dry.status_code == 200, dry.text
    assert dry.json()['dry_run'] is True
