"""Live cohort-(a) scenarios: the region-of-interest human edit surface.

The HTTP wire model still spells these fields ``plate_*`` (a frozen API
contract); the OpenSearch documents spell them ``region_*`` via
``RegionFields``. Every assertion below deliberately checks the *stored*
name, because the whole point of the indirection is that the two can
differ without anything silently falling through.
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


def test_set_region_bbox_writes_human_provenance(
    kb: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[0]
    bbox = [0.11, 0.22, 0.33, 0.44]

    resp = kb.put(f'/crops/{crop_id}/plate', json={'bbox_norm': bbox, 'label_source': 'human'})
    assert resp.status_code == 200, resp.text
    assert resp.json()['plate_status'] == 'detected'

    src = _source(opensearch, crop_id)
    assert src['region_bbox_norm'] == pytest.approx(bbox)
    assert src['region_status'] == 'detected'
    # A human-set box is ground truth (score 1.0) and terminal.
    assert src['region_score'] == 1.0
    assert src['region_verified'] is True
    assert src['region_validated'] is True
    assert src['region_detector'] == HUMAN_DETECTOR
    assert src['region_bbox_frame'] == 'source'
    assert src['region_label_source'] == 'human'
    # A region edit must never touch the class side.
    assert 'class_validated' in src


def test_clear_region_bbox_records_a_deliberate_negative(
    kb: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[1]
    resp = kb.put(f'/crops/{crop_id}/plate', json={'bbox_norm': None})
    assert resp.status_code == 200, resp.text
    assert resp.json()['plate_status'] == 'no_plate_visible'

    src = _source(opensearch, crop_id)
    assert src['region_bbox_norm'] is None
    assert src['region_score'] is None
    assert src['region_status'] == 'no_plate_visible'
    assert src['region_validated'] is True


@pytest.mark.parametrize(
    'bbox',
    [
        [0.5, 0.5, 0.4, 0.6],  # x2 <= x1 -> degenerate
        [0.1, 0.1, 1.4, 0.6],  # outside [0, 1]
    ],
)
def test_malformed_region_bbox_is_rejected(
    kb: Any, opensearch: Any, region_cohort: list[str], bbox: list[float]
) -> None:
    crop_id = region_cohort[2]
    before = get_doc(opensearch, INDEXES['items'], crop_id)
    resp = kb.put(f'/crops/{crop_id}/plate', json={'bbox_norm': bbox})
    assert resp.status_code == 400, resp.text
    after = get_doc(opensearch, INDEXES['items'], crop_id)
    assert after['_seq_no'] == before['_seq_no']


def test_patch_region_metadata_writes_text_and_source(
    kb: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[3]
    resp = kb.patch(
        f'/crops/{crop_id}/plate_meta',
        json={'plate_text': 'LIVE-HARNESS-7', 'label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text
    # updated_fields echoes the frozen plate_* wire contract, never the
    # RegionFields storage key (region_text) checked below via `src`.
    assert 'plate_text' in resp.json()['updated_fields']

    src = _source(opensearch, crop_id)
    assert src['region_text'] == 'LIVE-HARNESS-7'
    assert src['region_text_source'] == 'human'
    assert src['region_text_confidence'] == 1.0
    assert src['region_validated'] is True
    # A text-only edit must not touch the bbox or its cluster bucket.
    assert src['region_bbox_norm'] is not None


def test_patch_region_status_routes_false_positives_to_the_fp_bucket(
    kb: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[4]
    resp = kb.patch(
        f'/crops/{crop_id}/plate_meta',
        json={'plate_status': 'false_positive', 'label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text

    src = _source(opensearch, crop_id)
    assert src['region_status'] == 'false_positive'
    assert src['region_cluster_id'] == FP_REGION_CLUSTER_ID
    assert src['region_cluster_subid'] is None
    assert src['region_validated'] is True

    # Un-marking releases it so the next re-cluster re-absorbs it.
    resp = kb.patch(
        f'/crops/{crop_id}/plate_meta',
        json={'plate_status': 'detected', 'label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text
    src = _source(opensearch, crop_id)
    assert src['region_status'] == 'detected'
    assert src['region_cluster_id'] is None


def test_patch_region_rejects_a_non_human_status_and_an_empty_body(
    kb: Any, region_cohort: list[str]
) -> None:
    crop_id = region_cohort[5]
    bad_status = kb.patch(
        f'/crops/{crop_id}/plate_meta', json={'plate_status': 'pending_detection'}
    )
    assert bad_status.status_code == 400, bad_status.text

    empty = kb.patch(f'/crops/{crop_id}/plate_meta', json={'label_source': 'human'})
    assert empty.status_code == 400, empty.text


def test_batch_region_clear(kb: Any, opensearch: Any, region_cohort: list[str]) -> None:
    batch = region_cohort[6:9]
    resp = kb.put('/crops/batch_plate', json={'crop_ids': batch, 'bbox_norm': None})
    assert resp.status_code == 200, resp.text
    assert resp.json()['updated'] == len(batch)
    assert resp.json()['conflicts'] == []
    for crop_id in batch:
        src = _source(opensearch, crop_id)
        assert src['region_status'] == 'no_plate_visible'
        assert src['region_bbox_norm'] is None


def test_bulk_region_status_confirms_many_regions_at_once(
    kb: Any, opensearch: Any, region_cohort: list[str]
) -> None:
    batch = region_cohort[10:15]
    resp = kb.post(
        '/plates/batch_status',
        json={
            'crop_ids': batch,
            'plate_status': 'detected',
            'plate_verified': True,
            'label_source': 'human',
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
    kb: Any, region_cohort: list[str]
) -> None:
    resp = kb.post(
        '/plates/batch_status',
        json={'crop_ids': region_cohort[16:17], 'plate_status': 'detection_failed'},
    )
    assert resp.status_code == 400, resp.text


def test_region_browse_reflects_the_human_edits(kb: Any) -> None:
    resp = kb.get('/plates', params={'page_size': 50, 'detector': HUMAN_DETECTOR})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['total'] >= 1, body
    item = body['items'][0]
    assert item['plate_detector'] == HUMAN_DETECTOR
    assert item['plate_thumbnail_url'].endswith('/region_thumbnail')


def test_training_candidate_cohorts_are_queryable(kb: Any) -> None:
    resp = kb.get('/plates/training_candidates', params={'mode': 'human_corrected'})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['mode'] == 'human_corrected'
    assert body['total'] >= 1, body
    assert body['items'][0]['selection_reason']

    unknown = kb.get('/plates/training_candidates', params={'mode': 'not_a_mode'})
    assert unknown.status_code == 400, unknown.text


def test_region_thumbnail_serves_real_pixels(kb: Any, region_cohort: list[str]) -> None:
    resp = kb.get(f'/crops/{region_cohort[20]}/region_thumbnail')
    assert resp.status_code == 200, resp.text
    assert resp.headers['content-type'].startswith('image/')
    assert len(resp.content) > 100
