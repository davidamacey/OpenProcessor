"""Paging depth guard + stable ``crop_id`` sort tiebreaker for
``GET /regions`` and ``GET /regions/training_candidates``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, get_region_fields, index_name
from src.config.curation import base_curation_config


F = get_region_fields()
CFG = base_curation_config()
ITEMS = index_name(CFG, IndexRole.ITEMS)

# GET /regions requires an active region profile (no-profile gating contract).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


def _client(fake: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _fake() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {ITEMS: {'r1': {'crop_id': 'r1', F.bbox_norm: [0.1, 0.1, 0.2, 0.2]}}}
    )


def test_list_regions_page_too_deep_is_422() -> None:
    fake = _fake()
    client = _client(fake)
    resp = client.get('/curation/projects/default/regions', params={'page': 400, 'page_size': 30})
    assert resp.status_code == 422, resp.text


def test_list_regions_page_within_window_is_fine() -> None:
    fake = _fake()
    client = _client(fake)
    resp = client.get('/curation/projects/default/regions', params={'page': 300, 'page_size': 30})
    assert resp.status_code == 200, resp.text


def test_training_candidates_page_too_deep_is_422() -> None:
    fake = _fake()
    client = _client(fake)
    resp = client.get(
        '/curation/projects/default/regions/training_candidates',
        params={'mode': 'human_corrected', 'page': 400, 'page_size': 30},
    )
    assert resp.status_code == 422, resp.text


def test_list_regions_request_body_carries_crop_id_tiebreaker() -> None:
    """Direct assertion on the actual request body sent to OpenSearch."""
    from unittest.mock import AsyncMock

    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'hits': {'total': {'value': 0}, 'hits': []}})
    client = _client(fake)

    resp = client.get('/curation/projects/default/regions')
    assert resp.status_code == 200, resp.text
    sort = fake.search.call_args.kwargs['body']['sort']
    assert sort[-1] == {'crop_id': {'order': 'asc'}}


def test_training_candidates_request_body_carries_crop_id_tiebreaker() -> None:
    from unittest.mock import AsyncMock

    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'hits': {'total': {'value': 0}, 'hits': []}})
    client = _client(fake)

    resp = client.get(
        '/curation/projects/default/regions/training_candidates', params={'mode': 'human_corrected'}
    )
    assert resp.status_code == 200, resp.text
    sort = fake.search.call_args.kwargs['body']['sort']
    assert sort[-1] == {'crop_id': {'order': 'asc'}}


# ---------------------------------------------------------------------------
# W8-cleanup: GET /regions and /regions/training_candidates now query the
# region_boxes nested list, not the retired item-level region_bbox_norm /
# region_detector / region_score scalars. These exercise the real match
# semantics via QueryFakeOpenSearch (not a canned response), so a nested
# box_query built with the wrong dotted-path field name would fail loudly.
# ---------------------------------------------------------------------------


def test_list_regions_default_filter_matches_accepted_box_only() -> None:
    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'has-accepted': {
                    'crop_id': 'has-accepted',
                    F.boxes: [
                        {'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'accepted'}
                    ],
                },
                'proposed-only': {
                    'crop_id': 'proposed-only',
                    F.boxes: [
                        {'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'proposed'}
                    ],
                },
                'no-boxes': {'crop_id': 'no-boxes'},
            }
        }
    )
    client = _client(fake)
    resp = client.get('/curation/projects/default/regions', params={'page_size': 50})
    assert resp.status_code == 200, resp.text
    assert {i['crop_id'] for i in resp.json()['items']} == {'has-accepted'}


def test_list_regions_detector_filter_matches_the_per_box_detector() -> None:
    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'by-det-a': {
                    'crop_id': 'by-det-a',
                    F.boxes: [
                        {
                            'box_id': 'b1',
                            'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                            'state': 'accepted',
                            'detector': 'det_a',
                        }
                    ],
                },
                'by-det-b': {
                    'crop_id': 'by-det-b',
                    F.boxes: [
                        {
                            'box_id': 'b1',
                            'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                            'state': 'accepted',
                            'detector': 'det_b',
                        }
                    ],
                },
            }
        }
    )
    client = _client(fake)
    resp = client.get(
        '/curation/projects/default/regions', params={'page_size': 50, 'detector': 'det_a'}
    )
    assert resp.status_code == 200, resp.text
    assert {i['crop_id'] for i in resp.json()['items']} == {'by-det-a'}


def test_training_candidates_false_positives_matches_the_kept_fp_box() -> None:
    from src.config.region_state import RegionStatus

    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'fp-item': {
                    'crop_id': 'fp-item',
                    F.status: RegionStatus.FALSE_POSITIVE.value,
                    F.boxes: [
                        {
                            'box_id': 'b1',
                            'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                            'state': 'false_positive',
                        }
                    ],
                },
                'detected-item': {
                    'crop_id': 'detected-item',
                    F.status: RegionStatus.DETECTED.value,
                    F.boxes: [
                        {'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'accepted'}
                    ],
                },
            }
        }
    )
    client = _client(fake)
    resp = client.get(
        '/curation/projects/default/regions/training_candidates',
        params={'mode': 'false_positives', 'page_size': 50},
    )
    assert resp.status_code == 200, resp.text
    assert {i['crop_id'] for i in resp.json()['items']} == {'fp-item'}


def test_low_conf_correct_excludes_a_verifier_rejected_primary_box() -> None:
    """W8-cleanup M4: the cohort must require the SAME box to be
    `accepted`, not just any low-score box from the primary detector plus
    `verified=True` somewhere on the item -- otherwise a primary box the
    verifier REJECTED at a low score still matches as long as some other
    (accepted) box exists, contaminating this "primary detector correct
    but low-confidence" training cohort with cases where the primary
    detector was actually wrong."""
    from _region_profile_fixture import REFERENCE_REGION_DETECTOR_MODEL

    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'rejected-primary': {
                    'crop_id': 'rejected-primary',
                    F.verified: True,
                    F.boxes: [
                        {
                            'box_id': 'b1',
                            'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                            'state': 'rejected',
                            'detector': REFERENCE_REGION_DETECTOR_MODEL,
                            'score': 0.3,
                        },
                        {
                            'box_id': 'b2',
                            'bbox_norm': [0.3, 0.3, 0.4, 0.4],
                            'state': 'accepted',
                            'detector': 'some_segmenter',
                            'score': 0.9,
                        },
                    ],
                },
                'accepted-low-conf': {
                    'crop_id': 'accepted-low-conf',
                    F.verified: True,
                    F.boxes: [
                        {
                            'box_id': 'b1',
                            'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                            'state': 'accepted',
                            'detector': REFERENCE_REGION_DETECTOR_MODEL,
                            'score': 0.4,
                        }
                    ],
                },
                'accepted-high-conf': {
                    'crop_id': 'accepted-high-conf',
                    F.verified: True,
                    F.boxes: [
                        {
                            'box_id': 'b1',
                            'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                            'state': 'accepted',
                            'detector': REFERENCE_REGION_DETECTOR_MODEL,
                            'score': 0.9,
                        }
                    ],
                },
            }
        }
    )
    client = _client(fake)
    resp = client.get(
        '/curation/projects/default/regions/training_candidates',
        params={'mode': 'low_conf_correct', 'page_size': 50},
    )
    assert resp.status_code == 200, resp.text
    assert {i['crop_id'] for i in resp.json()['items']} == {'accepted-low-conf'}
