"""Tests for ``GET /curation/review/{tab}`` Phase 3 additions (curation-strategy
plan §7 Phase 3): the ``sort``/``min_mistakenness``/``hide_near_duplicates``
query params, the additive ``sort_applied``/``sort_fallback_reason``
response fields, and the golden-body regression guard for all 9 tabs with
``?sort`` absent (see ``test_review_disagreements.py`` for the
model_disagreements-specific tab tests this file doesn't duplicate).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation.review_request import TIEBREAK


ALL_TABS = (
    'all',
    'mismatches',
    'vlm_low_conf',
    'outliers',
    'uncertainty',
    'model_disagreements',
    'regions',
    'primary_low_conf',
    'classifier_blind_spots',
)

LEGACY_SORT_CLAUSE = {
    'all': [{'cluster_distance': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}}],
    'mismatches': [{'updated_at': {'order': 'desc'}}],
    'vlm_low_conf': [{'updated_at': {'order': 'desc'}}],
    'outliers': [
        {'cluster_distance': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}}
    ],
    'uncertainty': [
        {'probe_pred_entropy': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}}
    ],
    'model_disagreements': [
        {'probe_pred_entropy': {'order': 'asc', 'missing': '_last', 'unmapped_type': 'double'}}
    ],
    'regions': [
        {'region_score': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
        {
            'region_candidate_score': {
                'order': 'desc',
                'missing': '_last',
                'unmapped_type': 'double',
            }
        },
    ],
    'primary_low_conf': [
        {'crop_area_norm': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
        {
            'confidence': {
                'order': 'asc',
                'missing': '_last',
                'unmapped_type': 'double',
            }
        },
    ],
    'classifier_blind_spots': [
        {'crop_area_norm': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
        {'confidence': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
    ],
}

EXPECTED_DEFAULT_SORT_ID = {
    'all': 'atypicality',
    'mismatches': 'recent',
    'vlm_low_conf': 'recent',
    'outliers': 'atypicality',
    'uncertainty': 'uncertainty_entropy',
    'model_disagreements': 'disagreement_entropy_asc',
    'regions': 'region_score',
    'primary_low_conf': 'primary_low_conf_default',
    'classifier_blind_spots': 'classifier_blind_spots_default',
}


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = AsyncMock()
    fake_os.search = AsyncMock(return_value={'hits': {'total': {'value': 0}, 'hits': []}})
    # C3: a zero-result tab computes empty_reason via a couple of `count`
    # calls; default to "nothing scored yet" for every field they check.
    fake_os.count = AsyncMock(return_value={'count': 0})
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.delenv('OP_SCORES_ENABLED', raising=False)
    monkeypatch.delenv('OP_SCORES_SHADOW', raising=False)

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os

    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


@pytest.mark.parametrize('tab', ALL_TABS)
def test_sort_absent_is_byte_identical_to_legacy_clause(tab: str, app_client: TestClient) -> None:
    """The golden-body regression guard at the router level: every tab's
    OpenSearch request body carries the exact legacy sort clause when
    ?sort is omitted, and the response's pre-existing keys are unaffected
    by the Phase 3 envelope additions."""
    r = app_client.get(f'/curation/review/{tab}')
    assert r.status_code == 200, r.text
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    # The legacy clause, then the crop_id tiebreak every queue ends in.
    assert body['sort'] == [*LEGACY_SORT_CLAUSE[tab], TIEBREAK]

    out = r.json()
    # Pre-existing envelope keys, unchanged in shape/value.
    assert out['total'] == 0
    assert out['page'] == 1
    assert out['page_size'] == 30
    assert out['items'] == []
    # Additive-only Phase 3 keys.
    assert out['sort_applied'] == EXPECTED_DEFAULT_SORT_ID[tab]
    assert out['sort_fallback_reason'] is None


@pytest.mark.parametrize('tab', ALL_TABS)
def test_sort_default_literal_matches_absent(tab: str, app_client: TestClient) -> None:
    r = app_client.get(f'/curation/review/{tab}?sort=default')
    assert r.status_code == 200, r.text
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    # The legacy clause, then the crop_id tiebreak every queue ends in.
    assert body['sort'] == [*LEGACY_SORT_CLAUSE[tab], TIEBREAK]
    assert r.json()['sort_applied'] == EXPECTED_DEFAULT_SORT_ID[tab]


def test_explicit_stable_sort_overrides_tab_default(app_client: TestClient) -> None:
    """mismatches' legacy default is 'recent'; explicitly asking for
    'atypicality' should apply cluster_distance desc instead."""
    r = app_client.get('/curation/review/mismatches?sort=atypicality')
    assert r.status_code == 200, r.text
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    assert body['sort'] == [
        {'cluster_distance': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
        TIEBREAK,
    ]
    assert r.json()['sort_applied'] == 'atypicality'


def test_unknown_sort_returns_400(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/all?sort=not-a-real-sort')
    assert r.status_code == 400
    assert 'unknown review sort' in r.json()['detail']


def test_shadow_sort_returns_400(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/all?sort=uniqueness')
    assert r.status_code == 400
    assert 'not selectable' in r.json()['detail']


def test_disabled_sort_returns_400_when_scores_disabled(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/all?sort=mistakenness')
    assert r.status_code == 400
    assert 'not selectable' in r.json()['detail']


def test_mistakenness_sort_selectable_when_promoted(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_SCORES_ENABLED', '1')
    monkeypatch.setenv('OP_SCORES_SHADOW', '1')
    r = app_client.get('/curation/review/all?sort=mistakenness')
    assert r.status_code == 200, r.text
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    assert body['sort'] == [
        {'mistakenness_score': {'order': 'desc', 'missing': '_last', 'unmapped_type': 'double'}},
        TIEBREAK,
    ]
    assert r.json()['sort_applied'] == 'mistakenness'


def test_min_mistakenness_filter_is_null_safe(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/all?min_mistakenness=0.5')
    assert r.status_code == 200, r.text
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must = body['query']['bool']['must']
    assert {
        'bool': {
            'should': [
                {'range': {'mistakenness_score': {'gte': 0.5}}},
                {'bool': {'must_not': {'exists': {'field': 'mistakenness_score'}}}},
            ],
            'minimum_should_match': 1,
        }
    } in must


def test_min_mistakenness_absent_does_not_add_filter(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/all')
    assert r.status_code == 200
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must = body['query']['bool']['must']
    assert not any('mistakenness_score' in str(m) for m in must)


def test_hide_near_duplicates_filter_is_null_safe(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/all?hide_near_duplicates=true')
    assert r.status_code == 200, r.text
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must = body['query']['bool']['must']
    assert {
        'bool': {
            'should': [
                {'term': {'dup_is_representative': True}},
                {'bool': {'must_not': {'exists': {'field': 'dup_is_representative'}}}},
            ],
            'minimum_should_match': 1,
        }
    } in must


def test_hide_near_duplicates_false_by_default(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/all')
    assert r.status_code == 200
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must = body['query']['bool']['must']
    assert not any('dup_is_representative' in str(m) for m in must)


def test_response_item_whitelist_includes_new_score_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = AsyncMock()
    fake_os.search = AsyncMock(
        return_value={
            'hits': {
                'total': {'value': 1},
                'hits': [
                    {
                        '_id': 'crop-1',
                        '_source': {
                            'crop_id': 'crop-1',
                            'image_path': '/x.jpg',
                            'class_name': 'sedan',
                            'mistakenness_score': 0.87,
                            'uniqueness_score': 0.42,
                            'dup_group_id': 'g1',
                            'dup_group_size': 3,
                            'dup_is_representative': True,
                        },
                    }
                ],
            }
        }
    )
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)

    r = client.get('/curation/review/all')
    assert r.status_code == 200, r.text
    item = r.json()['items'][0]
    assert item['mistakenness_score'] == 0.87
    assert item['uniqueness_score'] == 0.42
    assert item['dup_group_id'] == 'g1'
    assert item['dup_group_size'] == 3
    assert item['dup_is_representative'] is True


def test_test_holdout_filter_unchanged_by_sort_params(app_client: TestClient) -> None:
    """Hard constraint: test_holdout=true filtering must be unaffected by
    any Phase 3 addition."""
    r = app_client.get('/curation/review/all?sort=recent')
    assert r.status_code == 200
    body = app_client.fake_os.search.call_args.kwargs['body']  # type: ignore[attr-defined]
    must_not = body['query']['bool']['must_not']
    assert {'term': {'test_holdout': True}} in must_not


def test_review_page_too_deep_is_422(app_client: TestClient) -> None:
    """F-7: from+size past the 10000 result-window ceiling must 422
    explicitly rather than let OpenSearch 500 past index.max_result_window."""
    r = app_client.get('/curation/review/all', params={'page': 400, 'page_size': 30})
    assert r.status_code == 422, r.text


def test_review_page_within_window_is_fine(app_client: TestClient) -> None:
    r = app_client.get('/curation/review/all', params={'page': 300, 'page_size': 30})
    assert r.status_code == 200, r.text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
