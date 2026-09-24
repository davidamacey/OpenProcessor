"""``GET /training_cohorts`` serves the training-cohort catalog.

Each cohort says what it is (``label``, ``description``), which backend
cut-offs define it (``cutoffs``) and how to fetch its rows (``endpoint`` +
``params``, ``row_kind``), so the frontend keeps no cohort definitions,
descriptions or thresholds of its own. Region cohorts are the
``/regions/training_candidates`` modes and only appear when a region
profile is configured; their descriptions are the same strings that
endpoint returns as ``selection_reason``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation.training_cohorts import (
    LOW_CONFIDENCE_MAX,
    REGION_LOW_SCORE_MAX,
    TRAINING_CANDIDATE_MODES,
)


@pytest.fixture
def client() -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: object()
    with TestClient(app) as c:
        yield c


def _by_id(client: TestClient, **params: Any) -> dict[str, dict[str, Any]]:
    r = client.get('/curation/training_cohorts', params=params)
    assert r.status_code == 200, r.text
    cohorts = r.json()['cohorts']
    for c in cohorts:
        assert set(c) == {
            'id',
            'label',
            'description',
            'cutoffs',
            'endpoint',
            'params',
            'row_kind',
        }
    return {c['id']: c for c in cohorts}


def test_core_cohorts_carry_backend_cutoffs(client: TestClient) -> None:
    cohorts = _by_id(client, class_id=4)
    assert {'validated', 'needs_labeling', 'low_confidence', 'model_disagreements'} <= set(cohorts)
    low = cohorts['low_confidence']
    assert low['cutoffs'] == {'classifier_conf_lt': LOW_CONFIDENCE_MAX}
    assert low['endpoint'] == '/crops'
    assert low['params'] == {'class_id': 4, 'classifier_conf_lt': LOW_CONFIDENCE_MAX}
    assert cohorts['validated']['params'] == {'class_id': 4, 'label_validated': True}
    assert cohorts['model_disagreements']['endpoint'] == '/review/model_disagreements'


def test_low_confidence_cutoff_is_the_review_band() -> None:
    from src.services.curation import review_queries

    must, _mn, _r = review_queries.build_tab_query(
        'primary_low_conf', include_test=False, text=None, max_rank=None
    )
    assert f"'lt': {LOW_CONFIDENCE_MAX}" in str(must)


def test_region_cohorts_only_with_a_region_profile(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'src.services.curation.training_cohorts.get_active_region_profile', lambda: None
    )
    assert not set(_by_id(client)) & set(TRAINING_CANDIDATE_MODES)

    monkeypatch.setattr(
        'src.services.curation.training_cohorts.get_active_region_profile', lambda: object()
    )
    cohorts = _by_id(client)
    for mode in TRAINING_CANDIDATE_MODES:
        assert cohorts[mode]['endpoint'] == '/regions/training_candidates'
        assert cohorts[mode]['params'] == {'mode': mode}
        assert cohorts[mode]['row_kind'] == 'region'
    assert cohorts['low_conf_correct']['cutoffs'] == {'region_score_lt': REGION_LOW_SCORE_MAX}


def test_region_cohort_description_is_the_selection_reason() -> None:
    from src.routers.curation.regions import _training_candidate_query

    for mode, spec in TRAINING_CANDIDATE_MODES.items():
        _query, reason = _training_candidate_query(mode)
        assert reason == spec.description
