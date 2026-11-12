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
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation.training_cohorts import (
    LOW_CONFIDENCE_MAX,
    REGION_LOW_SCORE_MAX,
    TRAINING_CANDIDATE_MODES,
    cohort_catalog,
)


@pytest.fixture
def client() -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: (
        object()
    )  # FastAPI parses a class here as a dependency with parameters
    with TestClient(app) as c:
        yield c


def _by_id(client: TestClient, **params: Any) -> dict[str, dict[str, Any]]:
    r = client.get('/curation/projects/default/training_cohorts', params=params)
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
            'count',
        }
    return {c['id']: c for c in cohorts}


@pytest.fixture
def counts(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    counter = AsyncMock(return_value=7)
    monkeypatch.setattr('src.routers.curation.cohorts.cohort_count', counter)
    return counter


def test_each_cohort_serves_its_count_for_the_class(client: TestClient, counts: AsyncMock) -> None:
    cohorts = _by_id(client, class_id=4)
    assert {c['count'] for c in cohorts.values()} == {7}
    assert {call.args[2] for call in counts.await_args_list} == {4}
    assert counts.await_count == len(cohorts)


def test_no_class_means_nocounts(client: TestClient, counts: AsyncMock) -> None:
    assert {c['count'] for c in _by_id(client).values()} == {None}
    counts.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_crop_cohortcounts_through_the_crops_listing_query() -> None:
    from src.routers.curation.cohorts import cohort_count

    seen: dict[str, Any] = {}

    class Fake:
        async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, int]:
            seen['body'] = body
            return {'count': 12}

    entry = next(c for c in cohort_catalog(4) if c['id'] == 'validated')
    assert await cohort_count(Fake(), entry, 4) == 12
    text = str(seen['body'])
    assert "'class_id': 4" in text
    assert "'class_validated': True" in text


@pytest.mark.asyncio
async def test_a_region_cohortcounts_through_the_training_candidates_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation.cohorts import cohort_count

    page = AsyncMock(return_value={'total': 5})
    monkeypatch.setattr('src.routers.curation.cohorts.training_candidates', page)
    monkeypatch.setattr('src.services.curation.training_cohorts.get_active_region_profile', object)
    entry = next(c for c in cohort_catalog(4) if c['id'] == 'disagreement')
    assert await cohort_count(object(), entry, 4) == 5
    assert page.await_args is not None
    kwargs = page.await_args.kwargs
    assert (kwargs['mode'], kwargs['limit'], kwargs['class_id']) == ('disagreement', 1, 4)


@pytest.mark.usefixtures('counts')
def test_core_cohorts_carry_backend_cutoffs(client: TestClient) -> None:
    cohorts = _by_id(client, class_id=4)
    assert {'validated', 'needs_labeling', 'low_confidence', 'model_disagreements'} <= set(cohorts)
    low = cohorts['low_confidence']
    assert low['cutoffs'] == {'classifier_conf_lt': LOW_CONFIDENCE_MAX}
    assert low['endpoint'] == '/crops'
    assert low['params'] == {'class_id': 4, 'classifier_conf_lt': LOW_CONFIDENCE_MAX}
    assert cohorts['validated']['params'] == {'class_id': 4, 'label_validated': True}
    assert cohorts['model_disagreements']['endpoint'] == '/crops'
    assert cohorts['model_disagreements']['params'] == {
        'class_id': 4,
        'review_tab': 'model_disagreements',
    }


def test_low_confidence_cutoff_is_the_review_band() -> None:
    from src.services.curation import review_queries

    must, _mn, _r = review_queries.build_tab_query(
        'primary_low_conf',
        vector_field='pe_embedding',
        include_test=False,
        text=None,
        max_rank=None,
    )
    assert f"'lt': {LOW_CONFIDENCE_MAX}" in str(must)


@pytest.mark.usefixtures('counts')
def test_region_cohorts_only_with_a_region_profile(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'src.services.curation.training_cohorts.get_active_region_profile', lambda: None
    )
    assert not set(_by_id(client)) & set(TRAINING_CANDIDATE_MODES)

    monkeypatch.setattr('src.services.curation.training_cohorts.get_active_region_profile', object)
    cohorts = _by_id(client)
    for mode in TRAINING_CANDIDATE_MODES:
        assert cohorts[mode]['endpoint'] == '/regions/training_candidates'
        assert cohorts[mode]['params'] == {'mode': mode}
        assert cohorts[mode]['row_kind'] == 'region'
    assert cohorts['low_conf_correct']['cutoffs'] == {'region_score_lt': REGION_LOW_SCORE_MAX}


def test_region_cohort_description_is_the_selection_reason() -> None:
    from src.routers.curation.regions import _training_candidate_query

    for mode, spec in TRAINING_CANDIDATE_MODES.items():
        _query, _box, reason = _training_candidate_query(mode)
        assert reason == spec.description
