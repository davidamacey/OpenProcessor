"""Paging depth guard + stable ``crop_id`` sort tiebreaker for
``GET /regions`` and ``GET /regions/training_candidates``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, get_curation_config, get_region_fields, index_name


F = get_region_fields()
CFG = get_curation_config()
ITEMS = index_name(CFG, IndexRole.ITEMS)

# GET /regions requires an active region profile (no-profile gating contract).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


def _client(fake: Any) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _fake() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {ITEMS: {'r1': {'crop_id': 'r1', F.bbox_norm: [0.1, 0.1, 0.2, 0.2]}}}
    )


def test_list_regions_page_too_deep_is_422() -> None:
    fake = _fake()
    client = _client(fake)
    resp = client.get('/curation/regions', params={'page': 400, 'page_size': 30})
    assert resp.status_code == 422, resp.text


def test_list_regions_page_within_window_is_fine() -> None:
    fake = _fake()
    client = _client(fake)
    resp = client.get('/curation/regions', params={'page': 300, 'page_size': 30})
    assert resp.status_code == 200, resp.text


def test_training_candidates_page_too_deep_is_422() -> None:
    fake = _fake()
    client = _client(fake)
    resp = client.get(
        '/curation/regions/training_candidates',
        params={'mode': 'human_corrected', 'page': 400, 'page_size': 30},
    )
    assert resp.status_code == 422, resp.text


def test_list_regions_request_body_carries_crop_id_tiebreaker() -> None:
    """Direct assertion on the actual request body sent to OpenSearch."""
    from unittest.mock import AsyncMock

    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'hits': {'total': {'value': 0}, 'hits': []}})
    client = _client(fake)

    resp = client.get('/curation/regions')
    assert resp.status_code == 200, resp.text
    sort = fake.search.call_args.kwargs['body']['sort']
    assert sort[-1] == {'crop_id': {'order': 'asc'}}


def test_training_candidates_request_body_carries_crop_id_tiebreaker() -> None:
    from unittest.mock import AsyncMock

    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'hits': {'total': {'value': 0}, 'hits': []}})
    client = _client(fake)

    resp = client.get('/curation/regions/training_candidates', params={'mode': 'human_corrected'})
    assert resp.status_code == 200, resp.text
    sort = fake.search.call_args.kwargs['body']['sort']
    assert sort[-1] == {'crop_id': {'order': 'asc'}}
