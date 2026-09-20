"""Tests for ``GET,PUT /curation/settings`` (curation deployment-settings
plan) -- CRUD + validation.

The integration proof that a PUT actually changes ``GET /methods`` AND
real endpoint behavior (not just what ``/methods`` displays) lives in
``test_curation_settings_integration.py`` -- it depends on the
``strategy_registry.py`` / ``review_sorts.py`` rewiring, which is a
separate, later commit from the endpoints this file tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_curation_settings_client import FakeSettingsOpenSearch


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_field_coverage_cache() -> Iterator[None]:
    from src.services.curation.strategy_registry import _reset_field_coverage_cache

    _reset_field_coverage_cache()
    yield
    _reset_field_coverage_cache()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as kb_router

    fake_os = FakeSettingsOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def test_get_with_no_doc_yet_returns_empty_defaults(app_client: TestClient) -> None:
    r = app_client.get('/curation/settings')
    assert r.status_code == 200
    body = r.json()
    assert body == {'defaults': {}, 'updated_at': None, 'updated_by': None}


def test_put_creates_the_document(app_client: TestClient) -> None:
    r = app_client.put('/curation/settings', json={'defaults': {'cluster': 'ahc'}})
    assert r.status_code == 200
    body = r.json()
    assert body['defaults'] == {'cluster': 'ahc'}
    assert body['updated_at'] is not None
    assert body['updated_by'] is None

    r2 = app_client.get('/curation/settings')
    assert r2.json()['defaults'] == {'cluster': 'ahc'}


def test_put_again_partially_updates_without_clobbering_other_axes(app_client: TestClient) -> None:
    app_client.put(
        '/curation/settings', json={'defaults': {'cluster': 'ahc', 'sort': 'atypicality'}}
    )
    r = app_client.put('/curation/settings', json={'defaults': {'cluster': 'ivf'}})
    assert r.status_code == 200
    assert r.json()['defaults'] == {'cluster': 'ivf', 'sort': 'atypicality'}


def test_put_with_invalid_axis_returns_422_listing_valid_axes(app_client: TestClient) -> None:
    r = app_client.put('/curation/settings', json={'defaults': {'not_a_real_axis': 'whatever'}})
    assert r.status_code == 422
    detail = r.json()['detail']
    assert 'not_a_real_axis' in detail
    assert 'cluster' in detail  # one of the valid axes is named in the message


def test_put_with_invalid_id_for_a_valid_axis_returns_422_listing_valid_ids(
    app_client: TestClient,
) -> None:
    r = app_client.put('/curation/settings', json={'defaults': {'cluster': 'not_a_real_method'}})
    assert r.status_code == 422
    detail = r.json()['detail']
    assert 'not_a_real_method' in detail
    assert 'ivf' in detail  # a real, currently-advertised cluster id


def test_put_rejects_an_axis_methods_advertises_but_has_no_settable_default(
    app_client: TestClient,
) -> None:
    """'score' is a real GET /methods axis but has no single-selectable-id
    default concept (scorers are additive, not mutually exclusive) --
    PUT must reject it rather than silently storing a dead value."""
    r = app_client.put('/curation/settings', json={'defaults': {'score': 'mistakenness'}})
    assert r.status_code == 422
    assert 'score' in r.json()['detail']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
