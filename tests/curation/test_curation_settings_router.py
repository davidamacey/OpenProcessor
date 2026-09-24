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


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """F-28.1's settings-doc cache is module-level and keyed by index name
    -- every test in this file shares the default index, and a fresh
    ``FakeSettingsOpenSearch`` per test must never see a prior test's
    cached value."""
    from src.clients import curation_opensearch

    curation_opensearch._settings_cache.clear()
    yield
    curation_opensearch._settings_cache.clear()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as legacy_router

    fake_os = FakeSettingsOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    app.include_router(legacy_router)
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


def test_put_null_clears_a_previously_pinned_axis(app_client: TestClient) -> None:
    """Raised by the Cropwright settings-UI pass: once an axis is pinned,
    there was no way to express "go back to no shared override" -- every
    value had to be a currently-advertised id. A null value must clear it
    without touching other axes, and the cleared axis must then be absent
    from GET (not merely a no-op that keeps the stale value around)."""
    app_client.put(
        '/curation/settings', json={'defaults': {'cluster': 'ahc', 'sort': 'atypicality'}}
    )

    r = app_client.put('/curation/settings', json={'defaults': {'sort': None}})
    assert r.status_code == 200, r.text
    assert r.json()['defaults'] == {'cluster': 'ahc'}

    r2 = app_client.get('/curation/settings')
    assert r2.json()['defaults'] == {'cluster': 'ahc'}
    assert 'sort' not in r2.json()['defaults']


def test_put_null_for_an_axis_with_no_prior_override_is_a_harmless_no_op(
    app_client: TestClient,
) -> None:
    r = app_client.put('/curation/settings', json={'defaults': {'cluster': None}})
    assert r.status_code == 200, r.text
    assert r.json()['defaults'] == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


@pytest.mark.usefixtures('reference_region_profile')
def test_put_rejects_detection_profile_as_read_only(app_client: TestClient) -> None:
    """detection_profile is reported on GET /methods but not settable: the
    region cascade runs on OP_REGION_PROFILE, so a stored default would be
    a silent no-op."""
    r = app_client.put(
        '/curation/settings', json={'defaults': {'detection_profile': 'license_plate'}}
    )
    assert r.status_code == 422
    assert 'detection_profile' in r.json()['detail']


@pytest.mark.usefixtures('reference_region_profile')
def test_methods_marks_settable_axes_and_reports_active_region_profile(
    app_client: TestClient,
) -> None:
    from src.clients.curation_opensearch import CURATION_SETTINGS_DOC_ID
    from src.config import get_curation_config

    # A detection_profile override stored before the axis became read-only
    # must not change what is reported as active.
    app_client.fake_os._docs[  # type: ignore[attr-defined]
        (get_curation_config().settings_index, CURATION_SETTINGS_DOC_ID)
    ] = {'defaults': {'detection_profile': 'something_else'}}
    r = app_client.get('/curation/methods')
    assert r.status_code == 200
    entries = r.json()['strategies']
    by_axis: dict[str, set[bool]] = {}
    for e in entries:
        by_axis.setdefault(e['axis'], set()).add(e['settable'])
    assert by_axis['detection_profile'] == {False}
    assert by_axis['prompt_pack'] == {True}
    assert by_axis['cluster'] == {True}
    assert by_axis['score'] == {False}
    region = [e for e in entries if e['axis'] == 'detection_profile']
    assert [(e['id'], e['default']) for e in region] == [('license_plate', True)]
