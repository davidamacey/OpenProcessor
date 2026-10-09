"""Tests for ``GET,PUT /curation/settings`` -- CRUD + validation.

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

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.test_curation_settings_client import FakeSettingsOpenSearch


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """The settings-doc cache is module-level and keyed by index name
    -- every test in this file shares the default index, and a fresh
    ``FakeSettingsOpenSearch`` per test must never see a prior test's
    cached value."""
    from src.clients.curation_opensearch import settings_doc

    settings_doc._settings_cache.clear()
    yield
    settings_doc._settings_cache.clear()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = FakeSettingsOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def test_get_with_no_doc_yet_returns_empty_defaults(app_client: TestClient) -> None:
    r = app_client.get('/curation/projects/default/settings')
    assert r.status_code == 200
    body = r.json()
    links = body.pop('resource_links')
    assert [lk['id'] for lk in links][:3] == ['swagger', 'redoc', 'openapi_json']
    assert body == {'defaults': {}, 'updated_at': None, 'updated_by': None}


def test_put_creates_the_document(app_client: TestClient) -> None:
    r = app_client.put('/curation/projects/default/settings', json={'defaults': {'cluster': 'ahc'}})
    assert r.status_code == 200
    body = r.json()
    assert body['defaults'] == {'cluster': 'ahc'}
    assert body['updated_at'] is not None
    assert body['updated_by'] is None

    r2 = app_client.get('/curation/projects/default/settings')
    assert r2.json()['defaults'] == {'cluster': 'ahc'}


def test_put_again_partially_updates_without_clobbering_other_axes(app_client: TestClient) -> None:
    app_client.put(
        '/curation/projects/default/settings',
        json={'defaults': {'cluster': 'ahc', 'sort': 'atypicality'}},
    )
    r = app_client.put('/curation/projects/default/settings', json={'defaults': {'cluster': 'ivf'}})
    assert r.status_code == 200
    assert r.json()['defaults'] == {'cluster': 'ivf', 'sort': 'atypicality'}


def test_put_with_invalid_axis_returns_422_listing_valid_axes(app_client: TestClient) -> None:
    r = app_client.put(
        '/curation/projects/default/settings', json={'defaults': {'not_a_real_axis': 'whatever'}}
    )
    assert r.status_code == 422
    detail = r.json()['detail']
    assert 'not_a_real_axis' in detail
    assert 'cluster' in detail  # one of the valid axes is named in the message


def test_put_with_invalid_id_for_a_valid_axis_returns_422_listing_valid_ids(
    app_client: TestClient,
) -> None:
    r = app_client.put(
        '/curation/projects/default/settings', json={'defaults': {'cluster': 'not_a_real_method'}}
    )
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
    r = app_client.put(
        '/curation/projects/default/settings', json={'defaults': {'score': 'mistakenness'}}
    )
    assert r.status_code == 422
    assert 'score' in r.json()['detail']


def test_put_null_clears_a_previously_pinned_axis(app_client: TestClient) -> None:
    """Raised by the Cropwright settings-UI pass: once an axis is pinned,
    there was no way to express "go back to no shared override" -- every
    value had to be a currently-advertised id. A null value must clear it
    without touching other axes, and the cleared axis must then be absent
    from GET (not merely a no-op that keeps the stale value around)."""
    app_client.put(
        '/curation/projects/default/settings',
        json={'defaults': {'cluster': 'ahc', 'sort': 'atypicality'}},
    )

    r = app_client.put('/curation/projects/default/settings', json={'defaults': {'sort': None}})
    assert r.status_code == 200, r.text
    assert r.json()['defaults'] == {'cluster': 'ahc'}

    r2 = app_client.get('/curation/projects/default/settings')
    assert r2.json()['defaults'] == {'cluster': 'ahc'}
    assert 'sort' not in r2.json()['defaults']


def test_put_null_for_an_axis_with_no_prior_override_is_a_harmless_no_op(
    app_client: TestClient,
) -> None:
    r = app_client.put('/curation/projects/default/settings', json={'defaults': {'cluster': None}})
    assert r.status_code == 200, r.text
    assert r.json()['defaults'] == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


@pytest.fixture
def app_client_config_store(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A ``FakeConfigOpenSearch``-backed client -- unlike
    ``FakeSettingsOpenSearch``, this fake also serves the config store's
    ``get``/``index``/``search`` calls, needed now that
    ``detection_profile``/``prompt_pack`` PUTs delegate to
    ``store.activate_axis`` (W2)."""
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = FakeConfigOpenSearch()

    # N1 fix (W3/W4 round-3 review): `PUT /settings` now runs the same
    # `for_activation` validation `POST /{name}/activate` runs, so a
    # `detection_profile` PUT needs its models reported READY the same
    # way `test_region_profiles_router.py`'s `app_client` fixture already
    # does for the direct activate route -- 'license_plate_detector' is
    # `reference_region_profile`'s configured detector model.
    async def _fake_repo_index() -> list[dict]:
        return [
            {'name': 'license_plate_detector', 'state': 'READY', 'version': '1'},
            {'name': 'paddleocr_det_trt', 'state': 'READY', 'version': '1'},
            {'name': 'paddleocr_rec_trt', 'state': 'READY', 'version': '1'},
            {'name': 'ocr_pipeline', 'state': 'READY', 'version': '1'},
        ]

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    from src.services.triton_control import TritonControlService

    monkeypatch.setattr(
        TritonControlService, 'get_repository_index', lambda _self: _fake_repo_index()
    )

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


@pytest.fixture(autouse=True)
def _reset_config_store() -> Iterator[None]:
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.mark.usefixtures('reference_region_profile')
def test_put_detection_profile_now_settable_through_the_store(
    app_client_config_store: TestClient,
) -> None:
    """W2: detection_profile joined the config-store-backed axes (it was
    read-only pre-W2, when the region cascade only read ``OP_REGION_PROFILE``
    at process start; the detection worker now hot-reloads it, §4.5)."""
    r = app_client_config_store.put(
        '/curation/projects/default/settings',
        json={'defaults': {'detection_profile': 'license_plate'}},
    )
    assert r.status_code == 200, r.text
    assert r.json()['defaults']['detection_profile'] == 'license_plate'

    r_unknown = app_client_config_store.put(
        '/curation/projects/default/settings',
        json={'defaults': {'detection_profile': 'not_a_real_profile'}},
    )
    assert r_unknown.status_code == 422
    assert r_unknown.json()['detail']['error'] == 'unknown_profile'


@pytest.mark.usefixtures('reference_region_profile')
def test_methods_marks_settable_axes_and_reports_active_region_profile(
    app_client_config_store: TestClient,
) -> None:
    r = app_client_config_store.put(
        '/curation/projects/default/settings',
        json={'defaults': {'detection_profile': 'license_plate'}},
    )
    assert r.status_code == 200, r.text

    r2 = app_client_config_store.get('/curation/projects/default/methods')
    assert r2.status_code == 200
    entries = r2.json()['strategies']
    by_axis: dict[str, set[bool]] = {}
    for e in entries:
        by_axis.setdefault(e['axis'], set()).add(e['settable'])
    assert by_axis['detection_profile'] == {True}
    assert by_axis['prompt_pack'] == {True}
    assert by_axis['cluster'] == {True}
    assert by_axis['score'] == {False}
    region = [e for e in entries if e['axis'] == 'detection_profile']
    assert ('license_plate', True) in [(e['id'], e['default']) for e in region]
