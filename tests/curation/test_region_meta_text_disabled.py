"""Text-free region profile: a per-box ``text`` edit is rejected (422
``region_text_disabled``) before anything is written; other region
metadata still patches.

``region_text`` moved off ``PATCH /crops/{id}/region_meta`` in W8 (D
decision, 2026-09-26) onto the per-box routes -- this test now exercises
``PATCH /crops/{crop_id}/regions/{box_id}`` instead of the old item-level
``region_text`` key (``region_meta`` no longer accepts it at all;
``extra='forbid'`` 422s a stale client that still sends it)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import get_region_fields

from .test_regions_router import _FakeRegionOS


F = get_region_fields()


@pytest.fixture
def text_free(monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.services.detection import profile_registry

    monkeypatch.setenv('OP_REGION_DETECTION_TEXT_READER', 'none')
    profile_registry._reset_registry_for_tests()
    yield
    profile_registry._reset_registry_for_tests()


@pytest.fixture
def fake_os() -> _FakeRegionOS:
    return _FakeRegionOS(
        {
            'crop-1': {
                'crop_id': 'crop-1',
                F.status: 'detected',
                F.boxes: [{'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'accepted'}],
                F.count: 1,
            }
        }
    )


@pytest.fixture
def app_client(fake_os: _FakeRegionOS, text_free: Any) -> Any:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as client:
        yield client


def test_region_meta_rejects_the_removed_region_text_key(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    """``region_text`` is removed from ``ItemRegionMetaRequest`` entirely
    (D decision) -- ``extra='forbid'`` 422s a stale client regardless of
    whether the profile reads text."""
    before = dict(fake_os._docs['crop-1'])
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/region_meta',
        json={
            'region_text': 'ABC1234',
            'region_status': 'detected',
            'region_label_source': 'human',
        },
    )
    assert resp.status_code == 422, resp.text
    assert fake_os.update_calls == []
    assert fake_os.bulk_calls == []
    assert fake_os._docs['crop-1'] == before


def test_per_box_text_edit_is_rejected_without_a_write(
    app_client: TestClient, fake_os: _FakeRegionOS
) -> None:
    before = dict(fake_os._docs['crop-1'])
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/regions/b1',
        json={'text': 'ABC1234', 'region_label_source': 'human'},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()['detail'] == {'error': 'region_text_disabled'}
    assert fake_os.update_calls == []
    assert fake_os.bulk_calls == []
    assert fake_os._docs['crop-1'] == before


def test_other_region_meta_still_patches(app_client: TestClient, fake_os: _FakeRegionOS) -> None:
    resp = app_client.patch(
        '/curation/projects/default/crops/crop-1/region_meta',
        json={'region_status': 'verify_rejected', 'region_label_source': 'human'},
    )
    assert resp.status_code == 200, resp.text
    assert fake_os._docs['crop-1'][F.status] == 'verify_rejected'
