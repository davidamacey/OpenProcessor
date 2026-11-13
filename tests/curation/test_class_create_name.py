"""POST /classes accepts a typed display name and stores the registry slug."""

from __future__ import annotations

from typing import Any

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import router as curation_router


@pytest.fixture
def client() -> Any:
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    with TestClient(app) as c:
        yield c


def _create(client: TestClient, name: str) -> Any:
    return client.post('/curation/projects/default/classes', json={'name': name})


@pytest.mark.parametrize(
    ('typed', 'slug'),
    [
        ('Traffic light', 'traffic_light'),
        ('Stop-sign!', 'stop_sign'),
        ('traffic_light', 'traffic_light'),
    ],
)
def test_a_typed_name_is_created_under_its_slug(client: TestClient, typed: str, slug: str) -> None:
    resp = _create(client, typed)
    assert resp.status_code == 201, resp.text
    assert resp.json()['class_name'] == slug


def test_a_name_with_nothing_usable_is_refused(client: TestClient) -> None:
    resp = _create(client, '***')
    assert resp.status_code == 422, resp.text
    detail = resp.json()['detail']
    assert detail['error'] == 'invalid_class_name'
    assert 'suggested_name' not in detail
