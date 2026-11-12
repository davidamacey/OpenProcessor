"""POST /classes refuses a bad name with the slug the user most likely meant."""

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


def test_traffic_light_is_refused_with_traffic_light_suggested(client: TestClient) -> None:
    resp = _create(client, 'Traffic light')
    assert resp.status_code == 422, resp.text
    detail = resp.json()['detail']
    assert detail['error'] == 'invalid_class_name'
    assert detail['suggested_name'] == 'traffic_light'
    assert detail['message']


def test_punctuation_is_dropped_from_the_suggestion(client: TestClient) -> None:
    resp = _create(client, 'Stop-sign!')
    assert resp.status_code == 422, resp.text
    assert resp.json()['detail']['suggested_name'] == 'stop_sign'


def test_a_name_with_nothing_usable_has_no_suggestion(client: TestClient) -> None:
    resp = _create(client, '***')
    assert resp.status_code == 422, resp.text
    detail = resp.json()['detail']
    assert detail['error'] == 'invalid_class_name'
    assert 'suggested_name' not in detail


def test_a_valid_slug_is_created(client: TestClient) -> None:
    resp = _create(client, 'traffic_light')
    assert resp.status_code == 201, resp.text
