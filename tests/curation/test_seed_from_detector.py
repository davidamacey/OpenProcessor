"""``POST /classes/seed_from_detector``: seed the registry from the detector's labels."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.clients.curation_opensearch import ClassRegistry


URL = '/curation/projects/default/classes/seed_from_detector'
CONFIG_URL = '/curation/projects/default/ingest/config'


@pytest.fixture
def registry(tmp_path: Any) -> ClassRegistry:
    return ClassRegistry(path=tmp_path / 'class_registry.json')


@pytest.fixture
def labels_file(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'labels.txt'
    path.write_text('person\ncar\ntraffic light\nhot dog\n')
    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'fake_detector')
    monkeypatch.setenv('OP_INGEST_PRIMARY_LABELS_PATH', str(path))


@pytest.fixture
def client(registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    fake = AsyncMock()
    fake.get.side_effect = RuntimeError('404 not_found')  # no settings document
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def _names(registry: ClassRegistry) -> list[str]:
    return [c.class_name for c in registry.load().classes]


def test_dry_run_is_the_default_and_writes_nothing(
    client: TestClient, registry: ClassRegistry, labels_file: None
) -> None:
    r = client.post(URL, json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['dry_run'] is True
    assert [c['name'] for c in body['created']] == [
        'person',
        'car',
        'traffic_light',
        'hot_dog',
    ]
    assert all(c['class_id'] is None for c in body['created'])
    assert _names(registry) == []


def test_apply_creates_slugs_in_group_detector_and_is_idempotent(
    client: TestClient, registry: ClassRegistry, labels_file: None
) -> None:
    r = client.post(URL, json={'dry_run': False})
    assert r.status_code == 200, r.text
    assert [c['class_id'] for c in r.json()['created']] == [0, 1, 2, 3]
    assert _names(registry) == ['person', 'car', 'traffic_light', 'hot_dog']
    assert {c.group for c in registry.load().classes} == {'detector'}
    again = client.post(URL, json={'dry_run': False}).json()
    assert again['created'] == []
    assert {s['reason'] for s in again['skipped']} == {'exists'}
    assert len(registry.load().classes) == 4


def test_existing_hand_made_class_is_skipped_and_ids_append(
    client: TestClient, registry: ClassRegistry, labels_file: None
) -> None:
    registry.add_class('car', group='vehicle')
    body = client.post(URL, json={'dry_run': False}).json()
    assert [s['name'] for s in body['skipped']] == ['car']
    assert [c['class_id'] for c in body['created']] == [1, 2, 3]
    assert next(c for c in registry.load().classes if c.class_name == 'car').group == 'vehicle'


def test_names_subset_and_group(
    client: TestClient, registry: ClassRegistry, labels_file: None
) -> None:
    body = client.post(
        URL, json={'names': ['traffic light'], 'group': 'street', 'dry_run': False}
    ).json()
    assert [c['name'] for c in body['created']] == ['traffic_light']
    assert [(c.class_name, c.group) for c in registry.load().classes] == [
        ('traffic_light', 'street')
    ]


def test_unknown_name_is_422_and_writes_nothing(
    client: TestClient, registry: ClassRegistry, labels_file: None
) -> None:
    r = client.post(URL, json={'names': ['person', 'unicorn'], 'dry_run': False})
    assert r.status_code == 422
    assert 'unicorn' in r.text
    assert _names(registry) == []


def test_extra_fields_are_rejected(client: TestClient, labels_file: None) -> None:
    assert client.post(URL, json={'bogus': 1}).status_code == 422


def test_no_detector_configured_is_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', raising=False)
    assert client.post(URL, json={}).status_code == 503


def test_seeding_never_touches_items(
    client: TestClient, registry: ClassRegistry, labels_file: None
) -> None:
    from src.routers.curation import _raw_opensearch_dep

    fake = client.app.dependency_overrides[_raw_opensearch_dep]()
    client.post(URL, json={'dry_run': False})
    assert fake.method_calls == []


def test_detector_block_in_ingest_config(client: TestClient, labels_file: None) -> None:
    det = client.get(CONFIG_URL).json()['detector']
    assert det['model'] == 'fake_detector'
    assert det['assigns_class'] is False
    assert det['confidence_floor_applies'] is False
    assert det['n_labels'] == 4
    assert 'class_ids_filter' not in det
    assert det['labels'][2] == {'class_id': 2, 'name': 'traffic light', 'slug': 'traffic_light'}


def test_detector_block_confidence_floor_follows_assigns_class(
    client: TestClient, labels_file: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_INGEST_PRIMARY_ASSIGNS_CLASS', 'true')
    det = client.get(CONFIG_URL).json()['detector']
    assert det['confidence_floor_applies'] is True


def test_detector_block_is_null_without_a_detector(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', raising=False)
    assert client.get(CONFIG_URL).json()['detector'] is None
