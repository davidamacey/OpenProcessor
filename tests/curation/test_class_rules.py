"""Class-registry rules the backend enforces and serves.

- Class names match ``^[a-z0-9_]+$`` on create and rename (422 otherwise).
- Hotkeys: one character, not reserved, unique among active classes — on
  create as well as update. ``GET /classes`` serves ``reserved_hotkeys``.
- ``POST /classes/merge?dry_run=true`` reports what a merge would do
  without doing it.
- Class entries carry ``added_at`` from the registry.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.clients.curation_opensearch import ClassRegistry
from src.routers.curation.classes import RESERVED_HOTKEY_LETTERS


def _entry(registry: ClassRegistry, class_id: int) -> Any:
    entry = registry.get(class_id)
    assert entry is not None
    return entry


@pytest.fixture
def registry(tmp_path: Any) -> ClassRegistry:
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('sedan', group='vehicle')
    reg.add_class('suv', group='vehicle')
    return reg


@pytest.fixture
def fake_os() -> AsyncMock:
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'aggregations': {}})
    fake.count = AsyncMock(return_value={'count': 0})
    return fake


@pytest.fixture
def client(registry: ClassRegistry, fake_os: AsyncMock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize('name', ['Pickup Truck', 'pickup-truck', 'SUV', '', 'bus!'])
def test_create_rejects_non_slug_names(
    client: TestClient, registry: ClassRegistry, name: str
) -> None:
    r = client.post('/curation/classes', json={'name': name})
    assert r.status_code == 422, r.text
    assert len(registry.load().classes) == 2


def test_rename_rejects_non_slug_names(client: TestClient, registry: ClassRegistry) -> None:
    r = client.put('/curation/classes/0', json={'name': 'Sedan Car'})
    assert r.status_code == 422, r.text
    assert _entry(registry, 0).class_name == 'sedan'


def test_create_and_rename_accept_slug_names(client: TestClient) -> None:
    assert client.post('/curation/classes', json={'name': 'box_truck2'}).status_code == 201
    assert client.put('/curation/classes/0', json={'name': 'sedan_car'}).status_code == 200


def test_create_with_hotkey(client: TestClient, registry: ClassRegistry) -> None:
    r = client.post('/curation/classes', json={'name': 'van', 'hotkey_letter': 'V'})
    assert r.status_code == 201, r.text
    assert _entry(registry, r.json()['class_id']).hotkey_letter == 'v'


@pytest.mark.parametrize('letter', ['d', '/', 'b'])
def test_create_rejects_reserved_hotkey_without_creating(
    client: TestClient, registry: ClassRegistry, letter: str
) -> None:
    r = client.post('/curation/classes', json={'name': 'van', 'hotkey_letter': letter})
    assert r.status_code == 422, r.text
    assert len(registry.load().classes) == 2


def test_create_rejects_duplicate_hotkey(client: TestClient, registry: ClassRegistry) -> None:
    assert client.put('/curation/classes/0', json={'hotkey_letter': 's'}).status_code == 200
    r = client.post('/curation/classes', json={'name': 'van', 'hotkey_letter': 's'})
    assert r.status_code == 409, r.text
    assert len(registry.load().classes) == 2


def test_list_serves_reserved_hotkeys_and_added_at(client: TestClient) -> None:
    body = client.get('/curation/classes').json()
    assert body['reserved_hotkeys'] == sorted(RESERVED_HOTKEY_LETTERS)
    assert all(c['added_at'] for c in body['classes'])


def test_merge_dry_run_reports_counts_without_writing(
    client: TestClient, registry: ClassRegistry, fake_os: AsyncMock
) -> None:
    async def _count(*, index: str, body: dict[str, Any]) -> dict[str, int]:
        text = str(body)
        if 'must_not' not in text:
            # holdout-blocking query: source class AND test_holdout, no negation.
            return {'count': 2}
        if 'class_validated' in text:
            return {'count': 7}
        return {'count': 40}

    fake_os.count = AsyncMock(side_effect=_count)
    r = client.post('/curation/classes/merge?dry_run=true', json={'source_id': 1, 'target_id': 0})
    assert r.status_code == 200, r.text
    assert r.json() == {
        'dry_run': True,
        'source_id': 1,
        'target_id': 0,
        'would_relabel': 40,
        'would_unvalidate': 7,
        'holdout_blocking': 2,
        'blocked': True,
    }
    assert not _entry(registry, 1).deprecated
    fake_os.update_by_query.assert_not_called()


def test_merge_dry_run_unknown_class_is_400(client: TestClient) -> None:
    r = client.post('/curation/classes/merge?dry_run=true', json={'source_id': 9, 'target_id': 0})
    assert r.status_code == 400
