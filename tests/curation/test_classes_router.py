"""Tests for `src/routers/curation/classes.py`.

Server-side reserved-hotkey guard on `PUT /curation/classes/{class_id}`.
Mirrors the labeler frontend's own reserved-letter list (the
non-`/` subset — `/` has no backend hotkey entry point to guard).

We mount the shared curation `router` (all sub-modules register onto
one `APIRouter`, see `src/routers/curation/_common.py`) on a minimal
FastAPI app and back `get_class_registry()` with a real `ClassRegistry`
pointed at a tmp-path JSON file, seeded via `add_class`, so the tests
exercise the real load -> mutate -> atomic-write path rather than a mock.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.clients.curation_opensearch import ClassRegistry
from src.routers.curation.classes import RESERVED_HOTKEY_LETTERS


@pytest.fixture
def registry(tmp_path: Any) -> ClassRegistry:
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('sedan', group='vehicle')
    reg.add_class('suv', group='vehicle')
    return reg


@pytest.fixture
def fake_opensearch() -> AsyncMock:
    """Minimal stub for `list_classes`'s aggregation + region-count queries."""
    fake = AsyncMock()
    fake.search = AsyncMock(return_value={'aggregations': {}})
    fake.count = AsyncMock(return_value={'count': 0})
    return fake


@pytest.fixture
def app_client(
    registry: ClassRegistry, fake_opensearch: AsyncMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_opensearch

    with TestClient(app) as client:
        yield client


# =============================================================================
# Reserved-hotkey rejection (PUT /curation/classes/{id})
# =============================================================================


@pytest.mark.parametrize('letter', sorted(RESERVED_HOTKEY_LETTERS))
def test_put_class_rejects_reserved_hotkey(
    app_client: TestClient, registry: ClassRegistry, letter: str
) -> None:
    class_id = registry.load().classes[0].class_id
    resp = app_client.put(f'/curation/classes/{class_id}', json={'hotkey_letter': letter})
    assert resp.status_code == 422, resp.text
    # Not persisted.
    entry = registry.get(class_id)
    assert entry is not None
    assert entry.hotkey_letter is None


def test_put_class_accepts_nonreserved_hotkey(
    app_client: TestClient, registry: ClassRegistry
) -> None:
    class_id = registry.load().classes[0].class_id
    resp = app_client.put(f'/curation/classes/{class_id}', json={'hotkey_letter': 's'})
    assert resp.status_code == 200, resp.text
    entry = registry.get(class_id)
    assert entry is not None
    assert entry.hotkey_letter == 's'


def test_reserved_set_matches_frontend() -> None:
    """Backend and frontend reserved-letter lists can't silently drift.

    The labeler frontend additionally reserves '/' (opens the
    /review fuzzy-search picker), which has no PUT-time backend
    equivalent to guard here — deliberately excluded from this comparison.
    """
    assert frozenset({'g', 'n', 'd', 'z', 'x', 'u', 'a', 'm'}) == RESERVED_HOTKEY_LETTERS


# =============================================================================
# POST /curation/classes structurally cannot set a hotkey
# =============================================================================


def test_post_class_cannot_set_hotkey(app_client: TestClient) -> None:
    """`ClassCreateRequest` has no `hotkey_letter` field: the extra key is
    silently ignored by pydantic, and the created class has no hotkey.
    """
    resp = app_client.post(
        '/curation/classes', json={'name': 'hatchback', 'group': 'vehicle', 'hotkey_letter': 'd'}
    )
    assert resp.status_code == 201, resp.text
    class_id = resp.json()['class_id']

    list_resp = app_client.get('/curation/classes')
    assert list_resp.status_code == 200
    entry = next(c for c in list_resp.json()['classes'] if c['class_id'] == class_id)
    assert entry['hotkey_letter'] is None


# =============================================================================
# GET /curation/classes/{id} (contract audit: MISSING row)
# =============================================================================


def test_get_class_returns_the_same_entry_as_the_list(app_client: TestClient) -> None:
    listed = app_client.get('/curation/classes').json()['classes']
    target = listed[1]
    resp = app_client.get(f'/curation/classes/{target["class_id"]}')
    assert resp.status_code == 200, resp.text
    assert resp.json() == target


def test_get_class_unknown_id_is_404(app_client: TestClient) -> None:
    assert app_client.get('/curation/classes/9999').status_code == 404
