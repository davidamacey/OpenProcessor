"""Tests for `src/routers/curation/classes.py`.

Server-side reserved-hotkey guard on class create/update. The backend
owns the reserved set and serves it on `GET /classes`.

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


def test_reserved_set_covers_every_single_key_labeler_action() -> None:
    """The backend owns the full reserved set: the global labeling actions
    (accept, skip, discard, undo, ignore, un-ignore, select-all, move), the
    class-picker key '/' and the region-review keys (d/f/e/b)."""
    assert frozenset('gndzxuam/feb') == RESERVED_HOTKEY_LETTERS


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


# =============================================================================
# Region-class ``kind`` marking. This used to override
# sample_count/validated_count/cluster_size with the region inventory
# total, which made a region slot look like an item
# class with thousands of validated crops -- inflating /train's class
# picker and /export's per-class table. Region classes are now only
# flagged via ``kind='region'``; their item counts stay the real (usually
# zero) class-aggregation numbers so item-count consumers aren't fooled.
# =============================================================================


def test_region_class_is_marked_kind_region_and_keeps_real_item_counts(
    app_client: TestClient,
    registry: ClassRegistry,
    fake_opensearch: AsyncMock,
    reference_region_profile: None,
) -> None:
    """The example license_plate profile is active (region_class_name=
    'license_plate'); a registry class of that name is marked
    kind='region' but its sample_count/validated_count/cluster_size are
    NOT overridden with the region inventory total -- they stay whatever
    the class/cluster aggregations reported (0 here, since no item doc
    has that class_id)."""
    registry.add_class('license_plate', group='region')
    fake_opensearch.count = AsyncMock(side_effect=[{'count': 7}, {'count': 3}])

    resp = app_client.get('/curation/classes')
    assert resp.status_code == 200, resp.text
    entry = next(c for c in resp.json()['classes'] if c['class_name'] == 'license_plate')
    assert entry['kind'] == 'region'
    assert entry['sample_count'] == 0
    assert entry['validated_count'] == 0
    assert entry['cluster_size'] == 0


def test_non_region_class_is_marked_kind_item(
    app_client: TestClient,
    fake_opensearch: AsyncMock,
    reference_region_profile: None,
) -> None:
    resp = app_client.get('/curation/classes')
    assert resp.status_code == 200, resp.text
    for entry in resp.json()['classes']:
        assert entry['kind'] == 'item'


def test_region_class_kind_marking_is_a_noop_without_an_active_profile(
    app_client: TestClient,
    registry: ClassRegistry,
    fake_opensearch: AsyncMock,
) -> None:
    """No region profile configured -- a class happening to be named
    'license_plate' must NOT be marked kind='region' (it isn't hardcoded
    to that literal)."""
    registry.add_class('license_plate', group='region')

    resp = app_client.get('/curation/classes')
    assert resp.status_code == 200, resp.text
    entry = next(c for c in resp.json()['classes'] if c['class_name'] == 'license_plate')
    assert entry['kind'] == 'item'
    assert entry['sample_count'] == 0
    assert entry['validated_count'] == 0


def test_region_class_kind_marking_uses_a_differently_named_profiles_region_class(
    app_client: TestClient,
    registry: ClassRegistry,
    fake_opensearch: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment with a different region_class_name marks THAT class,
    not 'license_plate' -- proves the lookup isn't secretly hardcoded to
    the example's name."""
    from src.services.detection import profile_registry

    monkeypatch.setenv(f'{profile_registry.REGION_DETECTION_ENV_PREFIX}NAME', 'widget')
    monkeypatch.setenv(
        f'{profile_registry.REGION_DETECTION_ENV_PREFIX}REGION_CLASS_NAME', 'widget_label'
    )
    profile_registry._reset_registry_for_tests()
    try:
        registry.add_class('widget_label', group='region')

        resp = app_client.get('/curation/classes')
        assert resp.status_code == 200, resp.text
        entry = next(c for c in resp.json()['classes'] if c['class_name'] == 'widget_label')
        assert entry['kind'] == 'region'
    finally:
        profile_registry._reset_registry_for_tests()
