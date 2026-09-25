"""Tests for ``POST /classes/{id}/deprecate`` and ``POST /classes/{id}/restore``.

Direct retirement path for a class with no data and no merge target --
the counterpart to ``POST /classes/merge`` (source, ClassMergeRequest),
which needs both a source *and* a target and is meant for classes that
still have items.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.clients.curation_opensearch import ClassRegistry, ClassRegistryError


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
    """Defaults to zero references everywhere; tests override ``count``."""
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


# =============================================================================
# Registry set_deprecated
# =============================================================================


def test_set_deprecated_unknown_id_raises(registry: ClassRegistry) -> None:
    with pytest.raises(ClassRegistryError, match='not found'):
        registry.set_deprecated(9999, True)


def test_set_deprecated_true_clears_hotkey(registry: ClassRegistry) -> None:
    registry.set_deprecated(0, False)  # no-op path still exercised below
    reg = registry.load()
    for c in reg.classes:
        if c.class_id == 0:
            c.hotkey_letter = 'v'
    registry._atomic_write(reg)
    assert _entry(registry, 0).hotkey_letter == 'v'

    entry = registry.set_deprecated(0, True)
    assert entry.deprecated is True
    assert entry.hotkey_letter is None
    assert _entry(registry, 0).hotkey_letter is None


def test_set_deprecated_false_name_clash_raises(registry: ClassRegistry) -> None:
    registry.set_deprecated(0, True)  # sedan deprecated
    registry.add_class('sedan', group='vehicle')  # new active class reuses the name
    with pytest.raises(ClassRegistryError, match='already in use'):
        registry.set_deprecated(0, False)


# =============================================================================
# Deprecate route
# =============================================================================


def test_deprecate_empty_class_works(client: TestClient, registry: ClassRegistry) -> None:
    resp = client.post('/curation/classes/0/deprecate')
    assert resp.status_code == 200, resp.text
    assert resp.json()['deprecated'] is True
    assert _entry(registry, 0).deprecated is True


def test_deprecate_class_with_items_is_409_with_counts(
    client: TestClient, fake_os: AsyncMock, registry: ClassRegistry
) -> None:
    fake_os.count = AsyncMock(side_effect=[{'count': 3}, {'count': 1}])
    resp = client.post('/curation/classes/0/deprecate')
    assert resp.status_code == 409, resp.text
    detail = resp.json()['detail']
    assert detail['error'] == 'class_still_referenced'
    assert detail['class_id'] == 0
    assert detail['item_count'] == 3
    assert detail['confirmed_label_count'] == 1
    assert _entry(registry, 0).deprecated is False


def test_deprecate_class_with_only_confirmed_labels_is_409(
    client: TestClient, fake_os: AsyncMock, registry: ClassRegistry
) -> None:
    """Items-index count can be zero while confirmed-label docs still
    reference the class -- either nonzero count blocks."""
    fake_os.count = AsyncMock(side_effect=[{'count': 0}, {'count': 2}])
    resp = client.post('/curation/classes/0/deprecate')
    assert resp.status_code == 409, resp.text
    assert resp.json()['detail']['item_count'] == 0
    assert resp.json()['detail']['confirmed_label_count'] == 2
    assert _entry(registry, 0).deprecated is False


def test_deprecate_unknown_id_is_404(client: TestClient) -> None:
    resp = client.post('/curation/classes/9999/deprecate')
    assert resp.status_code == 404, resp.text


def test_deprecate_is_idempotent(
    client: TestClient, fake_os: AsyncMock, registry: ClassRegistry
) -> None:
    assert client.post('/curation/classes/0/deprecate').status_code == 200
    # Second call must succeed even though a real count would now 409 --
    # idempotent means it doesn't re-check references on an already
    # deprecated class.
    fake_os.count = AsyncMock(return_value={'count': 999})
    resp = client.post('/curation/classes/0/deprecate')
    assert resp.status_code == 200, resp.text
    assert resp.json()['deprecated'] is True


def test_deprecate_clears_hotkey(client: TestClient, registry: ClassRegistry) -> None:
    assert client.put('/curation/classes/0', json={'hotkey_letter': 'v'}).status_code == 200
    assert _entry(registry, 0).hotkey_letter == 'v'

    resp = client.post('/curation/classes/0/deprecate')
    assert resp.status_code == 200, resp.text
    assert resp.json()['hotkey_letter'] is None
    assert _entry(registry, 0).hotkey_letter is None


# =============================================================================
# Restore route
# =============================================================================


def test_restore_works(client: TestClient, registry: ClassRegistry) -> None:
    assert client.post('/curation/classes/0/deprecate').status_code == 200
    resp = client.post('/curation/classes/0/restore')
    assert resp.status_code == 200, resp.text
    assert resp.json()['deprecated'] is False
    assert _entry(registry, 0).deprecated is False


def test_restore_unknown_id_is_404(client: TestClient) -> None:
    assert client.post('/curation/classes/9999/restore').status_code == 404


def test_restore_name_clash_is_409(client: TestClient, registry: ClassRegistry) -> None:
    assert client.post('/curation/classes/0/deprecate').status_code == 200
    registry.add_class('sedan', group='vehicle')  # active class reclaims the name

    resp = client.post('/curation/classes/0/restore')
    assert resp.status_code == 409, resp.text
    assert _entry(registry, 0).deprecated is True


# =============================================================================
# Deprecated classes stay excluded from consumers that already filter them
# =============================================================================


def test_deprecated_class_disappears_from_non_deprecated_class_name_lists(
    client: TestClient, registry: ClassRegistry
) -> None:
    """``GET /classes`` still lists a deprecated class (with
    ``deprecated: true``, for history/dashboards); every consumer that
    already does ``if not c.deprecated`` -- VLM prompt class lists, the
    training class picker's default class set, export's live-id list --
    must stop offering it once deprecated."""
    assert client.post('/curation/classes/0/deprecate').status_code == 200

    listed = client.get('/curation/classes').json()['classes']
    entry = next(c for c in listed if c['class_id'] == 0)
    assert entry['deprecated'] is True  # still visible for history

    from src.routers.curation_train import _resolve_target_classes
    from src.services.training.jobs import TrainJobSpec

    spec = TrainJobSpec(job_id='j', include_classes=None, dataset_export_dir='/tmp/export')
    import src.routers.curation_train as curation_train_mod

    orig = curation_train_mod.get_class_registry
    curation_train_mod.get_class_registry = lambda: registry
    try:
        target_ids = _resolve_target_classes(spec)
    finally:
        curation_train_mod.get_class_registry = orig
    assert 0 not in target_ids
    assert 1 in target_ids
