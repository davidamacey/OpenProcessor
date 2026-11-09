"""Shared fixtures for the crop label/undo/exclude/discard route tests.

Registered as a plugin in ``tests/conftest.py`` (pytest ``pytest_plugins``,
https://docs.pytest.org/en/stable/how-to/writing_plugins.html) so test modules
use the fixtures without importing them from each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _class_registry_fake import InMemoryRegistry


if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

    from curation.query_fakes import QueryFakeOpenSearch


_NAMES = ('widget', 'gadget', 'gizmo')


@pytest.fixture
def registry_ids(tmp_path: Path) -> tuple[InMemoryRegistry, dict[str, int]]:
    reg = InMemoryRegistry()
    return reg, {name: reg.add(name) for name in _NAMES}


@pytest.fixture
def registry(registry_ids: tuple[InMemoryRegistry, dict[str, int]]) -> InMemoryRegistry:
    return registry_ids[0]


@pytest.fixture
def ids(registry_ids: tuple[InMemoryRegistry, dict[str, int]]) -> dict[str, int]:
    return registry_ids[1]


@pytest.fixture
def crops(monkeypatch: pytest.MonkeyPatch, registry: InMemoryRegistry):
    from types import SimpleNamespace

    from src.routers.curation import crops_exclude, crops_write

    monkeypatch.setattr(crops_write, 'get_class_registry', lambda: registry)
    return SimpleNamespace(
        label_crop=crops_write.label_crop,
        batch_label_crops=crops_write.batch_label_crops,
        move_crops=crops_write.move_crops,
        batch_exclude_crops=crops_exclude.batch_exclude_crops,
        batch_unexclude_crops=crops_exclude.batch_unexclude_crops,
    )


@pytest.fixture
def client_for():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    def _make(fake: QueryFakeOpenSearch) -> TestClient:
        app = FastAPI()
        from _curation_app import mount_curation_routers

        mount_curation_routers(app, curation_router)
        app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
        return TestClient(app)

    return _make
