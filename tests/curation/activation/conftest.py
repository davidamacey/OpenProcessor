"""Fixtures for the activation tests: an app over a fake config index, a READY-model state,
and a three-profile registry (single-region default, multi-region, other detector)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.clients.curation_opensearch import settings_doc
    from src.services.config_store.store import reset_config_stores

    settings_doc._settings_cache.clear()
    reset_config_stores()
    yield
    settings_doc._settings_cache.clear()
    reset_config_stores()


@pytest.fixture
def app_client(monkeypatch):
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = FakeConfigOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


@pytest.fixture
def ready_state(monkeypatch):
    from src.services.triton_control import TritonControlService

    state = {'ready': {'wheel_detector', 'other_detector'}}

    async def _repo():
        return [{'name': n, 'state': 'READY', 'version': '1'} for n in state['ready']]

    monkeypatch.setattr(TritonControlService, 'get_repository_index', lambda _s: _repo())
    return state


@pytest.fixture
def registry():
    from src.config import DetectionProfile
    from src.services.detection import profile_registry

    profile_registry._reset_registry_for_tests()
    profile_registry.register_profile(
        DetectionProfile(
            name='base',
            detector_model='wheel_detector',
            text_reader='none',
            max_regions_per_item=1,
        ),
        default=True,
    )
    profile_registry.register_profile(
        DetectionProfile(
            name='multi',
            detector_model='wheel_detector',
            text_reader='none',
            max_regions_per_item=3,
        )
    )
    profile_registry.register_profile(
        DetectionProfile(name='other', detector_model='other_detector', text_reader='none')
    )
    yield profile_registry
    profile_registry._reset_registry_for_tests()


@pytest.fixture
def multi_default_registry():
    from src.config import DetectionProfile
    from src.services.detection import profile_registry

    profile_registry._reset_registry_for_tests()
    profile_registry._ENV_RESOLVED = True
    profile_registry.register_profile(
        DetectionProfile(
            name='multi',
            detector_model='wheel_detector',
            text_reader='none',
            max_regions_per_item=3,
        ),
        default=True,
    )
    profile_registry.register_profile(
        DetectionProfile(
            name='base',
            detector_model='wheel_detector',
            text_reader='none',
            max_regions_per_item=1,
        )
    )
    yield profile_registry
    profile_registry._reset_registry_for_tests()
