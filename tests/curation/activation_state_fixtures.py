"""Shared fixtures for the activation tests: a READY-model state and a registry whose
default profile is the multi-region one.

Registered as a plugin in ``tests/conftest.py`` (pytest ``pytest_plugins``,
https://docs.pytest.org/en/stable/how-to/writing_plugins.html) so the projects
clone test uses them without importing fixtures from a conftest.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def ready_state(monkeypatch):
    from src.services.models.triton_control import TritonControlService

    state = {'ready': {'wheel_detector', 'other_detector'}}

    async def _repo():
        return [{'name': n, 'state': 'READY', 'version': '1'} for n in state['ready']]

    monkeypatch.setattr(TritonControlService, 'get_repository_index', lambda _s: _repo())
    return state


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
