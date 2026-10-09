"""A fresh project (created directly, or via clone_settings with no
'classes' axis) gets the active region profile's class seeded into its
own registry (coordinator follow-up to region_class.py's
``ensure_region_class``, aff64066/3389fb85). Without this, a project
created after startup had an empty registry and could never resolve its
region class by name -- ``_seed_region_classes()`` at startup only ever
covers projects that already existed when the process booted."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.config import DetectionProfile
from src.services.projects import lifecycle
from src.services.projects.registry import ProjectRegistry, set_project_registry

from .conftest import FakeLifecycleOpenSearch, fake_ensure_indexes


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation as curation_mod
    from src.services.projects import capacity as capacity_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)
    yield
    curation_mod._default_curation_config = None
    capacity_mod._cache = None
    set_project_registry(None)


@pytest.fixture(autouse=True)
def _noop_ensure_indexes():
    with patch(
        'src.routers.curation._common._ensure_indexes',
        new=AsyncMock(side_effect=fake_ensure_indexes),
    ):
        yield


@pytest.fixture(autouse=True)
def _region_profile(monkeypatch):
    import src.services.curation.region_class as region_class_mod

    monkeypatch.setattr(
        region_class_mod,
        'get_active_region_profile',
        lambda: DetectionProfile(name='p', region_class_name='wheel'),
    )


def _registry_for(client: FakeLifecycleOpenSearch) -> ProjectRegistry:
    registry = ProjectRegistry(lambda: client)
    set_project_registry(registry)
    return registry


def _class_names(record) -> list[str]:
    from src.clients.curation_opensearch.registry import ClassRegistry

    return [
        c.class_name for c in ClassRegistry(record.resources.class_registry_path).load().classes
    ]


def test_create_project_seeds_the_region_class() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)

    record, _ = asyncio.run(lifecycle.create_project(client, slug='cars', display_name='Cars'))

    assert _class_names(record) == ['wheel']


def test_clone_settings_without_classes_axis_still_seeds_region_class() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)

    source, _ = asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    # alpha already has 'wheel' seeded by its own create_project call;
    # add a second class so a full-classes clone would differ from a
    # settings-only clone.
    from src.clients.curation_opensearch.registry import ClassRegistry

    ClassRegistry(source.resources.class_registry_path).add_class('extra')

    target, _ = asyncio.run(
        lifecycle.create_project(
            client,
            slug='beta',
            display_name='Beta',
            clone_settings_from='alpha',
            clone_axes=['settings_defaults'],
        )
    )

    # The 'classes' axis was never cloned, so beta's registry is its own
    # -- but still seeded with the region class, not empty.
    assert _class_names(target) == ['wheel']
    assert 'extra' not in _class_names(target)


def test_clone_settings_with_classes_axis_does_not_duplicate_the_region_class() -> None:
    client = FakeLifecycleOpenSearch()
    _registry_for(client)

    source, _ = asyncio.run(lifecycle.create_project(client, slug='alpha', display_name='Alpha'))
    assert _class_names(source) == ['wheel']

    target, _ = asyncio.run(
        lifecycle.create_project(
            client,
            slug='beta',
            display_name='Beta',
            clone_settings_from='alpha',
            clone_axes=['classes'],
        )
    )

    assert _class_names(target) == ['wheel']
