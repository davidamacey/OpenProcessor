"""Activating a region profile registers its region class in the project
registry (kind region, group region); re-activation is a no-op and
deactivation never deletes a class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from curation.test_region_profiles_router import (  # noqa: F401
    PREFIX,
    _body,
    _reset_caches,
    app_client,
)
from src.clients.curation_opensearch import ClassRegistry


if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ClassRegistry:
    import src.services.curation.region_class as mod

    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    monkeypatch.setattr(mod, 'get_class_registry', lambda: reg)
    return reg


def _classes(reg: ClassRegistry) -> list[tuple[str, str]]:
    return [(c.class_name, c.group) for c in reg.load().classes]


def test_activation_registers_the_region_class_once_and_deactivation_keeps_it(
    app_client: TestClient,  # noqa: F811
    registry: ClassRegistry,
) -> None:
    app_client.post(PREFIX, json={'name': 'wheels', 'body': _body(region_class_name='wheel')})
    assert _classes(registry) == []

    r = app_client.post(f'{PREFIX}/wheels/activate', json={'expected_active': None, 'force': False})
    assert r.status_code == 200, r.text
    assert _classes(registry) == [('wheel', 'region')]

    active = app_client.get(f'{PREFIX}/active').json()['active']
    app_client.post(f'{PREFIX}/wheels/activate', json={'expected_active': active, 'force': False})
    assert _classes(registry) == [('wheel', 'region')]

    app_client.post(f'{PREFIX}/deactivate', json={'expected_active': active})
    assert _classes(registry) == [('wheel', 'region')]
