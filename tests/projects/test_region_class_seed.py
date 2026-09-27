"""A project whose region profile names a region class gets that class in
its registry, so region export, the region gallery and class lists can
resolve it by name. A fresh install used to have an empty registry, so
nothing ever resolved."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.clients.curation_opensearch import ClassRegistry
from src.config import DetectionProfile


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: DetectionProfile | None
) -> ClassRegistry:
    import src.services.curation.region_class as mod

    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    monkeypatch.setattr(mod, 'get_class_registry', lambda: reg)
    monkeypatch.setattr(mod, 'get_active_region_profile', lambda: profile)
    return reg


def test_seeds_the_region_class_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.region_class import ensure_region_class

    reg = _seed(tmp_path, monkeypatch, DetectionProfile(name='p', region_class_name='wheel'))

    first = ensure_region_class()
    second = ensure_region_class()

    names = [c.class_name for c in reg.load().classes]
    assert names == ['wheel']
    assert first == second == reg.load().classes[0].class_id


def test_existing_class_is_matched_case_insensitively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation.region_class import ensure_region_class

    reg = _seed(tmp_path, monkeypatch, DetectionProfile(name='p', region_class_name='wheel'))
    reg.add_class('car')
    wheel_id = reg.add_class('Wheel')

    assert ensure_region_class() == wheel_id
    assert [c.class_name for c in reg.load().classes] == ['car', 'Wheel']


def test_no_region_profile_seeds_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.region_class import ensure_region_class

    reg = _seed(tmp_path, monkeypatch, None)
    assert ensure_region_class() is None
    reg_no_name = _seed(tmp_path, monkeypatch, DetectionProfile(name='p', region_class_name=''))
    assert ensure_region_class() is None
    assert reg.load().classes == []
    assert reg_no_name.load().classes == []
