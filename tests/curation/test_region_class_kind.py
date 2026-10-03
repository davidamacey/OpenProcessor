"""A region class keeps ``kind='region'`` whatever this process believes the
active profile is (a worker whose config snapshot is not loaded yet sees none)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from src.services.curation.region_class import is_region_class


def _cls(name: str, group: str) -> Any:
    return cast('Any', SimpleNamespace(class_name=name, group=group))


def test_region_group_class_is_region_without_an_active_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        'src.services.curation.region_class.get_active_region_profile', lambda: None
    )
    assert is_region_class(_cls('wheel', 'region'))
    assert not is_region_class(_cls('car', 'vehicle'))


def test_active_profile_class_is_region_by_name_case_insensitively(monkeypatch) -> None:
    profile = SimpleNamespace(region_class_name='Wheel')
    monkeypatch.setattr(
        'src.services.curation.region_class.get_active_region_profile', lambda: profile
    )
    assert is_region_class(_cls('wheel', 'unknown'))
