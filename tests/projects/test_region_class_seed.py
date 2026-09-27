"""A project whose region profile names a region class gets that class in
its registry, so region export, the region gallery and class lists can
resolve it by name. A fresh install used to have an empty registry, so
nothing ever resolved."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.clients.curation_opensearch import ClassRegistry
from src.config import DetectionProfile


if TYPE_CHECKING:
    from pathlib import Path


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


def test_item_classes_exclude_the_region_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item labelers (VLM, auto-label, probe) must never assign the region
    class to a whole item; it is a sub-box class."""
    from src.services.curation.region_class import item_classes

    reg = _seed(tmp_path, monkeypatch, DetectionProfile(name='p', region_class_name='wheel'))
    reg.add_class('car')
    reg.add_class('Wheel')
    reg.set_deprecated(reg.add_class('old'), True)

    names = [c.class_name for c in item_classes(reg.load().classes)]
    assert 'Wheel' not in names
    assert names == ['car']


def test_label_batch_with_only_the_region_class_is_no_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from fastapi import HTTPException

    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmLabelBatchRequest

    reg = _seed(tmp_path, monkeypatch, DetectionProfile(name='p', region_class_name='wheel'))
    reg.add_class('wheel')
    monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: reg)

    def _no_labeler(*_a: object, **_k: object) -> None:
        raise AssertionError('the VLM must not be called with only the region class')

    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', _no_labeler)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=['a']), object()))
    assert exc_info.value.status_code == 409
