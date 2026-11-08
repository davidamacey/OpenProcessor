"""A project whose region profile names a region class gets that class in
its registry, so region export, the region gallery and class lists can
resolve it by name. A fresh install used to have an empty registry, so
nothing ever resolved."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from curation._vlm_test_support import empty_registry_reads

from src.config import DetectionProfile
from src.services.classes.registry import get_class_registry
from src.services.labeling.vlm_client import VlmIdentity


def _seed(monkeypatch: pytest.MonkeyPatch, profile: DetectionProfile | None) -> Any:
    import src.services.curation.region_class as mod

    monkeypatch.setattr(mod, 'get_active_region_profile', lambda: profile)
    return get_class_registry()


@pytest.mark.asyncio
async def test_seeds_the_region_class_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.region_class import ensure_region_class

    reg = _seed(monkeypatch, DetectionProfile(name='p', region_class_name='wheel'))

    first = await ensure_region_class()
    second = await ensure_region_class()

    names = [c.class_name for c in reg.load().classes]
    assert names == ['wheel']
    assert first == second == reg.load().classes[0].class_id


@pytest.mark.asyncio
async def test_existing_class_is_matched_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.curation.region_class import ensure_region_class

    reg = _seed(monkeypatch, DetectionProfile(name='p', region_class_name='wheel'))
    reg.seed(['car', 'Wheel'])

    assert await ensure_region_class() == 1
    assert [c.class_name for c in reg.load().classes] == ['car', 'Wheel']


@pytest.mark.asyncio
async def test_no_region_profile_seeds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.region_class import ensure_region_class

    reg = _seed(monkeypatch, None)
    assert await ensure_region_class() is None
    _seed(monkeypatch, DetectionProfile(name='p', region_class_name=''))
    assert await ensure_region_class() is None
    assert reg.load().classes == []


def test_item_classes_exclude_the_region_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item labelers (VLM, auto-label, probe) must never assign the region
    class to a whole item; it is a sub-box class."""
    from src.services.curation.region_class import item_classes

    reg = _seed(monkeypatch, DetectionProfile(name='p', region_class_name='wheel'))
    reg.seed(['car', 'Wheel', 'old'])
    reg.deprecate(2)

    names = [c.class_name for c in item_classes(reg.load().classes)]
    assert 'Wheel' not in names
    assert names == ['car']


@pytest.mark.asyncio
async def test_item_classes_exclude_a_seeded_region_class_after_profile_deactivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deactivating the project's region profile must not turn the
    already-seeded region class into a whole-item class (live regression:
    every COCO item was VLM-labelled with the seeded region class)."""
    from src.services.curation.region_class import ensure_region_class, item_classes

    reg = _seed(monkeypatch, DetectionProfile(name='p', region_class_name='wheel'))
    await ensure_region_class()
    reg.seed(['car'])
    _seed(monkeypatch, None)  # profile now off; registry keeps 'wheel'
    names = [c.class_name for c in item_classes(reg.load().classes)]
    assert names == ['car']


@pytest.mark.asyncio
async def test_label_batch_with_only_the_region_class_is_no_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.services.curation.vlm_batch as vlm_mod

    reg = _seed(monkeypatch, DetectionProfile(name='p', region_class_name='wheel'))
    reg.seed(['wheel'])

    class _NeverCalled:
        identity = VlmIdentity('env@None', 'test-vlm')

        async def label_or_propose_batch(self, *_a: object, **_k: object) -> None:
            raise AssertionError('the VLM must not be called with only the region class')

    labeler = _NeverCalled()
    monkeypatch.setenv('OP_VLM_URL', 'http://vlm.invalid:8000/v1')

    assert await vlm_mod.run_label_batch(empty_registry_reads(AsyncMock()), labeler, ['a']) == {
        'no_classes': True
    }


@pytest.mark.asyncio
async def test_worker_class_catalog_leaves_out_the_region_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.curation.worker.state import bound_class_catalog

    reg = _seed(monkeypatch, DetectionProfile(name='p', region_class_name='wheel'))
    reg.seed(['wheel', 'car'])
    car_id = next(c.class_id for c in reg.load().classes if c.class_name == 'car')

    names, name_to_id = await bound_class_catalog()
    assert names == ['car']
    assert name_to_id == {'car': car_id}
