"""``build_runtime`` and ``quiesce_and_swap`` (``scripts/curation/worker/runtime.py``)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.worker_hot_reload._helpers import _fake_args, _fake_ctors, _record, _swap_kw
from src.config.project_context import bind_project
from src.services.config_store import get_global_config_store
from src.services.config_store.index import activate, save_config
from src.services.config_store.store import get_config_store
from src.services.detection import profile_registry


pytestmark = pytest.mark.usefixtures('reference_region_profile')


@pytest.mark.asyncio
async def test_build_runtime_returns_refs() -> None:
    from scripts.curation.worker.runtime import build_runtime

    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()

    rt = await build_runtime(
        pool=MagicMock(),
        profile=profile,
        pack=pack,
        args=_fake_args(),
        profile_revision=3,
        pack_revision=None,
        vlm_endpoint=None,
        **_fake_ctors(),
    )
    assert rt.profile_ref == (profile.name, 3)
    assert rt.pack_ref == (pack.name, None)
    assert rt.detector is not None
    assert rt.vlm is None  # no active endpoint


@pytest.mark.asyncio
async def test_build_runtime_uses_the_passed_in_constructors_not_fresh_imports() -> None:
    """The monkeypatch-target fix: build_runtime must call the exact
    classes it was handed, never import its own copies of
    RegionDetector/PaddleOcrTextRecognizer/SegmenterClient/build_vlm_labeler --
    otherwise a test (or runner.py's producer loop rebuilding through a
    module-level name a test patched) silently keeps talking to the
    real, unpatched class."""
    from scripts.curation.worker.runtime import build_runtime

    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()
    ctors = _fake_ctors()
    args = _fake_args()
    endpoint = MagicMock()
    pool = MagicMock()

    rt = await build_runtime(
        pool=pool, profile=profile, pack=pack, args=args, vlm_endpoint=endpoint, **ctors
    )

    ctors['region_detector_cls'].assert_called_once_with(pool, profile)
    ctors['ocr_recognizer_cls'].assert_called_once_with(pool, profile)
    ctors['segmenter_cls'].assert_called_once()
    ctors['build_vlm'].assert_called_once_with(endpoint, pack)
    assert rt.detector is ctors['region_detector_cls'].return_value
    assert rt.vlm is ctors['build_vlm'].return_value
    assert rt.vlm_endpoint is endpoint


@pytest.mark.asyncio
async def test_quiesce_and_swap_drains_queues_before_building() -> None:
    from scripts.curation.worker.runtime import RuntimeHolder, current_want, quiesce_and_swap

    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()

    q: asyncio.Queue[Any] = asyncio.Queue()
    await q.put('item')
    holder = RuntimeHolder()
    drained_before_build = False

    async def _drain_soon() -> None:
        nonlocal drained_before_build
        await asyncio.sleep(0.01)
        drained_before_build = True
        q.task_done()

    project = _record('alpha')
    with bind_project(project):
        store = get_config_store(mode='pinned')

        asyncio.get_event_loop().create_task(_drain_soon())
        rt = await quiesce_and_swap(
            queues=[q],
            holder=holder,
            slug='alpha',
            store=store,
            pool=MagicMock(),
            args=_fake_args(),
            want=current_want(store, get_global_config_store(mode='pinned')),
            get_active_profile=lambda: profile,
            get_active_pack=lambda: pack,
            **_swap_kw(),
        )
    assert drained_before_build
    assert holder.get('alpha') is rt
    assert holder.get('beta') is None


@pytest.mark.asyncio
async def test_quiesce_and_swap_pins_before_resolving_active_profile() -> None:
    """B2: pin_active() must run strictly between the drain and the
    build, so `get_active_profile`/`get_active_pack` (which read
    `store.current`) already see the newly-pinned snapshot -- not the
    stale one from before this cycle's refresh."""
    from scripts.curation.worker.runtime import RuntimeHolder, current_want, quiesce_and_swap

    client = FakeConfigOpenSearch()
    project = _record('alpha')
    profile = profile_registry.get_active_region_profile()
    assert profile is not None

    with bind_project(project):
        store = get_config_store(mode='pinned')
        idx = store.index
        doc = await save_config(
            client, idx, kind='region_profile', name=profile.name, body={}, expected_revision=None
        )
        await activate(
            client,
            idx,
            axis='detection_profile',
            name=profile.name,
            revision=doc['revision'],
            expected_active=None,
        )
        await store.refresh(client)
        assert store.pending_snapshot is not None
        assert store.current.active_profile is None  # not pinned yet

        seen_active_profile_at_call_time: list[Any] = []

        def _get_active_profile() -> Any:
            seen_active_profile_at_call_time.append(store.current.active_profile)
            return profile

        holder = RuntimeHolder()
        await quiesce_and_swap(
            queues=[],
            holder=holder,
            slug='alpha',
            store=store,
            pool=MagicMock(),
            args=_fake_args(),
            want=current_want(store, get_global_config_store(mode='pinned')),
            get_active_profile=_get_active_profile,
            get_active_pack=profile_registry_pack,
            **_swap_kw(),
        )
        # By the time get_active_profile() ran, the store had already
        # been pinned to the new activation.
        assert seen_active_profile_at_call_time == [(profile.name, doc['revision'])]
        assert store.pending_snapshot is None


def profile_registry_pack() -> Any:
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    return resolve_prompt_pack()
