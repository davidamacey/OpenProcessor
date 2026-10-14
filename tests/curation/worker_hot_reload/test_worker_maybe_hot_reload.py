"""``maybe_hot_reload``: the producer-loop swap check."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.worker_hot_reload._helpers import _fake_args, _record, _swap_kw
from src.config.project_context import bind_project
from src.services.config_store.index import activate, save_config
from src.services.config_store.store import get_config_store
from src.services.detection import profile_registry


pytestmark = pytest.mark.usefixtures('reference_region_profile')


# =============================================================================
# maybe_hot_reload: the producer-loop swap check -- pointer 2
# =============================================================================


@pytest.mark.asyncio
async def test_maybe_hot_reload_never_swaps_when_activation_is_unchanged() -> None:
    """No activation change across repeated cycles must never rebuild --
    the swap decision compares the store's served
    ``(profile, pack)`` ``AxisRef`` pair, never object identity or the
    runtime's own always-populated ``profile_ref``/``pack_ref``."""
    from scripts.curation.worker.runtime import RuntimeHolder, maybe_hot_reload

    client = FakeConfigOpenSearch()
    project = _record('steady-project')
    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()
    ctors = _swap_kw()

    with bind_project(project):
        store = get_config_store(mode='pinned')
        holder = RuntimeHolder()

        for _ in range(3):
            await maybe_hot_reload(
                store=store,
                opensearch=client,
                holder=holder,
                slug='steady-project',
                pool=MagicMock(),
                args=_fake_args(),
                queues=[],
                get_active_profile=lambda: profile,
                get_active_pack=lambda: pack,
                **ctors,
            )

        # First cycle builds once (nothing synced yet, store shows
        # nothing active either -> the "off/off" baseline still counts
        # as a change from "never synced"); every later cycle, with the
        # store unchanged, must not call any constructor again.
        assert ctors['region_detector_cls'].call_count == 1


@pytest.mark.asyncio
async def test_maybe_hot_reload_swaps_exactly_once_on_a_real_pinned_activation() -> None:
    """B2 (reviewer probe #3, reproduced): pinned store, baseline sync,
    then save+activate a profile, then 3 `maybe_hot_reload` cycles. Must
    build exactly once (on the cycle right after the activation), not
    zero times -- a pinned store's `refresh()` only stages
    `pending_snapshot`; the pre-fix code compared `store.current`
    (unmoved) against the last-synced refs and never saw the change."""
    from scripts.curation.worker.runtime import RuntimeHolder, maybe_hot_reload

    client = FakeConfigOpenSearch()
    project = _record('pinned-project')
    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    pack = resolve_prompt_pack()
    ctors = _swap_kw()

    with bind_project(project):
        store = get_config_store(mode='pinned')
        holder = RuntimeHolder()

        # Baseline cycle: nothing activated yet.
        await maybe_hot_reload(
            store=store,
            opensearch=client,
            holder=holder,
            slug='pinned-project',
            pool=MagicMock(),
            args=_fake_args(),
            queues=[],
            get_active_profile=lambda: profile,
            get_active_pack=lambda: pack,
            **ctors,
        )
        assert ctors['region_detector_cls'].call_count == 1  # env default, first-ever build

        idx = store.index
        doc = await save_config(
            client, idx, kind='region_profile', name='wheel', body={}, expected_revision=None
        )
        await activate(
            client,
            idx,
            axis='detection_profile',
            name='wheel',
            revision=doc['revision'],
            expected_active=None,
        )

        for _ in range(3):
            await maybe_hot_reload(
                store=store,
                opensearch=client,
                holder=holder,
                slug='pinned-project',
                pool=MagicMock(),
                args=_fake_args(),
                queues=[],
                get_active_profile=lambda: profile,
                get_active_pack=lambda: pack,
                **ctors,
            )

        # Exactly one more build across all 3 post-activation cycles.
        assert ctors['region_detector_cls'].call_count == 2
        assert store.current.active_profile == ('wheel', doc['revision'])


@pytest.mark.asyncio
async def test_build_uses_the_activated_pack_not_the_env_default() -> None:
    """B4: a pack activation must rebuild with THAT pack (via
    `active_prompt_pack`), never the env/file default (`resolve_prompt_pack`).
    The reviewer's probe: activate a stored pack named differently from
    the env default, swap, and check the built runtime's pack name and
    ref -- not just that *a* pack got attached."""
    from scripts.curation.worker.runtime import RuntimeHolder, maybe_hot_reload
    from src.services.labeling.vlm_prompts import active_prompt_pack, resolve_prompt_pack

    client = FakeConfigOpenSearch()
    project = _record('pack-project')
    profile = profile_registry.get_active_region_profile()
    assert profile is not None
    env_pack = resolve_prompt_pack()
    ctors = _swap_kw()

    with bind_project(project):
        store = get_config_store(mode='pinned')
        idx = store.index
        pack_body = {**env_pack.to_dict(), 'name': 'mypack'}
        doc = await save_config(
            client,
            idx,
            kind='prompt_pack',
            name='mypack',
            body=pack_body,
            expected_revision=None,
        )
        await activate(
            client,
            idx,
            axis='prompt_pack',
            name='mypack',
            revision=doc['revision'],
            expected_active=None,
        )

        holder = RuntimeHolder()
        new_rt = await maybe_hot_reload(
            store=store,
            opensearch=client,
            holder=holder,
            slug='pack-project',
            pool=MagicMock(),
            args=_fake_args(),
            queues=[],
            get_active_profile=lambda: profile,
            get_active_pack=active_prompt_pack,
            **ctors,
        )

    assert new_rt is not None
    assert new_rt.pack.name == 'mypack'
    assert new_rt.pack_ref == ('mypack', doc['revision'])
