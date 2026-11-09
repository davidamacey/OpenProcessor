"""Tests for ``resolve_effective_default`` -- the single function both
``GET /methods``'s per-axis ``default``
flag and every real endpoint's omitted-param resolution must share.

Uses the in-memory settings row of ``config_store.memory_settings``.

Imports ``resolve_effective_default`` from
``src.services.curation.strategy_defaults`` directly (not the
``strategy_registry`` re-export) -- this test predates the
``strategy_registry.py`` rewiring commit, so the re-export doesn't exist
yet at this point in history; both import paths resolve to the same
function once that later commit lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from config_store.memory_settings import MemorySettings

from src.services.curation.strategy_defaults import resolve_effective_default


if TYPE_CHECKING:
    from config_store.memory_settings import MemorySettings


@pytest.fixture
def fake_os(memory_settings) -> object:
    """Stands for the client the callers pass; the settings row is the in-memory one."""
    return object()


@pytest.mark.asyncio
async def test_the_retired_cluster_axis_resolves_to_nothing_even_with_a_stored_override(
    fake_os: object,
    memory_settings: MemorySettings,
) -> None:
    """Clustering takes its method per run now: a settings document that still carries a
    ``cluster`` default (written before the axis went) must not resolve to anything."""
    await memory_settings.update_curation_settings(
        {'cluster': 'ivf'}, expected_revision=memory_settings.revision
    )
    assert await resolve_effective_default('cluster', fake_os) is None
    assert await resolve_effective_default('cluster') is None


@pytest.mark.asyncio
async def test_stale_sort_override_no_longer_advertised_falls_back(
    fake_os: object,
    memory_settings: MemorySettings,
) -> None:
    """An override that named a real id at PUT time but is no longer a
    currently-advertised id for the axis must fall back to the hardcoded
    default rather than returning the dead id."""
    await memory_settings.update_curation_settings(
        {'sort': 'a_sort_that_no_longer_exists'}, expected_revision=memory_settings.revision
    )
    assert await resolve_effective_default('sort', fake_os) is None


@pytest.mark.asyncio
async def test_sort_axis_has_no_hardcoded_default_but_honors_a_valid_override(
    fake_os: object,
    memory_settings: MemorySettings,
) -> None:
    """'sort' has no single hardcoded default (only a per-tab mapping,
    see review_sorts.default_sort_for_tab) -- with no override this must
    return None, and a valid override must still resolve."""
    assert await resolve_effective_default('sort', fake_os) is None

    await memory_settings.update_curation_settings(
        {'sort': 'atypicality'}, expected_revision=memory_settings.revision
    )
    assert await resolve_effective_default('sort', fake_os) == 'atypicality'


@pytest.mark.asyncio
async def test_sort_override_naming_a_shadow_id_falls_back_to_none(
    fake_os: object,
    memory_settings: MemorySettings,
) -> None:
    """A shadow/disabled sort id is not "currently advertised" for
    default-application purposes even though GET /methods still lists it
    -- an override naming one must not silently break every tab's queue."""
    await memory_settings.update_curation_settings(
        {'sort': 'uniqueness'}, expected_revision=memory_settings.revision
    )  # disabled by default (D2)
    assert await resolve_effective_default('sort', fake_os) is None


@pytest.mark.asyncio
async def test_detection_profile_and_prompt_pack_resolve_their_single_entry(
    fake_os: object,
    memory_settings: MemorySettings,
) -> None:
    from src.services.detection import cascade_detect  # noqa: F401 - registers the profile
    from src.services.detection.profile_registry import get_default_profile_name
    from src.services.labeling.vlm_prompts import resolve_prompt_pack

    assert (
        await resolve_effective_default('detection_profile', fake_os) == get_default_profile_name()
    )
    assert await resolve_effective_default('prompt_pack', fake_os) == resolve_prompt_pack().name


@pytest.mark.asyncio
async def test_broken_opensearch_client_falls_back_rather_than_raising() -> None:
    class _Broken:
        async def get(self, *args: object, **kwargs: object) -> None:  # noqa: ARG002
            raise RuntimeError('opensearch unreachable')

    assert await resolve_effective_default('sort', _Broken()) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
