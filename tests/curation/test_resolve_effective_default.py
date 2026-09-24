"""Tests for ``resolve_effective_default`` (curation deployment-settings
plan) -- the single function both ``GET /methods``'s per-axis ``default``
flag and every real endpoint's omitted-param resolution must share.

Uses the same ``FakeSettingsOpenSearch`` fake as
``test_curation_settings_client.py`` so a "PUT then resolve" round trip
exercises the real client functions, not a mock of them.

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

from curation.test_curation_settings_client import FakeSettingsOpenSearch
from src.clients import curation_opensearch
from src.clients.curation_opensearch import update_curation_settings
from src.services.curation.strategy_defaults import resolve_effective_default


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def fake_os() -> FakeSettingsOpenSearch:
    return FakeSettingsOpenSearch()


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """See test_curation_settings_client.py's fixture of the same name --
    every test here shares the same default settings index key."""
    curation_opensearch._settings_cache.clear()
    yield
    curation_opensearch._settings_cache.clear()


@pytest.mark.asyncio
async def test_no_override_falls_back_to_hardcoded_constant(
    fake_os: FakeSettingsOpenSearch,
) -> None:
    from src.services.curation.clustering.methods import DEFAULT_METHOD

    assert await resolve_effective_default('cluster', fake_os) == DEFAULT_METHOD


@pytest.mark.asyncio
async def test_no_opensearch_client_skips_lookup_and_returns_hardcoded() -> None:
    from src.services.curation.clustering.methods import DEFAULT_METHOD

    assert await resolve_effective_default('cluster') == DEFAULT_METHOD
    assert await resolve_effective_default('cluster', None) == DEFAULT_METHOD


@pytest.mark.asyncio
async def test_valid_override_is_returned(fake_os: FakeSettingsOpenSearch) -> None:
    from src.services.curation.clustering.methods import available_methods

    non_default = next(m for m in available_methods() if m != 'ivf')
    await update_curation_settings(fake_os, {'cluster': non_default})
    assert await resolve_effective_default('cluster', fake_os) == non_default


@pytest.mark.asyncio
async def test_stale_override_no_longer_advertised_falls_back(
    fake_os: FakeSettingsOpenSearch,
) -> None:
    """An override that named a real id at PUT time but is no longer a
    currently-advertised id for the axis must fall back to the hardcoded
    constant rather than returning the dead id."""
    from src.services.curation.clustering.methods import DEFAULT_METHOD

    await update_curation_settings(fake_os, {'cluster': 'a_method_that_no_longer_exists'})
    assert await resolve_effective_default('cluster', fake_os) == DEFAULT_METHOD


@pytest.mark.asyncio
async def test_sort_axis_has_no_hardcoded_default_but_honors_a_valid_override(
    fake_os: FakeSettingsOpenSearch,
) -> None:
    """'sort' has no single hardcoded default (only a per-tab mapping,
    see review_sorts.default_sort_for_tab) -- with no override this must
    return None, and a valid override must still resolve."""
    assert await resolve_effective_default('sort', fake_os) is None

    await update_curation_settings(fake_os, {'sort': 'atypicality'})
    assert await resolve_effective_default('sort', fake_os) == 'atypicality'


@pytest.mark.asyncio
async def test_sort_override_naming_a_shadow_id_falls_back_to_none(
    fake_os: FakeSettingsOpenSearch,
) -> None:
    """A shadow/disabled sort id is not "currently advertised" for
    default-application purposes even though GET /methods still lists it
    -- an override naming one must not silently break every tab's queue."""
    await update_curation_settings(fake_os, {'sort': 'uniqueness'})  # shadow status
    assert await resolve_effective_default('sort', fake_os) is None


@pytest.mark.asyncio
async def test_detection_profile_and_prompt_pack_resolve_their_single_entry(
    fake_os: FakeSettingsOpenSearch,
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

    from src.services.curation.clustering.methods import DEFAULT_METHOD

    assert await resolve_effective_default('cluster', _Broken()) == DEFAULT_METHOD


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
