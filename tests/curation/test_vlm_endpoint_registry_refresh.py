"""A by-name VLM endpoint route re-reads the registry before it answers 404.

A write made through another API worker bumps the revision counter in a doc this worker may
have cached as absent, so the first read after it misses; the prompt-pack and region-profile
routes already force the refresh on a miss.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.routers.curation import vlm_endpoints


pytestmark = pytest.mark.asyncio


class _Store:
    def __init__(self) -> None:
        self.current = object()
        self.refreshes: list[bool] = []
        self.known: set[str] = set()

    async def refresh(self, _client: Any) -> None:
        self.refreshes.append(False)

    async def refresh_on_miss(self, _client: Any) -> None:
        self.refreshes.append(True)
        self.known.add('late')


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    s = _Store()
    monkeypatch.setattr(vlm_endpoints, 'get_global_config_store', lambda: s)
    monkeypatch.setattr(
        vlm_endpoints, 'get_vlm_endpoint', lambda name: object() if name in s.known else None
    )
    return s


async def test_a_name_missing_from_the_snapshot_forces_one_refresh(store: _Store) -> None:
    await vlm_endpoints._registry(object(), name='late')
    assert store.refreshes == [False, True]


async def test_a_known_name_needs_no_forced_refresh(store: _Store) -> None:
    store.known.add('here')
    await vlm_endpoints._registry(object(), name='here')
    assert store.refreshes == [False]


async def test_listing_routes_never_force(store: _Store) -> None:
    await vlm_endpoints._registry(object())
    assert store.refreshes == [False]
