"""A by-name miss re-reads the registry cheaply and at most once per burst (#442).

The forced read used to be a full reload on every miss: N concurrent misses on an unknown
name (the API is unauthenticated) meant N index refreshes plus N searches.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from src.services.config_store import store as store_mod
from src.services.config_store.snapshot import _EMPTY_SNAPSHOT, ConfigSnapshot
from src.services.config_store.store import ConfigStore, ConfigStoreUnavailableError


pytestmark = pytest.mark.asyncio


class _Backend:
    """Stands in for the config index: a revision counter and a snapshot loader."""

    def __init__(self) -> None:
        self.revision = 1
        self.revision_reads = 0
        self.loads = 0
        self.fail = False

    async def get_revision(self, _client: Any, _index: str) -> int:
        self.revision_reads += 1
        await asyncio.sleep(0)
        if self.fail:
            raise OSError('opensearch down')
        return self.revision

    async def load(self, _client: Any, revision: int) -> ConfigSnapshot:
        self.loads += 1
        await asyncio.sleep(0)
        if self.fail:
            raise OSError('opensearch down')
        return replace(_EMPTY_SNAPSHOT, config_revision=revision, loaded_at=1.0)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> _Backend:
    b = _Backend()
    monkeypatch.setattr(store_mod, 'get_config_revision', b.get_revision)

    async def _load(_store: ConfigStore, client: Any, revision: int) -> ConfigSnapshot:
        return await b.load(client, revision)

    monkeypatch.setattr(ConfigStore, '_load_snapshot', _load)
    return b


async def _loaded_store(backend: _Backend) -> ConfigStore:
    store = ConfigStore('idx')
    await store.refresh(object())
    backend.loads = backend.revision_reads = 0
    return store


async def test_unchanged_revision_costs_no_reload(backend: _Backend) -> None:
    store = await _loaded_store(backend)
    await store.refresh_on_miss(object())
    assert backend.loads == 0
    assert backend.revision_reads == 1


async def test_moved_revision_reloads_and_publishes(backend: _Backend) -> None:
    store = await _loaded_store(backend)
    backend.revision = 2
    got = await store.refresh_on_miss(object())
    assert backend.loads == 1
    assert got.config_revision == 2
    assert store.current.config_revision == 2


async def test_a_storm_of_misses_shares_one_read(backend: _Backend) -> None:
    store = await _loaded_store(backend)
    backend.revision = 2
    await asyncio.gather(*(store.refresh_on_miss(object()) for _ in range(50)))
    assert backend.revision_reads == 1
    assert backend.loads == 1
    assert store.current.config_revision == 2


async def test_a_later_miss_reads_again(backend: _Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    store = await _loaded_store(backend)
    await store.refresh_on_miss(object())
    clock = [store._miss_read_at + store_mod.MISS_REFRESH_MIN_INTERVAL_S + 0.1]
    monkeypatch.setattr(store_mod.time, 'monotonic', lambda: clock[0])
    backend.revision = 3
    await store.refresh_on_miss(object())
    assert backend.revision_reads == 2
    assert store.current.config_revision == 3


async def test_failure_raises_without_staling_the_snapshot(backend: _Backend) -> None:
    store = await _loaded_store(backend)
    backend.fail = True
    with pytest.raises(ConfigStoreUnavailableError):
        await store.refresh_on_miss(object())
    assert store.current.stale is False


async def test_a_failed_burst_fails_every_caller_with_one_attempt(backend: _Backend) -> None:
    store = await _loaded_store(backend)
    backend.fail = True
    results = await asyncio.gather(
        *(store.refresh_on_miss(object()) for _ in range(20)), return_exceptions=True
    )
    assert all(isinstance(r, ConfigStoreUnavailableError) for r in results)
    assert backend.revision_reads == 1


async def test_pinned_store_stages_and_leaves_current(backend: _Backend) -> None:
    store = ConfigStore('idx', mode='pinned')
    await store.refresh(object())
    store.pin_active()
    backend.revision = 5
    await store.refresh_on_miss(object())
    assert store.current.config_revision == 1
    assert store.pending_snapshot is not None
    assert store.pending_snapshot.config_revision == 5
