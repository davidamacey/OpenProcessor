"""A by-name miss re-reads the registry cheaply and at most once per burst (#442).

The forced read used to be a full reload on every miss: N concurrent misses on an unknown
name (the API is unauthenticated) meant N full snapshot loads.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from src.services.config_store import store as store_mod
from src.services.config_store.snapshot import _EMPTY_SNAPSHOT, ConfigSnapshot
from src.services.config_store.store import ConfigStore, ConfigStoreUnavailableError


pytestmark = pytest.mark.asyncio


class _Backend:
    """Stands in for the database: an epoch counter and a snapshot loader."""

    def __init__(self) -> None:
        self.revision = 1
        self.revision_reads = 0
        self.loads = 0
        self.fail = False

    async def get_revision(self, _store: ConfigStore) -> int:
        self.revision_reads += 1
        await asyncio.sleep(0)
        if self.fail:
            raise OSError('database down')
        return self.revision

    async def load(self, _store: ConfigStore) -> ConfigSnapshot:
        self.loads += 1
        await asyncio.sleep(0)
        if self.fail:
            raise OSError('database down')
        return replace(_EMPTY_SNAPSHOT, config_revision=self.revision, loaded_at=1.0)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> _Backend:
    b = _Backend()

    async def epoch(store: ConfigStore) -> int:
        return await b.get_revision(store)

    async def load(store: ConfigStore) -> ConfigSnapshot:
        return await b.load(store)

    monkeypatch.setattr(ConfigStore, '_epoch', epoch)
    monkeypatch.setattr(ConfigStore, '_load', load)
    return b


async def _loaded_store(backend: _Backend) -> ConfigStore:
    store = ConfigStore(label='idx')
    await store.refresh()
    backend.loads = backend.revision_reads = 0
    return store


async def test_unchanged_revision_costs_no_reload(backend: _Backend) -> None:
    store = await _loaded_store(backend)
    await store.refresh_on_miss()
    assert backend.loads == 0
    assert backend.revision_reads == 1


async def test_moved_revision_reloads_and_publishes(backend: _Backend) -> None:
    store = await _loaded_store(backend)
    backend.revision = 2
    got = await store.refresh_on_miss()
    assert backend.loads == 1
    assert got.config_revision == 2
    assert store.current.config_revision == 2


async def test_a_storm_of_misses_shares_one_read(backend: _Backend) -> None:
    store = await _loaded_store(backend)
    backend.revision = 2
    await asyncio.gather(*(store.refresh_on_miss() for _ in range(50)))
    assert backend.revision_reads == 1
    assert backend.loads == 1
    assert store.current.config_revision == 2


async def test_a_later_miss_reads_again(backend: _Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    store = await _loaded_store(backend)
    await store.refresh_on_miss()
    clock = [store._miss_read_at + store_mod.MISS_REFRESH_MIN_INTERVAL_S + 0.1]
    monkeypatch.setattr(store_mod.time, 'monotonic', lambda: clock[0])
    backend.revision = 3
    await store.refresh_on_miss()
    assert backend.revision_reads == 2
    assert store.current.config_revision == 3


async def test_failure_raises_without_staling_the_snapshot(backend: _Backend) -> None:
    store = await _loaded_store(backend)
    backend.fail = True
    with pytest.raises(ConfigStoreUnavailableError):
        await store.refresh_on_miss()
    assert store.current.stale is False


async def test_a_failed_burst_fails_every_caller_with_one_attempt(backend: _Backend) -> None:
    store = await _loaded_store(backend)
    backend.fail = True
    results = await asyncio.gather(
        *(store.refresh_on_miss() for _ in range(20)), return_exceptions=True
    )
    assert all(isinstance(r, ConfigStoreUnavailableError) for r in results)
    assert backend.revision_reads == 1


async def test_pinned_store_stages_and_leaves_current(backend: _Backend) -> None:
    store = ConfigStore(label='idx', mode='pinned')
    await store.refresh()
    store.pin_active()
    backend.revision = 5
    await store.refresh_on_miss()
    assert store.current.config_revision == 1
    assert store.pending_snapshot is not None
    assert store.pending_snapshot.config_revision == 5
