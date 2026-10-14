"""Concurrent first requests must not race through the
``ensure_indexes`` bootstrap sequence independently -- the module-level
lock serializes them, and the inner re-check under the lock makes every
caller after the first a no-op once bootstrap has completed.
"""

from __future__ import annotations

import asyncio

import pytest

from src.services.curation import index_bootstrap


@pytest.fixture(autouse=True)
def _reset_bootstrapped_flag():
    index_bootstrap._INDEXES_BOOTSTRAPPED = set()
    yield
    index_bootstrap._INDEXES_BOOTSTRAPPED = set()


class _SlowFakeOpenSearch:
    """Stands in for a real OpenSearch client slow enough that several
    concurrent ``ensure_indexes`` callers overlap in time."""

    def __init__(self) -> None:
        self.create_calls = 0

    async def count(self, *, index: str, **_kw: object) -> dict[str, int]:  # noqa: ARG002
        return {'count': 1}


@pytest.mark.asyncio
async def test_concurrent_callers_run_the_bootstrap_sequence_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def _fake_create_curation_indexes(_opensearch: object, *, force_recreate: bool) -> None:
        nonlocal calls
        calls += 1
        # Yield control so overlapping callers actually interleave here
        # if the lock weren't in place.
        await asyncio.sleep(0.02)

    monkeypatch.setattr(index_bootstrap, 'create_curation_indexes', _fake_create_curation_indexes)
    # Every migration succeeds; one that fails leaves the project unbootstrapped (retried).
    for name in (
        n for n in vars(index_bootstrap) if n.startswith('ensure_') and n != 'ensure_indexes'
    ):

        async def _noop(_opensearch: object) -> dict[str, bool]:
            return {'acknowledged': True}

        monkeypatch.setattr(index_bootstrap, name, _noop)

    os_client = _SlowFakeOpenSearch()
    await asyncio.gather(*(index_bootstrap.ensure_indexes(os_client) for _ in range(8)))

    assert calls == 1, f'create_curation_indexes ran {calls} times, expected exactly 1'
    assert {'default'} == index_bootstrap._INDEXES_BOOTSTRAPPED


@pytest.mark.asyncio
async def test_already_bootstrapped_skips_the_lock_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast path (project already bootstrapped) must not even try to
    acquire the lock -- a locked-out lock (e.g. held by a hung bootstrap)
    would otherwise stall every request forever."""
    index_bootstrap._INDEXES_BOOTSTRAPPED = {'default'}
    await index_bootstrap._ensure_indexes_lock.acquire()  # simulate a stuck holder
    try:
        await asyncio.wait_for(index_bootstrap.ensure_indexes(object()), timeout=1.0)
    finally:
        index_bootstrap._ensure_indexes_lock.release()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
