"""F-28.4: concurrent first requests must not race through the
``_ensure_indexes`` bootstrap sequence independently -- the module-level
lock serializes them, and the inner re-check under the lock makes every
caller after the first a no-op once bootstrap has completed.
"""

from __future__ import annotations

import asyncio

import pytest

from src.routers.curation import _common


@pytest.fixture(autouse=True)
def _reset_bootstrapped_flag():
    _common._INDEXES_BOOTSTRAPPED = False
    yield
    _common._INDEXES_BOOTSTRAPPED = False


class _SlowFakeOpenSearch:
    """Stands in for a real OpenSearch client slow enough that several
    concurrent ``_ensure_indexes`` callers overlap in time."""

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

    monkeypatch.setattr(_common, 'create_curation_indexes', _fake_create_curation_indexes)
    for name in (
        'ensure_items_vlm_raw_label_fields',
        'ensure_items_label_cluster_fields',
        'ensure_items_provenance_fields',
        'ensure_items_request_id_field',
        'ensure_items_quality_fields',
        'ensure_items_region_embedding',
        'ensure_items_score_fields',
        'ensure_items_probe_fields',
        'ensure_items_viz_fields',
        'ensure_items_history_fields',
        'ensure_items_exclusion_fields',
        'ensure_items_text_reader_fields',
        'ensure_items_validation_split_fields',
        'ensure_items_pe_v6_embedding_fields',
    ):

        async def _noop(_opensearch: object) -> None:
            return None

        monkeypatch.setattr(_common, name, _noop)

    os_client = _SlowFakeOpenSearch()
    await asyncio.gather(*(_common._ensure_indexes(os_client) for _ in range(8)))

    assert calls == 1, f'create_curation_indexes ran {calls} times, expected exactly 1'
    assert _common._INDEXES_BOOTSTRAPPED is True


@pytest.mark.asyncio
async def test_already_bootstrapped_skips_the_lock_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast path (flag already true) must not even try to acquire the
    lock -- a locked-out lock (e.g. held by a hung bootstrap) would
    otherwise stall every request forever."""
    _common._INDEXES_BOOTSTRAPPED = True
    await _common._ensure_indexes_lock.acquire()  # simulate a stuck holder
    try:
        await asyncio.wait_for(_common._ensure_indexes(object()), timeout=1.0)
    finally:
        _common._ensure_indexes_lock.release()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
