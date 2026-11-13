"""The location-purge step: every stored coordinate is brought within the policy, counts only (CM-4)."""

from __future__ import annotations

from typing import Any

import pytest

from curation.capture_backfill_fakes import BackfillCtx, FakeBackfillOpenSearch, bind_backfill
from src.services.source_backfill import job
from src.tasks.runtime.context import JobResult


def _image(lat: float, lon: float, **extra: Any) -> dict[str, Any]:
    return {
        'image_path': '/x.jpg',
        'capture': {
            'parser': 'capture@1',
            'has_location': True,
            'location': {'lat': lat, 'lon': lon},
            'altitude_m': 410.0,
            'location_accuracy_m': 4.0,
            'heading_deg': 90.0,
            'location_precision': 'exact',
            'camera_make': 'Acme',
            **extra,
        },
    }


@pytest.mark.asyncio
async def test_a_switch_to_drop_removes_every_position_field_and_a_rerun_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind_backfill(monkeypatch, 'drop')
    images: dict[str, dict[str, Any]] = {
        'a': _image(47.123456, 8.987654),
        'b': _image(-33.9, 151.2),
        'c': {'image_path': '/y'},
    }
    client = FakeBackfillOpenSearch(images, {})

    first = await job.run(BackfillCtx(client, ['location_purge']))

    assert isinstance(first, JobResult)
    assert first.result is not None
    assert first.result['location_purged'] == 2
    for key in ('a', 'b'):
        cap = images[key]['capture']
        assert not {'location', 'altitude_m', 'location_accuracy_m', 'heading_deg'} & set(cap)
        assert cap['has_location'] is True
        assert cap['camera_make'] == 'Acme'
    assert 'capture' not in images['c']
    assert '47.12' not in str(first.result)

    second = await job.run(BackfillCtx(client, ['location_purge']))
    assert isinstance(second, JobResult)
    assert second.result is not None
    assert second.result['location_purged'] == 0


@pytest.mark.asyncio
async def test_a_switch_to_store_coarse_rounds_exact_positions_and_leaves_coarse_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind_backfill(monkeypatch, 'store_coarse')
    images: dict[str, dict[str, Any]] = {
        'a': _image(47.123456, 8.987654),
        'b': _image(10.0, 20.0, location_precision='coarse'),
    }
    client = FakeBackfillOpenSearch(images, {})

    first = await job.run(BackfillCtx(client, ['location_purge']))

    assert isinstance(first, JobResult)
    assert first.result is not None
    assert first.result['location_purged'] == 1
    cap = images['a']['capture']
    assert (cap['location'], cap['location_precision']) == ({'lat': 47.12, 'lon': 8.99}, 'coarse')
    assert 'altitude_m' not in cap
    assert images['b']['capture']['altitude_m'] == 410.0


@pytest.mark.asyncio
async def test_store_leaves_everything_and_asks_nothing_of_the_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind_backfill(monkeypatch, 'store')
    images: dict[str, dict[str, Any]] = {'a': _image(47.1, 8.9)}
    client = FakeBackfillOpenSearch(images, {})
    await job.run(BackfillCtx(client, ['location_purge']))
    assert client.by_query_calls == []
    assert images['a']['capture']['location'] == {'lat': 47.1, 'lon': 8.9}


@pytest.mark.asyncio
async def test_the_pass_is_bounded_and_the_first_pass_waits_for_every_cached_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind_backfill(monkeypatch, 'drop')
    waits: list[int] = []

    async def settle() -> None:
        waits.append(1)

    monkeypatch.setattr(job, 'settle', settle)
    client = FakeBackfillOpenSearch({'a': _image(1.5, 2.5)}, {})
    ctx = BackfillCtx(client, ['location_purge'])
    await job.run(ctx)
    assert waits == [1]
    assert client.by_query_calls[0]['max_docs'] == 5000
    assert client.by_query_calls[0]['conflicts'] == 'proceed'

    resumed = BackfillCtx(client, ['location_purge'], resume_from=ctx.checkpoints[-1])
    await job.run(resumed)
    assert waits == [1]


@pytest.mark.asyncio
async def test_the_purge_reads_the_stored_policy_not_this_workers_cached_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earlier job in the same worker cached `store`; the switch to `drop` was stored after."""
    bind_backfill(monkeypatch, 'store')

    async def read(*, fresh: bool = False) -> tuple[str, int]:
        return ('drop', 2) if fresh else ('store', 1)

    monkeypatch.setattr(job, 'read_location_policy', read)
    images = {'a': _image(1.5, 2.5)}
    client = FakeBackfillOpenSearch(images, {})
    await job.run(BackfillCtx(client, ['location_purge']))
    assert client.by_query_calls, 'the purge ran under the stale cached policy and did nothing'
    assert 'location' not in images['a']['capture']


@pytest.mark.asyncio
async def test_an_unknown_step_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_backfill(monkeypatch, 'drop')
    with pytest.raises(job.UnknownStepError):
        await job.run(BackfillCtx(FakeBackfillOpenSearch({}, {}), ['frobnicate']))


class _ConflictingOpenSearch(FakeBackfillOpenSearch):
    """The first ``conflict_passes`` passes skip every document on a version conflict, as
    `conflicts=proceed` does when a concurrent writer touched them."""

    def __init__(self, images: dict[str, dict[str, Any]], conflict_passes: int | None) -> None:
        super().__init__(images, {})
        self.conflict_passes = conflict_passes

    async def update_by_query(
        self, *, index: str, body: dict[str, Any], **kw: Any
    ) -> dict[str, Any]:
        if self.conflict_passes is None or len(self.by_query_calls) < self.conflict_passes:
            self.by_query_calls.append({'index': index, 'body': body, **kw})
            return {'updated': 0, 'version_conflicts': 1}
        return await super().update_by_query(index=index, body=body, **kw)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []

    async def sleep(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(job.asyncio, 'sleep', sleep)
    return waited


@pytest.mark.asyncio
async def test_a_pass_with_conflicts_then_a_clean_pass_completes_and_counts_only_updates(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    bind_backfill(monkeypatch, 'store_coarse')
    monkeypatch.setattr(job, 'settle', lambda: _noop())
    images = {'a': _image(47.123456, 8.987654)}
    client = _ConflictingOpenSearch(images, conflict_passes=2)

    result = await job.run(BackfillCtx(client, ['location_purge']))

    assert isinstance(result, JobResult)
    assert result.result is not None
    assert result.result['location_purged'] == 1
    assert images['a']['capture']['location_precision'] == 'coarse'
    assert sleeps == [job.CONFLICT_BACKOFF_S] * 2


@pytest.mark.asyncio
async def test_persistent_conflicts_fail_the_job_instead_of_completing(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    bind_backfill(monkeypatch, 'store_coarse')
    monkeypatch.setattr(job, 'settle', lambda: _noop())
    images = {'a': _image(47.123456, 8.987654)}
    client = _ConflictingOpenSearch(images, conflict_passes=None)

    with pytest.raises(job.PurgeIncompleteError, match='1 documents still hold coordinates'):
        await job.run(BackfillCtx(client, ['location_purge']))

    assert len(client.by_query_calls) == job.MAX_CONFLICT_PASSES
    assert images['a']['capture']['location_precision'] == 'exact'


async def _noop() -> None:
    return None
