"""The bulk-ingest runner against a fake ingest route and an in-memory ledger: bounded
concurrency, resume from the ledger, cancel, failure reporting and the stop-on-outage rule.

The queue itself (claims, retries, dead rows) is tested on PostgreSQL in
``test_ingest_source_files_claim.py`` and the whole job in ``tests/tasks/test_families_c_ingest.py``; here the
ledger is a list so each behaviour of the runner is observable on its own.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.domain.job_perf import JobPerfSummary
from src.services.curation.bulk_ingest import limits, runner, sender as sender_mod
from src.services.curation.bulk_ingest.runner import BulkRunAbortedError, ClaimedPath
from src.services.curation.bulk_ingest.source import (
    BulkSource,
    InvalidSourceError,
    iter_manifest_paths,
    validate_source,
)


@pytest.fixture(autouse=True)
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.services.curation import image_serving

    root = tmp_path / 'served'
    root.mkdir()
    monkeypatch.setattr(image_serving, '_configured_roots', lambda config=None: (root.resolve(),))  # noqa: ARG005
    return root


class MemoryLedger:
    """The queue of one source as a dict: ``pending`` -> ``claimed`` -> ``done``/``failed``."""

    def __init__(self, paths: list[str]) -> None:
        self.rows = dict.fromkeys(paths, 'pending')
        self.errors: dict[str, tuple[str, str]] = {}
        self.images: dict[str, str | None] = {}

    async def remaining(self) -> int:
        return sum(1 for s in self.rows.values() if s in {'pending', 'claimed', 'failed'})

    async def claim(self, limit: int) -> list[ClaimedPath]:
        picked = [k for k, s in self.rows.items() if s == 'pending'][:limit]
        for key in picked:
            self.rows[key] = 'claimed'
        return [ClaimedPath(k, k) for k in picked]

    async def settle(self, done: Any, failed: Any) -> None:
        for key, image in done:
            self.rows[key] = 'done'
            self.images[key] = image
        for key, error, kind in failed:
            self.rows[key] = 'failed'
            self.errors[key] = (error, kind)

    async def release(self) -> None:
        for key, state in self.rows.items():
            if state == 'claimed':
                self.rows[key] = 'pending'

    def count(self, state: str) -> int:
        return sum(1 for s in self.rows.values() if s == state)


class MemoryReporter:
    def __init__(self) -> None:
        self.stop = False
        self.snapshots: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []

    def cancelled(self) -> bool:
        return self.stop

    async def progress(
        self,
        done: int,
        total: int | None = None,
        stage: str | None = None,
        counters: Any = None,
        *,
        force: bool = False,  # noqa: ARG002
    ) -> None:
        self.snapshots.append({'done': done, 'total': total, 'stage': stage, **(counters or {})})

    async def checkpoint(self, cursor: Any) -> None:
        self.checkpoints.append(dict(cursor or {}))


class FakeSender:
    """Answers each batch as the ingest route would: ``success`` per path unless the
    path is listed as a duplicate or a failure. Records the call shape."""

    def __init__(
        self,
        *,
        duplicates: set[str] | None = None,
        failing: dict[str, str] | None = None,
        fail_all: bool = False,
    ) -> None:
        self.calls: list[list[str]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.duplicates = duplicates or set()
        self.failing = failing or {}
        self.fail_all = fail_all
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()
        self.run_ids: set[str] = set()
        self.stage_calls = 0

    async def post_batch(self, paths: list[str], *, source: str, run_id: str) -> dict[str, Any]:
        self.calls.append(list(paths))
        self.run_ids.add(run_id)
        assert source == 'tag'
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.entered.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0.005)
            if self.fail_all:
                raise sender_mod.BatchRequestError('HTTP 503: triton down')
            results = []
            for path in paths:
                name = Path(path).name
                if name in self.failing:
                    results.append(
                        {
                            'status': 'failed',
                            'image_path': path,
                            'error': self.failing[name],
                            'error_kind': 'unservable_path',
                        }
                    )
                    continue
                status = 'duplicate' if name in self.duplicates else 'success'
                results.append({'status': status, 'image_path': path, 'image_id': name})
            return {
                'summary': {
                    'crops_indexed': 2 * len(paths),
                    'n_embedded': 2 * len(paths),
                    'n_not_embedded': 0,
                    'n_embed_failed': 0,
                },
                'results': results,
            }
        finally:
            self.in_flight -= 1

    async def stage_seconds(self) -> dict[str, float]:
        self.stage_calls += 1
        return {'embed': 10.0 * self.stage_calls, 'decode': 1.0 * self.stage_calls}


def _paths(n: int) -> list[str]:
    return [f'/served/imgs/{i:03d}.jpg' for i in range(n)]


def _run(
    ledger: MemoryLedger,
    sender: FakeSender,
    *,
    batch_size: int = 4,
    concurrency: int = 2,
    reporter: MemoryReporter | None = None,
    prior: dict[str, Any] | None = None,
) -> tuple[runner.BulkIngestRun, MemoryReporter]:
    reporter = reporter or MemoryReporter()
    return (
        runner.BulkIngestRun(
            sender=sender,
            ledger=ledger,
            reporter=reporter,
            options=runner.BulkOptions('tag', batch_size, concurrency),
            run_id='run-1',
            prior=prior,
            servers=['triton-a'],
        ),
        reporter,
    )


def test_drains_the_queue_with_bounded_concurrency_and_reports_progress() -> None:
    ledger, sender = MemoryLedger(_paths(23)), FakeSender()
    run, reporter = _run(ledger, sender, concurrency=3)
    result = asyncio.run(run.run())

    assert (result['discovered'], result['ingested'], result['failed']) == (23, 23, 0)
    assert result['discovery_complete'] is True
    assert result['processed'] == 23
    assert result['crops_indexed'] == 46
    assert result['images_per_s'] > 0
    assert result['stage_s']['batch_requests'] > 0
    assert result['stage_s']['embed'] > 0  # the pipeline's own timers, as the API serves them
    assert 2 <= sender.max_in_flight <= 3
    assert sorted(p for call in sender.calls for p in call) == sorted(_paths(23))
    assert all(len(call) <= 4 for call in sender.calls)
    assert sender.run_ids == {'run-1'}
    assert ledger.count('done') == 23
    assert reporter.snapshots[-1]['done'] == 23
    assert reporter.checkpoints[-1]['ingested'] == 23
    JobPerfSummary.model_validate(result['perf_summary'])
    assert result['perf_summary']['servers'] == ['triton-a']


def test_counts_duplicates_apart_from_ingested_images() -> None:
    ledger = MemoryLedger(_paths(6))
    run, _ = _run(ledger, FakeSender(duplicates={'000.jpg', '001.jpg'}))
    result = asyncio.run(run.run())
    assert (result['ingested'], result['duplicates'], result['failed']) == (4, 2, 0)
    assert ledger.count('done') == 6
    assert ledger.images['/served/imgs/002.jpg'] == '002.jpg'


def test_images_the_route_refuses_are_failed_rows_with_their_reason_and_kind() -> None:
    ledger = MemoryLedger(_paths(4))
    sender = FakeSender(failing={'001.jpg': 'cannot read', '003.jpg': 'not under a source root'})
    run, _ = _run(ledger, sender)
    result = asyncio.run(run.run())

    assert (result['discovered'], result['ingested'], result['failed']) == (4, 2, 2)
    assert {Path(k).name: v for k, v in ledger.errors.items()} == {
        '001.jpg': ('cannot read', 'unservable_path'),
        '003.jpg': ('not under a source root', 'unservable_path'),
    }


def test_a_path_the_route_never_answered_for_is_a_failure_not_a_success() -> None:
    class Dropping(FakeSender):
        async def post_batch(self, paths: list[str], **kw: Any) -> dict[str, Any]:
            body = await super().post_batch(paths, **kw)
            body['results'] = body['results'][:-1]
            return body

    ledger = MemoryLedger(_paths(3))
    run, _ = _run(ledger, Dropping(), batch_size=3)
    result = asyncio.run(run.run())
    assert (result['ingested'], result['failed']) == (2, 1)
    assert list(ledger.errors.values()) == [('no result returned', runner.ERROR_KIND_BATCH)]


def test_a_resumed_run_continues_the_totals_of_the_earlier_attempt() -> None:
    ledger = MemoryLedger(_paths(8))
    for key in _paths(4):
        ledger.rows[key] = 'done'
    sender = FakeSender()
    prior = {'ingested': 4, 'duplicates': 0, 'crops_indexed': 8, 'n_embedded': 8}
    run, _ = _run(ledger, sender, prior=prior)
    result = asyncio.run(run.run())

    assert sorted(p for call in sender.calls for p in call) == _paths(8)[4:]
    assert result['discovered'] == 4  # what this attempt had left
    assert (result['ingested'], result['crops_indexed'], result['n_embedded']) == (8, 16, 16)


def test_a_failed_row_goes_back_to_the_queue_for_the_next_run() -> None:
    ledger = MemoryLedger(_paths(4))
    run, _ = _run(ledger, FakeSender(failing={'002.jpg': 'transient'}))
    asyncio.run(run.run())
    assert ledger.count('failed') == 1
    # the queue (not the runner) decides a failed row is claimable again; once it is pending
    ledger.rows[_paths(4)[2]] = 'pending'
    sender = FakeSender()
    run2, _ = _run(ledger, sender)
    result = asyncio.run(run2.run())
    assert sender.calls == [[_paths(4)[2]]]
    assert (result['ingested'], result['failed']) == (1, 0)


def test_cancel_stops_after_in_flight_batches_and_hands_the_claims_back() -> None:
    ledger = MemoryLedger(_paths(40))
    sender = FakeSender()
    sender.gate = asyncio.Event()
    run, reporter = _run(ledger, sender, batch_size=4, concurrency=2)

    async def scenario() -> dict[str, Any]:
        task = asyncio.create_task(run.run())
        await sender.entered.wait()
        reporter.stop = True
        sender.gate.set()  # type: ignore[union-attr]
        return await task

    result = asyncio.run(scenario())
    stored = sum(len(c) for c in sender.calls)
    assert 0 < stored < 40
    assert result['cancelled'] is True
    assert result['ingested'] == stored
    assert ledger.count('claimed') == 0  # nothing stays held by a run that stopped
    assert ledger.count('done') == stored
    assert ledger.count('pending') == 40 - stored


def test_a_dead_pipeline_aborts_the_run_instead_of_failing_every_image() -> None:
    ledger = MemoryLedger(_paths(60))
    sender = FakeSender(fail_all=True)
    run, _ = _run(ledger, sender, batch_size=2, concurrency=1)
    with pytest.raises(BulkRunAbortedError, match='triton down'):
        asyncio.run(run.run())

    assert len(sender.calls) == limits.MAX_CONSECUTIVE_FAILED_BATCHES
    assert ledger.count('failed') == 2 * limits.MAX_CONSECUTIVE_FAILED_BATCHES
    assert ledger.count('claimed') == 0


def test_a_crashing_batch_is_recorded_and_the_run_goes_on() -> None:
    class Crashing(FakeSender):
        async def post_batch(self, paths: list[str], **kw: Any) -> dict[str, Any]:
            if len(self.calls) == 0:
                self.calls.append(list(paths))
                raise ValueError('bad body')
            return await super().post_batch(paths, **kw)

    ledger = MemoryLedger(_paths(8))
    run, _ = _run(ledger, Crashing(), batch_size=4, concurrency=1)
    result = asyncio.run(run.run())
    assert (result['ingested'], result['failed']) == (4, 4)
    assert {e[0] for e in ledger.errors.values()} == {'bad body'}


def test_jsonl_manifest_entries_are_read_by_their_path_field(served: Path) -> None:
    folder = served / 'imgs'
    folder.mkdir()
    manifest = served / 'list.jsonl'
    manifest.write_text(
        f'{{"path": "{folder / "000.jpg"}", "note": "x"}}\n{{"nopath": 1}}\n{folder / "001.jpg"}\n'
    )
    assert list(iter_manifest_paths(manifest)) == [str(folder / '000.jpg'), str(folder / '001.jpg')]


def test_invalid_sources_are_refused_with_the_reason(served: Path, tmp_path: Path) -> None:
    with pytest.raises(InvalidSourceError, match='source root'):
        validate_source(BulkSource('folder', str(tmp_path)))
    with pytest.raises(InvalidSourceError, match='not a directory'):
        validate_source(BulkSource('folder', str(served / 'nope')))
    with pytest.raises(InvalidSourceError, match='not a file'):
        validate_source(BulkSource('manifest', str(served)))
    validate_source(BulkSource('folder', str(served)))
