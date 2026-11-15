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
from src.services.curation.bulk_ingest.runner import BulkRunAbortedError, ClaimedPath, RetryWait
from src.services.curation.bulk_ingest.source import (
    BulkSource,
    InvalidSourceError,
    iter_manifest_paths,
    validate_source,
)
from src.services.curation.bulk_ingest.source_queue import failure_row


@pytest.fixture(autouse=True)
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.services.curation import image_serving

    root = tmp_path / 'served'
    root.mkdir()
    monkeypatch.setattr(image_serving, '_configured_roots', lambda config=None: (root.resolve(),))
    return root


class MemoryLedger:
    """The queue of one source as a dict: ``pending`` -> ``claimed`` -> ``done``/``failed``.

    A failure the real ledger turns into a pause (``failure_row``) leaves the row ``waiting``:
    claimable again once ``now`` (seconds the run has waited, advanced by the patched
    ``WAIT_SLEEP``) reaches its due time. Any other failed row stays failed until a test resets it.
    Whether a row's attempts move is SQL, tested on PostgreSQL; here ``attempts_used`` records what
    the ledger was told."""

    def __init__(self, paths: list[str]) -> None:
        self.rows = dict.fromkeys(paths, 'pending')
        self.errors: dict[str, tuple[str, str]] = {}
        self.images: dict[str, str | None] = {}
        self.now = 0.0
        self.due: dict[str, float] = {}
        self.pauses: dict[str, list[float]] = {}
        self.attempts_used: dict[str, int] = {}
        self.dead: set[str] = set()

    async def remaining(self) -> int:
        return sum(1 for s in self.rows.values() if s in {'pending', 'claimed', 'failed'})

    async def claim(self, limit: int) -> list[ClaimedPath]:
        picked = [
            k
            for k, s in self.rows.items()
            if s == 'pending' or (s == 'failed' and k in self.due and self.due[k] <= self.now)
        ][:limit]
        for key in picked:
            self.rows[key] = 'claimed'
            self.due.pop(key, None)
        return [ClaimedPath(k, k) for k in picked]

    async def settle(self, done: Any, failed: Any, retry_after_s: float | None = None) -> None:
        for key, image in done:
            self.rows[key] = 'done'
            self.images[key] = image
        for key, error, kind in failed:
            row = failure_row(key, error, kind, retry_after_s)
            self.rows[key] = 'failed'
            self.errors[key] = (error, kind)
            if row.terminal:
                self.dead.add(key)
            elif row.retry_in_s is None:
                self.attempts_used[key] = self.attempts_used.get(key, 0) + 1
            else:
                self.due[key] = self.now + row.retry_in_s
                self.pauses.setdefault(key, []).append(row.retry_in_s)

    async def waiting(self) -> RetryWait | None:
        if not self.due:
            return None
        return RetryWait(len(self.due), max(0.0, min(self.due.values()) - self.now))

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
        force: bool = False,
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
        busy_after: float | None = None,
        transient_once: dict[str, str] | None = None,
        busy_first_calls: int = 0,
    ) -> None:
        self.busy_after = busy_after
        self.busy_first_calls = busy_first_calls
        self.transient_once = dict(transient_once or {})
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
            if self.busy_after is not None:
                raise sender_mod.BatchBusyError('HTTP 503: engine down', self.busy_after)
            if self.busy_first_calls > 0:
                self.busy_first_calls -= 1
                raise sender_mod.BatchBusyError('HTTP 503: engine down', 0.0)
            if self.fail_all:
                raise sender_mod.BatchRequestError('HTTP 503: triton down')
            results = []
            routes: dict[str, int] = {}
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
                if name in self.transient_once:
                    results.append(
                        {
                            'status': 'failed',
                            'image_path': path,
                            'error': 'engine restarting',
                            'error_kind': self.transient_once.pop(name),
                        }
                    )
                    continue
                status = 'duplicate' if name in self.duplicates else 'success'
                if status == 'success':
                    route = 'gpu' if int(Path(name).stem) % 2 == 0 else 'cpu_small'
                    routes[route] = routes.get(route, 0) + 1
                results.append({'status': status, 'image_path': path, 'image_id': name})
            return {
                'summary': {
                    'crops_indexed': 2 * len(paths),
                    'n_embedded': 2 * len(paths),
                    'n_not_embedded': 0,
                    'n_embed_failed': 0,
                    'decode_routes': routes,
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
    **extra: Any,
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
            **extra,
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


def test_the_run_reports_the_unreadable_entries_discovery_gave_it() -> None:
    run, _ = _run(MemoryLedger([]), FakeSender(), unreadable=2, unreadable_samples=['a.jpg'])
    result = asyncio.run(run.run())

    assert result['discovered'] == 0
    assert result['unreadable'] == 2
    assert result['unreadable_samples'] == ['a.jpg']


def test_the_run_tallies_the_decode_routes_of_the_images_it_ingested() -> None:
    ledger, sender = MemoryLedger(_paths(10)), FakeSender(duplicates={'003.jpg'})
    run, reporter = _run(ledger, sender)
    result = asyncio.run(run.run())

    assert result['decode_routes'] == {'gpu': 5, 'cpu_small': 4}
    assert sum(result['decode_routes'].values()) == result['ingested']
    assert reporter.snapshots[-1]['decode_routes'] == result['decode_routes']
    assert JobPerfSummary.model_validate(result['perf_summary']).decode_routes == {
        'gpu': 5,
        'cpu_small': 4,
    }


def test_a_resumed_run_continues_the_decode_route_tally() -> None:
    ledger, sender = MemoryLedger(_paths(4)), FakeSender()
    run, reporter = _run(ledger, sender, prior={'ingested': 6, 'decode_routes': {'gpu': 6}})
    result = asyncio.run(run.run())

    assert result['decode_routes'] == {'gpu': 8, 'cpu_small': 2}
    assert reporter.checkpoints[-1]['decode_routes'] == {'gpu': 8, 'cpu_small': 2}


# --- transient failures: no attempt, a pause, and a run that waits instead of stalling or spinning


def _waits(monkeypatch: pytest.MonkeyPatch, ledger: MemoryLedger) -> list[float]:
    """Replace the run's wait with a fake clock: each slice advances the ledger's ``now``."""
    steps: list[float] = []

    async def sleep(seconds: float) -> None:
        steps.append(seconds)
        ledger.now += seconds

    monkeypatch.setattr(runner, 'WAIT_SLEEP', sleep)
    return steps


def test_a_file_that_failed_transiently_waits_for_its_pause_and_is_then_stored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = MemoryLedger(_paths(4))
    steps = _waits(monkeypatch, ledger)
    sender = FakeSender(transient_once={'002.jpg': 'engine_unavailable'})
    run, reporter = _run(ledger, sender)
    result = asyncio.run(run.run())

    assert ledger.count('done') == 4  # the run did not end with the file left waiting
    assert (result['ingested'], result['waiting']) == (4, 0)
    assert ledger.pauses == {_paths(4)[2]: [30.0]}
    assert ledger.attempts_used == {}
    assert ledger.dead == set()
    assert sum(steps) == pytest.approx(30.0)
    assert all(step <= 1.0 for step in steps)  # a cancel is seen within a second
    assert result['stage_s']['retry_wait'] == pytest.approx(30.0)
    # the wait is visible in the job's progress: its stage and the number of waiting files
    waits = [s for s in reporter.snapshots if s['stage'] == 'retry_wait']
    assert waits
    assert waits[0]['waiting'] == 1
    assert reporter.snapshots[-1]['stage'] == 'ingest'
    assert sender.calls[-1] == [_paths(4)[2]]


def test_a_file_that_waited_and_was_then_stored_is_not_a_failure_of_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = MemoryLedger(_paths(4))
    _waits(monkeypatch, ledger)
    sender = FakeSender(transient_once={'002.jpg': 'engine_unavailable'})
    run, reporter = _run(ledger, sender)
    result = asyncio.run(run.run())

    assert ledger.count('done') == 4
    assert result['failed'] == 0  # the run ends `completed`, not `completed_with_errors`
    assert result['processed'] == 4
    # while it waited it was waiting, not failed, and the run's remaining count still held it
    waits = [s for s in reporter.snapshots if s['stage'] == 'retry_wait']
    assert (waits[0]['failed'], waits[0]['waiting']) == (0, 1)
    assert all(s['failed'] <= s['discovered'] - s['waiting'] for s in reporter.snapshots)


def test_a_batch_whose_503_retries_ran_out_and_was_stored_later_is_not_a_failure_of_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = MemoryLedger(_paths(4))
    _waits(monkeypatch, ledger)
    run, reporter = _run(ledger, FakeSender(busy_first_calls=1), batch_size=4, concurrency=1)
    result = asyncio.run(run.run())

    assert ledger.count('done') == 4
    assert (result['ingested'], result['failed'], result['processed']) == (4, 0, 4)
    waits = [s for s in reporter.snapshots if s['stage'] == 'retry_wait']
    assert (waits[0]['failed'], waits[0]['waiting']) == (0, 4)


def test_a_resumed_run_that_finds_only_waiting_files_waits_and_stores_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = MemoryLedger(_paths(3))
    for key in _paths(3):  # what an earlier run left: failed, 20 s from due
        ledger.rows[key] = 'failed'
        ledger.due[key] = 20.0
    steps = _waits(monkeypatch, ledger)
    sender = FakeSender()
    run, _ = _run(ledger, sender, batch_size=3)
    result = asyncio.run(run.run())

    assert (result['discovered'], result['ingested'], result['failed']) == (3, 3, 0)
    assert ledger.count('done') == 3
    assert sum(steps) == pytest.approx(20.0)
    assert sender.calls == [_paths(3)]


def test_a_stop_request_ends_the_wait_within_one_slice_and_leaves_the_files_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = MemoryLedger(_paths(2))
    reporter = MemoryReporter()
    steps = _waits(monkeypatch, ledger)
    inner = runner.WAIT_SLEEP

    async def stop_after_two(seconds: float) -> None:
        await inner(seconds)
        reporter.stop = len(steps) >= 2

    monkeypatch.setattr(runner, 'WAIT_SLEEP', stop_after_two)
    sender = FakeSender(transient_once={'000.jpg': 'database_busy', '001.jpg': 'database_busy'})
    run, _ = _run(ledger, sender, reporter=reporter)
    result = asyncio.run(run.run())

    assert result['cancelled'] is True
    assert len(steps) == 2
    assert ledger.count('failed') == 2
    assert ledger.count('claimed') == 0
    assert ledger.dead == set()
    assert ledger.attempts_used == {}


def test_a_row_that_is_due_but_never_claimable_does_not_make_the_run_spin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another runner holds the row locked: ``waiting`` says 0 s, ``claim`` says nothing."""

    class Locked(MemoryLedger):
        async def waiting(self) -> RetryWait | None:
            return RetryWait(1, 0.0)

    ledger = Locked([])
    reporter = MemoryReporter()
    steps: list[float] = []

    async def sleep(seconds: float) -> None:
        steps.append(seconds)
        reporter.stop = len(steps) >= 3

    monkeypatch.setattr(runner, 'WAIT_SLEEP', sleep)
    run, _ = _run(ledger, FakeSender(), reporter=reporter)
    asyncio.run(run.run())
    assert steps == [limits.RETRY_WAIT_SLICE_S] * 3  # a pause per round, never a tight loop


def test_a_failure_that_is_not_transient_is_not_waited_for(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = MemoryLedger(_paths(3))
    steps = _waits(monkeypatch, ledger)
    run, _ = _run(ledger, FakeSender(failing={'001.jpg': 'cannot read'}))
    result = asyncio.run(run.run())

    assert steps == []
    assert result['stage_s']['retry_wait'] == 0
    assert ledger.dead == {_paths(3)[1]}  # terminal, exactly as before


def test_a_batch_whose_503_retries_ran_out_is_recorded_busy_with_the_retry_after() -> None:
    ledger = MemoryLedger(_paths(12))
    sender = FakeSender(busy_after=90.0)
    run, _ = _run(ledger, sender, batch_size=2, concurrency=1)
    with pytest.raises(BulkRunAbortedError, match='engine down'):
        asyncio.run(run.run())

    # 5 such batches in a row still stop the run, and no file is dead or used an attempt
    assert len(sender.calls) == limits.MAX_CONSECUTIVE_FAILED_BATCHES
    assert {kind for _, kind in ledger.errors.values()} == {'busy'}
    assert ledger.count('failed') == 2 * limits.MAX_CONSECUTIVE_FAILED_BATCHES
    assert ledger.dead == set()
    assert ledger.attempts_used == {}
    assert {p for ps in ledger.pauses.values() for p in ps} == {90.0}
    assert ledger.count('claimed') == 0


def test_a_batch_that_failed_for_another_reason_still_uses_an_attempt() -> None:
    ledger = MemoryLedger(_paths(12))
    run, _ = _run(ledger, FakeSender(fail_all=True), batch_size=2, concurrency=1)
    with pytest.raises(BulkRunAbortedError):
        asyncio.run(run.run())
    assert {kind for _, kind in ledger.errors.values()} == {runner.ERROR_KIND_BATCH}
    assert set(ledger.attempts_used.values()) == {1}
    assert not ledger.pauses
