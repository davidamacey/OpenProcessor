"""The bulk-ingest runner against a fake ingest route: bounded concurrency, the
ledger-driven resume, cancel, failure reporting and the stop-on-outage rule.

The pipeline behind ``POST /ingest/batch`` is exercised in
``tests/integration/test_bulk_ingest_routes.py``; here the route is replaced so each
behaviour of the runner is observable on its own.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.services.curation.bulk_ingest import limits, runner, sender as sender_mod
from src.services.curation.bulk_ingest.source import (
    BulkSource,
    InvalidSourceError,
    iter_source_paths,
    validate_source,
)
from src.services.curation.bulk_ingest.store import BulkStore, open_store


@pytest.fixture(autouse=True)
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.services.curation import image_serving

    root = tmp_path / 'served'
    root.mkdir()
    monkeypatch.setenv('OP_BULK_INGEST_DIR', str(tmp_path / 'bulk'))
    monkeypatch.setattr(image_serving, '_configured_roots', lambda config=None: (root.resolve(),))  # noqa: ARG005
    return root


def _images(root: Path, n: int, *, folder: str = 'imgs') -> Path:
    directory = root / folder
    directory.mkdir(exist_ok=True)
    for i in range(n):
        (directory / f'{i:03d}.jpg').write_bytes(b'img-%d' % i)
    return directory


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


def _start(
    sender: FakeSender, source: BulkSource, *, batch_size: int = 4, concurrency: int = 2
) -> tuple[BulkStore, bool, runner.BulkIngestRun]:
    options = runner.BulkOptions('tag', batch_size, concurrency)
    store, resumed = runner.claim_run(source, options)
    run = runner.BulkIngestRun(sender=sender, store=store, source=source, options=options)
    return store, resumed, run


def test_runs_a_folder_with_bounded_concurrency_and_reports_progress(served: Path) -> None:
    folder = _images(served, 23)
    sender = FakeSender()
    store, resumed, run = _start(sender, BulkSource('folder', str(folder)), concurrency=3)
    asyncio.run(run.run())

    state = store.job.read()
    assert not resumed
    assert state['status'] == 'completed'
    assert (state['discovered'], state['ingested'], state['failed']) == (23, 23, 0)
    assert state['discovery_complete'] is True
    assert state['processed'] == 23
    assert state['crops_indexed'] == 46
    assert state['images_per_s'] > 0
    assert state['stage_s']['batch_requests'] > 0
    assert state['stage_s']['embed'] > 0  # the pipeline's own timers, as the API serves them
    assert 2 <= sender.max_in_flight <= 3
    assert sorted(p for call in sender.calls for p in call) == sorted(
        str(p) for p in folder.iterdir()
    )
    assert all(len(call) <= 4 for call in sender.calls)
    assert sender.run_ids == {store.run_id}


def test_counts_duplicates_apart_from_ingested_images(served: Path) -> None:
    folder = _images(served, 6)
    sender = FakeSender(duplicates={'000.jpg', '001.jpg'})
    store, _, run = _start(sender, BulkSource('folder', str(folder)))
    asyncio.run(run.run())
    state = store.job.read()
    assert (state['ingested'], state['duplicates'], state['status']) == (4, 2, 'completed')


def test_images_the_route_refuses_are_reported_with_their_reason(served: Path) -> None:
    folder = _images(served, 4)
    manifest = served / 'list.txt'
    manifest.write_text(
        '# a comment\n' + '\n'.join(str(p) for p in sorted(folder.iterdir())) + '\n\n'
    )
    sender = FakeSender(failing={'001.jpg': 'cannot read', '003.jpg': 'not under a source root'})
    store, _, run = _start(sender, BulkSource('manifest', str(manifest)))
    asyncio.run(run.run())

    state = store.job.read()
    assert state['status'] == 'completed_with_errors'
    assert (state['discovered'], state['ingested'], state['failed']) == (4, 2, 2)
    rows, total = store.failures(offset=0, limit=10)
    assert total == 2
    assert {Path(r['path']).name: r['error'] for r in rows} == {
        '001.jpg': 'cannot read',
        '003.jpg': 'not under a source root',
    }
    assert store.done()[1] == 2


def test_a_path_the_route_never_answered_for_is_a_failure_not_a_success(served: Path) -> None:
    folder = _images(served, 3)

    class Dropping(FakeSender):
        async def post_batch(self, paths: list[str], **kw: Any) -> dict[str, Any]:
            body = await super().post_batch(paths, **kw)
            body['results'] = body['results'][:-1]
            return body

    store, _, run = _start(Dropping(), BulkSource('folder', str(folder)), batch_size=3)
    asyncio.run(run.run())
    state = store.job.read()
    assert (state['ingested'], state['failed'], state['status']) == (2, 1, 'completed_with_errors')
    assert store.failures(offset=0, limit=5)[0][0]['error'] == 'no result returned'


def test_jsonl_manifest_entries_are_read_by_their_path_field(served: Path) -> None:
    folder = _images(served, 2)
    manifest = served / 'list.jsonl'
    manifest.write_text(
        f'{{"path": "{folder / "000.jpg"}", "note": "x"}}\n{{"nopath": 1}}\n{folder / "001.jpg"}\n'
    )
    assert list(iter_source_paths(BulkSource('manifest', str(manifest)))) == [
        str(folder / '000.jpg'),
        str(folder / '001.jpg'),
    ]


def test_resume_skips_what_the_ledger_holds(served: Path) -> None:
    folder = _images(served, 8)
    source = BulkSource('folder', str(folder))
    first, _, run = _start(FakeSender(), source)
    asyncio.run(run.run())
    # Interrupt: the run only got through the first four before it died.
    (first.directory / 'done.jsonl').write_text(
        ''.join(f'{{"p": "{folder / f"{i:03d}.jpg"}", "o": "ingested"}}\n' for i in range(4))
    )
    first.job.update(status='interrupted')

    sender = FakeSender()
    store, resumed, run2 = _start(sender, source)
    asyncio.run(run2.run())

    assert resumed
    assert store.run_id == first.run_id
    state = store.job.read()
    assert (state['attempts'], state['skipped'], state['ingested']) == (2, 4, 8)
    # Totals continue across attempts: 4 images stored by the first attempt's counters
    # (8 images x 2 crops) plus this attempt's 4 x 2.
    assert (state['crops_indexed'], state['n_embedded']) == (16 + 8, 16 + 8)
    assert state['status'] == 'completed'
    assert sorted(p for call in sender.calls for p in call) == [
        str(folder / f'{i:03d}.jpg') for i in range(4, 8)
    ]


def test_resume_retries_the_images_that_failed(served: Path) -> None:
    folder = _images(served, 4)
    source = BulkSource('folder', str(folder))
    first, _, run = _start(FakeSender(failing={'002.jpg': 'transient'}), source)
    asyncio.run(run.run())
    assert first.job.read()['status'] == 'completed_with_errors'

    sender = FakeSender()
    second, resumed, run2 = _start(sender, source)
    asyncio.run(run2.run())

    assert resumed
    assert second.run_id == first.run_id
    assert sender.calls == [[str(folder / '002.jpg')]]
    state = second.job.read()
    assert (state['status'], state['ingested'], state['failed']) == ('completed', 4, 0)
    assert second.failures(offset=0, limit=10)[1] == 0


def test_a_completed_run_is_not_resumed_a_new_one_starts(served: Path) -> None:
    folder = _images(served, 3)
    source = BulkSource('folder', str(folder))
    first, _, run = _start(FakeSender(), source)
    asyncio.run(run.run())
    second, resumed, _ = _start(FakeSender(), source)
    assert not resumed
    assert second.run_id != first.run_id


def test_start_refuses_while_a_run_is_live_and_names_it(served: Path) -> None:
    folder = _images(served, 3)
    store, _, _ = _start(FakeSender(), BulkSource('folder', str(folder)))
    with pytest.raises(runner.BulkBusyError) as exc:
        runner.claim_run(BulkSource('folder', str(folder)), runner.BulkOptions('tag'))
    assert exc.value.run_id == store.run_id


def test_cancel_stops_after_in_flight_batches_and_a_start_continues(served: Path) -> None:
    folder = _images(served, 40)
    source = BulkSource('folder', str(folder))

    async def scenario() -> tuple[BulkStore, FakeSender]:
        sender = FakeSender()
        sender.gate = asyncio.Event()
        store, _, run = _start(sender, source, batch_size=4, concurrency=2)
        task = asyncio.create_task(run.run())
        await sender.entered.wait()
        runner.cancel_run(store)
        sender.gate.set()
        await task
        return store, sender

    store, sender = asyncio.run(scenario())
    state = store.job.read()
    assert state['status'] == 'cancelled'
    stored = sum(len(c) for c in sender.calls)
    assert 0 < stored < 40
    assert state['ingested'] == stored

    sender2 = FakeSender()
    store2, resumed, run2 = _start(sender2, source)
    asyncio.run(run2.run())
    assert resumed
    assert store2.run_id == store.run_id
    assert store2.job.read()['ingested'] == 40
    assert sum(len(c) for c in sender2.calls) == 40 - stored
    assert not store2.job.cancel_requested()


def test_a_dead_pipeline_fails_the_run_instead_of_failing_every_image(served: Path) -> None:
    folder = _images(served, 60)
    sender = FakeSender(fail_all=True)
    store, _, run = _start(sender, BulkSource('folder', str(folder)), batch_size=2, concurrency=1)
    asyncio.run(run.run())

    state = store.job.read()
    assert state['status'] == 'failed'
    assert 'triton down' in state['error']
    assert len(sender.calls) == limits.MAX_CONSECUTIVE_FAILED_BATCHES
    assert state['failed'] == 2 * limits.MAX_CONSECUTIVE_FAILED_BATCHES
    assert store.done()[0] == set()


def test_a_dead_pipeline_run_can_be_started_again_to_resume(served: Path) -> None:
    folder = _images(served, 10)
    source = BulkSource('folder', str(folder))
    first, _, run = _start(FakeSender(fail_all=True), source, batch_size=2, concurrency=1)
    asyncio.run(run.run())
    sender = FakeSender()
    second, resumed, run2 = _start(sender, source)
    asyncio.run(run2.run())
    assert resumed
    assert second.run_id == first.run_id
    assert second.job.read()['status'] == 'completed'
    assert second.job.read()['ingested'] == 10


def test_invalid_sources_are_refused_with_the_reason(served: Path, tmp_path: Path) -> None:
    with pytest.raises(InvalidSourceError, match='source root'):
        validate_source(BulkSource('folder', str(tmp_path)))
    with pytest.raises(InvalidSourceError, match='not a directory'):
        validate_source(BulkSource('folder', str(served / 'nope')))
    with pytest.raises(InvalidSourceError, match='not a file'):
        validate_source(BulkSource('manifest', str(served)))
    validate_source(BulkSource('folder', str(served)))


def test_run_ids_are_validated_before_they_name_a_directory(served: Path) -> None:
    assert open_store('../../etc') is None
    assert open_store('bulk_20260101T000000_deadbeef') is None
