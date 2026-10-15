"""A dispatcher-started run actually starts, a failed start leaves no orphan, and cancel
finalizes an orphan (live S3 check): ``run_tick`` used to call ``asyncio.create_task`` from
the ``to_thread`` worker (``no running event loop``), after ``claim_run`` had already
written a ``queued`` run that then blocked every later start and the project's delete."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from typing import TYPE_CHECKING, Any

import pytest

from src.services.curation.bulk_ingest import dispatcher, runner, source_runs
from src.services.curation.bulk_ingest.object_queue import QueueEntry
from src.services.curation.bulk_ingest.source import BulkSource
from src.services.curation.bulk_ingest.sources import IngestSource, save_source, source_queue
from src.services.curation.bulk_ingest.store import active_run, list_stores


if TYPE_CHECKING:
    from pathlib import Path


class _Sender:
    async def post_batch(self, paths: list[str], *, source: str, run_id: str) -> dict[str, Any]:  # noqa: ARG002
        return {
            'summary': {},
            'results': [{'status': 'success', 'image_path': p, 'image_id': p} for p in paths],
        }

    async def stage_seconds(self) -> dict[str, float]:
        return {}


@pytest.fixture(autouse=True)
def _bulk_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_BULK_INGEST_DIR', str(tmp_path / 'bulk'))
    monkeypatch.setattr(source_runs, 'url_sender', lambda *_a, **_k: _Sender())
    monkeypatch.setattr(runner, '_tasks', {})


def _webhook_source() -> IngestSource:
    src = IngestSource(
        source_id='src_orphans', name='t', backend='s3', bucket='b', webhook_enabled=True
    )
    save_source(src)
    source_queue(src.source_id).enqueue([QueueEntry('k.jpg', 'e')])
    return src


def test_run_tick_starts_the_run_the_way_production_calls_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _webhook_source()
    monkeypatch.setattr('src.services.projects.bootstrap.for_each_project', lambda: iter(['p']))
    failures: list[Any] = []
    monkeypatch.setattr(
        dispatcher.logger, 'warning', lambda event, **kw: failures.append((event, kw))
    )

    async def scenario() -> None:
        await dispatcher.run_tick()
        await asyncio.gather(*runner._tasks.values())

    asyncio.run(scenario())
    assert failures == []
    runs = list_stores()
    assert len(runs) == 1
    assert runs[0].job.read()['status'] in {'completed', 'completed_with_errors'}


def test_a_spawn_that_cannot_start_leaves_a_failed_run_not_a_live_orphan() -> None:
    src = _webhook_source()
    errors: list[BaseException] = []

    def from_a_thread() -> None:  # no running loop in this thread: spawn cannot start
        try:
            source_runs.start_source_run(
                src, base_url='http://x', api_prefix='', project='p', scan=False
            )
        except RuntimeError as exc:
            errors.append(exc)

    worker = threading.Thread(target=contextvars.copy_context().run, args=(from_a_thread,))
    worker.start()
    worker.join()
    assert errors
    assert active_run() is None
    (store,) = list_stores()
    assert store.job.read()['status'] == 'failed'
    # The next claim is not refused as busy.
    runner.claim_run(BulkSource('queue', src.source_id, (), False), runner.BulkOptions('t', 4, 1))


def test_cancel_finalizes_an_orphan_with_no_task_even_after_a_restart() -> None:
    spec = BulkSource('folder', '/x')
    store, _ = runner.claim_run(spec, runner.BulkOptions('t', 4, 1))
    store.job.update(updated_at='2020-01-01T00:00:00+00:00')  # claimed long ago, never spawned
    assert active_run() is not None  # its heartbeat is fresh, so it blocks a start
    with pytest.raises(runner.BulkBusyError):
        runner.claim_run(spec, runner.BulkOptions('t', 4, 1))

    runner.cancel_run(store)

    state = store.job.read()
    assert state['status'] == 'cancelled'
    assert state['finished_at']
    assert active_run() is None


def test_cancel_still_signals_a_run_that_has_a_task_here() -> None:
    spec = BulkSource('folder', '/x')
    store, _ = runner.claim_run(spec, runner.BulkOptions('t', 4, 1))

    async def scenario() -> None:
        runner._tasks[store.run_id] = asyncio.create_task(asyncio.sleep(0))
        runner.cancel_run(store)

    asyncio.run(scenario())
    assert store.job.cancel_requested()
    assert store.job.read()['status'] == 'queued'  # its own task finalizes it
