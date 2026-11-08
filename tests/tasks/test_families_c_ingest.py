"""``bulk_ingest_run`` and ``source_poll`` through the job path: queue, claim, handler, rows.

The ingest route is a fake sender (its own tests are the runner's); the queue rows, the job
rows, the scan and the wrapper are production code. Celery's ``Task.apply`` runs the registered
task with the real wrapper (https://docs.celeryq.dev/en/stable/userguide/testing.html).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from curation.ingest_db.app_fixtures import record_for
from fixtures.job_rows import make_project, project_session, sql

from src.config.project_context import bind_project
from src.db.repos import ingest_sources as repo
from src.db.session import async_session_factory
from src.services.curation.bulk_ingest import source_scan
from src.services.curation.bulk_ingest.object_store import ObjectFetchError
from src.services.curation.bulk_ingest.source_runs import start_poll, start_source_run
from src.services.curation.bulk_ingest.sources import IngestSource, WatchTrigger


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


class Sender:
    """Answers each batch as the ingest route would; ``bad*`` files fail for good."""

    def __init__(self) -> None:
        self.closed = False
        self.paths: list[str] = []

    async def post_batch(self, paths: list[str], *, source: str, run_id: str) -> dict[str, Any]:  # noqa: ARG002 - the fake ignores the run id
        self.paths += paths
        assert source == 'tag'
        results = []
        for path in paths:
            name = path.rsplit('/', 1)[-1]
            if name.startswith('bad'):
                results.append(
                    {
                        'status': 'failed',
                        'image_path': path,
                        'error': 'cannot decode',
                        'error_kind': 'decode_failed',
                    }
                )
            else:
                results.append({'status': 'success', 'image_path': path, 'image_id': name})
        done = len(results)
        summary = {'crops_indexed': done, 'n_embedded': done, 'n_not_embedded': 0}
        return {'summary': summary | {'n_embed_failed': 0}, 'results': results}

    async def stage_seconds(self) -> dict[str, float]:
        return {}

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def sender(monkeypatch: pytest.MonkeyPatch) -> Sender:
    fake = Sender()
    monkeypatch.setattr(
        'src.services.curation.bulk_ingest.sender.url_sender', lambda *_a, **_k: fake
    )
    return fake


@pytest.fixture
def worker_app(queue_env):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


@pytest.fixture
def world(pg, job_env, producer, wakes, kinds, worker_app, monkeypatch):
    """A project the handlers bind (the registry lookup is the one thing replaced)."""
    pid = make_project(pg)
    slug = sql(pg, 'SELECT slug FROM projects WHERE id = %s', pid)[0][0]
    record = record_for(pid, slug)

    async def bound(_ctx: Any) -> Any:
        return bind_project(record)

    monkeypatch.setattr('src.tasks.jobs_c.bound_project', bound)
    return pid, record


def make_source(pg, pid: uuid.UUID, root: Path, **over: Any) -> IngestSource:
    source = IngestSource(
        name=f's{uuid.uuid4().hex[:8]}',
        backend='folder',
        path=str(root),
        extensions=['.jpg'],
        source_tag='tag',
        batch_size=2,
        concurrency=1,
        one_shot=True,
        **over,
    )
    with project_session(pg.engine('app'), pid) as session:
        row = repo.insert(
            session,
            pid,
            name=source.name,
            kind='folder',
            config=source.config(),
            schedule=source.triggers(),
            enabled=source.triggered,
        )
    source.source_id = str(row.id)
    return source


def start(
    pid: uuid.UUID, source: IngestSource, *, scan: bool = True, poll: bool = False
) -> uuid.UUID:
    async def go() -> uuid.UUID | bool:
        async with async_session_factory()() as session:
            if poll:
                return await start_poll(session, pid, source)
            return await start_source_run(session, pid, source, scan=scan)

    result = asyncio.run(go())
    assert result is not False
    if poll:
        (job,) = sql(
            pg_of(),
            "SELECT id FROM jobs WHERE kind = 'source_poll' ORDER BY created_at DESC LIMIT 1",
        )[0]
        return job  # type: ignore[no-any-return]
    return result  # type: ignore[return-value]


_PG: list[Any] = []


def pg_of() -> Any:
    return _PG[0]


@pytest.fixture(autouse=True)
def _remember_pg(pg) -> None:
    _PG[:] = [pg]


def apply(worker_app, task: str, job_id: uuid.UUID) -> None:
    async def run() -> None:
        outcome = await asyncio.to_thread(
            worker_app.tasks[f'op.families_c.{task}'].apply, args=[str(job_id)], task_id=str(job_id)
        )
        assert outcome.successful(), outcome.traceback

    asyncio.run(run())


def images(root: Path, names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b'x')


def files(pg, source: IngestSource) -> dict[str, str]:
    rows = sql(
        pg,
        'SELECT file_key, status FROM ingest_source_files WHERE source_id = %s',
        uuid.UUID(source.source_id),
    )
    return {key.partition('\x1f')[0].rsplit('/', 1)[-1]: status for key, status in rows}


def test_a_run_scans_the_folder_drains_the_queue_and_reports_the_failures(
    pg, world, sender, worker_app, tmp_path
) -> None:
    pid, _ = world
    images(tmp_path / 'in', ['a.jpg', 'b.jpg', 'c.jpg', 'bad.jpg'])
    source = make_source(pg, pid, tmp_path / 'in')
    job = start(pid, source)

    apply(worker_app, 'bulk_ingest_run', job)

    [(status, result)] = sql(pg, 'SELECT status, result FROM jobs WHERE id = %s', job)
    assert (status, result.get('failed'), result.get('ingested')) == (
        'completed_with_errors',
        1,
        3,
    ), result
    assert (result['ingested'], result['failed']) == (3, 1)
    assert result['perf_summary']['images'] == 3
    assert files(pg, source) == {
        'a.jpg': 'done',
        'b.jpg': 'done',
        'c.jpg': 'done',
        'bad.jpg': 'failed',
    }
    with project_session(pg.engine('app'), pid) as session:
        counts = repo.counts(session, uuid.UUID(source.source_id))
    assert (counts['done'], counts['dead']) == (3, 1)  # a decode failure is never retried
    assert sender.closed
    assert len(sender.paths) == 4


def test_a_rerun_ingests_only_the_files_that_are_new_and_retries_the_failed_one(
    pg, world, sender, worker_app, tmp_path
) -> None:
    pid, _ = world
    images(tmp_path / 'in', ['a.jpg', 'b.jpg'])
    source = make_source(pg, pid, tmp_path / 'in')
    first = start(pid, source)
    apply(worker_app, 'bulk_ingest_run', first)
    sender.paths.clear()

    images(tmp_path / 'in', ['c.jpg'])
    second = start(pid, source)
    apply(worker_app, 'bulk_ingest_run', second)

    assert [p.rsplit('/', 1)[-1] for p in sender.paths] == ['c.jpg']
    [(status,)] = sql(pg, 'SELECT status FROM jobs WHERE id = %s', second)
    assert status == 'completed'
    assert set(files(pg, source).values()) == {'done'}


def test_the_claims_of_a_run_that_died_are_drained_by_the_next_one(
    pg, world, sender, worker_app, tmp_path
) -> None:
    pid, _ = world
    images(tmp_path / 'in', ['a.jpg', 'b.jpg'])
    source = make_source(pg, pid, tmp_path / 'in')
    dead = start(pid, source, scan=False)
    sql(
        pg,
        "UPDATE jobs SET status = 'failed', finished_at = now(), error_code = 'x', error_message = 'x' WHERE id = %s",
        dead,
    )
    queued = sorted(str(tmp_path / 'in' / n) for n in ('a.jpg', 'b.jpg'))
    with project_session(pg.engine('app'), pid) as session:
        repo.enqueue(session, uuid.UUID(source.source_id), queued)
        repo.claim(session, uuid.UUID(source.source_id), dead, 10)
    assert set(files(pg, source).values()) == {'claimed'}

    job = start(pid, source, scan=False)
    apply(worker_app, 'bulk_ingest_run', job)

    assert set(files(pg, source).values()) == {'done'}
    assert sorted(sender.paths) == queued


def test_a_poll_that_finds_files_starts_a_run_and_a_quiet_one_starts_none(
    pg, world, worker_app, tmp_path
) -> None:
    pid, _ = world
    images(tmp_path / 'busy', ['a.jpg'])
    (tmp_path / 'quiet').mkdir()
    busy = make_source(pg, pid, tmp_path / 'busy', watch=WatchTrigger(True, 10))
    quiet = make_source(pg, pid, tmp_path / 'quiet', watch=WatchTrigger(True, 10))
    # a folder watcher only queues settled files
    import os
    import time

    old = time.time() - 600
    os.utime(tmp_path / 'busy' / 'a.jpg', (old, old))

    apply(worker_app, 'source_poll', start(pid, quiet, poll=True))
    assert sql(
        pg, "SELECT count(*) FROM jobs WHERE kind = 'bulk_ingest_run' AND project_id = %s", pid
    ) == [(0,)]

    job = start(pid, busy, poll=True)
    apply(worker_app, 'source_poll', job)
    [(result,)] = sql(pg, 'SELECT result FROM jobs WHERE id = %s', job)
    assert (result['pending'], result['run_started']) == (1, True)
    runs = sql(
        pg, "SELECT params FROM jobs WHERE kind = 'bulk_ingest_run' AND project_id = %s", pid
    )
    assert [r[0]['source_id'] for r in runs] == [busy.source_id]
    assert [r[0]['scan'] for r in runs] == [False]  # the poll already scanned


def test_a_failed_poll_records_the_error_and_backs_off_and_a_good_one_clears_it(
    pg, world, worker_app, tmp_path, monkeypatch
) -> None:
    pid, _ = world
    (tmp_path / 'in').mkdir()
    source = make_source(pg, pid, tmp_path / 'in', watch=WatchTrigger(True, 10))

    def failing(*_a: Any, **_k: Any) -> None:
        raise ObjectFetchError('listing unavailable')

    scan_source = source_scan.scan_source
    monkeypatch.setattr(source_scan, 'scan_source', failing)
    for expected in (1, 2):
        job = start(pid, source, poll=True)
        apply(worker_app, 'source_poll', job)
        [(result,)] = sql(pg, 'SELECT result FROM jobs WHERE id = %s', job)
        assert result == {'failed': True, 'failures': expected}
        [(state,)] = sql(
            pg, 'SELECT scan_state FROM ingest_sources WHERE id = %s', uuid.UUID(source.source_id)
        )
        assert 'listing unavailable' in state['last_poll_error']
        assert state['failures'] == expected
        assert state['backoff_until'] > 0

    monkeypatch.setattr(source_scan, 'scan_source', scan_source)
    apply(worker_app, 'source_poll', start(pid, source, poll=True))
    [(state,)] = sql(
        pg, 'SELECT scan_state FROM ingest_sources WHERE id = %s', uuid.UUID(source.source_id)
    )
    assert (state['failures'], state['last_poll_error'], state['backoff_until']) == (0, None, None)


def test_a_poll_of_a_removed_source_ends_without_error(pg, world, worker_app, tmp_path) -> None:
    pid, _ = world
    (tmp_path / 'in').mkdir()
    source = make_source(pg, pid, tmp_path / 'in')
    job = start(pid, source, poll=True)
    sql(pg, 'DELETE FROM ingest_sources WHERE id = %s', uuid.UUID(source.source_id))

    apply(worker_app, 'source_poll', job)

    assert sql(pg, 'SELECT status, result FROM jobs WHERE id = %s', job) == [
        ('completed', {'source': 'removed'})
    ]
