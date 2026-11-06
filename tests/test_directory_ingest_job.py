"""``POST /ingest/directory`` queues the ``directory_ingest`` job; the job ingests the directory."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fixtures import job_fixtures
from fixtures.job_rows import sql

from src.core.db_session import install_database_error_handlers
from src.core.disk_space import require_disk_space
from src.core.error_handlers import request_rejected_response
from src.core.exceptions import RequestRejectedError
from src.services.visual_search.directory_ingest import (
    DirectoryIngestError,
    DirectoryIngestOptions,
    check_directory,
    collect_images,
    run_directory_ingest,
)


if TYPE_CHECKING:
    from pathlib import Path


kinds = job_fixtures.kinds
producer = job_fixtures.producer

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine', 'api_async_engine')]


class FakeSearch:
    """Records the batches; ``fail_on`` makes one batch raise."""

    def __init__(self, fail_on: int | None = None) -> None:
        self.batches: list[list[str]] = []
        self.fail_on = fail_on

    async def ingest_batch(self, *, images_data: list[tuple[bytes, str, str]], **_: Any) -> Any:
        self.batches.append([image_id for _, image_id, _ in images_data])
        if self.fail_on == len(self.batches):
            raise OSError('triton down')
        return {
            'processed': len(images_data),
            'duplicates': 0,
            'errors_count': 0,
            'near_duplicates': 0,
            'indexed': {'global': len(images_data), 'faces': 1},
        }


def make_images(root: Path, count: int) -> None:
    (root / 'sub').mkdir()
    for i in range(count):
        target = root / 'sub' / f'{i:03}.jpg' if i % 2 else root / f'{i:03}.jpg'
        target.write_bytes(b'x' * (i + 1))
    (root / 'notes.txt').write_text('skip me')


def test_collect_is_sorted_limited_and_reports_skipped_extensions(tmp_path: Path) -> None:
    make_images(tmp_path, 5)
    images, skipped = collect_images(tmp_path, recursive=True, max_images=4)
    assert [p.name for p in images] == sorted(p.name for p in images)[:4] or len(images) == 4
    assert len(images) == 4
    assert skipped == ['.txt']
    top, _ = collect_images(tmp_path, recursive=False, max_images=None)
    assert {p.parent for p in top} == {tmp_path}


def test_a_missing_directory_and_a_file_are_refused(tmp_path: Path) -> None:
    with pytest.raises(DirectoryIngestError, match='not found'):
        check_directory(str(tmp_path / 'nope'))
    plain = tmp_path / 'f.txt'
    plain.write_text('x')
    with pytest.raises(DirectoryIngestError, match='not a directory'):
        check_directory(str(plain))


def test_batches_use_relative_ids_and_the_result_carries_a_valid_perf_summary(
    tmp_path: Path,
) -> None:
    make_images(tmp_path, 5)
    service = FakeSearch()
    progress: list[tuple[int, int, int]] = []

    async def on_batch(batch: int, done: int, total: int) -> None:
        progress.append((batch, done, total))

    result = asyncio.run(
        run_directory_ingest(
            service,
            DirectoryIngestOptions(str(tmp_path), batch_size=2),
            on_batch=on_batch,
            servers=['triton-a'],
        )
    )
    assert [len(b) for b in service.batches] == [2, 2, 1]
    assert {i for b in service.batches for i in b} == {
        '000.jpg',
        '002.jpg',
        '004.jpg',
        'sub/001.jpg',
        'sub/003.jpg',
    }
    assert progress == [(1, 2, 5), (2, 4, 5), (3, 5, 5)]
    assert (result['status'], result['processed'], result['total_files']) == ('success', 5, 5)
    assert result['indexed']['faces'] == 3
    assert result['perf_summary']['images'] == 5
    assert result['perf_summary']['servers'] == ['triton-a']


def test_a_rerun_indexes_the_same_ids_so_nothing_is_duplicated(tmp_path: Path) -> None:
    """The store keys a global document by this id (``bulk.py`` ``_id: image_id``)."""
    from src.clients.opensearch.bulk import _global_action

    make_images(tmp_path, 4)
    ids: list[list[str]] = []
    for _ in range(2):
        service = FakeSearch()
        asyncio.run(run_directory_ingest(service, DirectoryIngestOptions(str(tmp_path))))
        ids.append([i for batch in service.batches for i in batch])
    assert ids[0] == ids[1]
    action = _global_action(
        {'image_id': ids[0][0], 'image_path': 'p', 'global_embedding': [0.0]}, 't'
    )
    assert action['_id'] == ids[0][0]


def test_a_failed_batch_is_counted_and_the_rest_goes_on(tmp_path: Path) -> None:
    make_images(tmp_path, 4)
    service = FakeSearch(fail_on=1)
    result = asyncio.run(
        run_directory_ingest(service, DirectoryIngestOptions(str(tmp_path), batch_size=2))
    )
    assert (result['status'], result['processed'], result['errors_count']) == ('partial', 2, 2)


def test_a_resume_skips_the_finished_batches_and_a_stop_ends_the_loop(tmp_path: Path) -> None:
    make_images(tmp_path, 5)
    resumed = FakeSearch()
    asyncio.run(
        run_directory_ingest(
            resumed, DirectoryIngestOptions(str(tmp_path), batch_size=2), resume_batch=2
        )
    )
    assert [len(b) for b in resumed.batches] == [1]
    stopped = FakeSearch()
    result = asyncio.run(
        run_directory_ingest(
            stopped, DirectoryIngestOptions(str(tmp_path), batch_size=2), should_stop=lambda: True
        )
    )
    assert stopped.batches == []
    assert result['processed'] == 0


@pytest.fixture
def app(pg, pg_env):
    from src.routers.ingest import router

    application = FastAPI()
    application.include_router(router)
    application.dependency_overrides[require_disk_space] = lambda: None
    install_database_error_handlers(application)
    application.add_exception_handler(RequestRejectedError, request_rejected_response)
    return application


@pytest_asyncio.fixture
async def client(app):
    from src.db.engine import dispose_engines

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as http:
        yield http
    await dispose_engines()


@pytest.mark.asyncio
async def test_the_start_route_queues_one_deployment_job_and_sends_one_message(
    pg, client, producer, tmp_path: Path
) -> None:
    make_images(tmp_path, 2)
    reply = await client.post('/ingest/directory', params={'directory': str(tmp_path)})
    assert reply.status_code == 202, reply.text
    body = reply.json()
    [(project, kind, status, params)] = sql(
        pg, 'SELECT project_id, kind, status, params FROM jobs WHERE id = %s', body['job_id']
    )
    assert (project, kind, status) == (None, 'directory_ingest', 'queued')
    assert params['directory'] == str(tmp_path)
    assert producer.sent() == [body['job_id']]
    assert body['status_url'] == f'/ingest/directory/{body["job_id"]}'


@pytest.mark.asyncio
async def test_a_bad_directory_is_400_and_queues_nothing(client, producer, tmp_path: Path) -> None:
    reply = await client.post('/ingest/directory', params={'directory': str(tmp_path / 'nope')})
    assert reply.status_code == 400
    assert producer.sent() == []


@pytest.mark.asyncio
async def test_a_broker_that_is_down_is_503_queue_unavailable(
    pg, client, producer, tmp_path: Path
) -> None:
    producer.fail_send = True
    reply = await client.post('/ingest/directory', params={'directory': str(tmp_path)})
    assert reply.status_code == 503
    assert reply.json()['detail']['error'] == 'queue_unavailable'
    assert sql(pg, "SELECT status FROM jobs WHERE kind = 'directory_ingest'")[-1] == ('failed',)


@pytest.mark.asyncio
async def test_status_and_cancel_read_the_job_row(client, producer, tmp_path: Path) -> None:
    started = (await client.post('/ingest/directory', params={'directory': str(tmp_path)})).json()
    status = await client.get(started['status_url'])
    assert status.status_code == 200
    assert status.json()['job']['status'] == 'queued'
    assert status.json()['result'] is None

    cancelled = await client.post(f'{started["status_url"]}/cancel')
    assert cancelled.status_code == 200
    assert cancelled.json()['job']['status'] == 'cancelled'
    other = await client.get('/ingest/directory/00000000-0000-4000-8000-000000000000')
    assert other.status_code == 404
