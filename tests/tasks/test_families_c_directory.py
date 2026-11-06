"""``directory_ingest`` through the job path: queue, claim, ingest the directory, finish with a perf summary."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fixtures.job_rows import sql

from src.db.session import async_session_factory
from src.services.jobs.service import start_job
from src.services.visual_search.directory_ingest import DirectoryIngestOptions


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


class Search:
    batches: list[list[str]] = []

    def __init__(self, _client: Any) -> None:
        pass

    async def ingest_batch(self, *, images_data: list[tuple[bytes, str, str]], **_: Any) -> Any:
        Search.batches.append([image_id for _, image_id, _ in images_data])
        return {'processed': len(images_data), 'indexed': {'global': len(images_data)}}


class Client:
    closed = False

    async def close(self) -> None:
        Client.closed = True


@pytest.fixture
def worker_app(queue_env):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


def test_the_registered_task_ingests_the_directory_and_stores_a_valid_summary(
    pg, job_env, producer, wakes, kinds, worker_app, queue_env, tmp_path: Path, monkeypatch
) -> None:
    for i in range(3):
        (tmp_path / f'{i}.jpg').write_bytes(b'x')
    Search.batches = []
    monkeypatch.setattr('src.clients.opensearch.client.OpenSearchClient', Client)
    monkeypatch.setattr('src.services.visual_search.VisualSearchService', Search)

    async def start() -> Any:
        async with async_session_factory()() as session:
            return await start_job(
                session,
                'directory_ingest',
                None,
                params=DirectoryIngestOptions(str(tmp_path), batch_size=2).to_params(),
            )

    job_id = asyncio.run(start()).job.id

    async def run() -> None:
        outcome = await asyncio.to_thread(
            worker_app.tasks['op.families_c.directory_ingest'].apply,
            args=[str(job_id)],
            task_id=str(job_id),
        )
        assert outcome.successful(), outcome.traceback

    asyncio.run(run())

    [(status, result, checkpoint)] = sql(
        pg, 'SELECT status, result, checkpoint FROM jobs WHERE id = %s', job_id
    )
    assert status == 'completed'
    assert (result['status'], result['processed'], result['total_files']) == ('success', 3, 3)
    assert result['perf_summary']['images'] == 3
    assert [len(b) for b in Search.batches] == [2, 1]
    assert checkpoint == {'batch': 2}
    assert Client.closed, 'the wrapper closes the client the handler asked for'
