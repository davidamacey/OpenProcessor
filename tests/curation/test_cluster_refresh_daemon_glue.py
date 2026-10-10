"""WP-1.5 (#212): the daemon's HTTP glue around the schedule, with the API faked."""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

import httpx

from scripts.curation import cluster_refresh_daemon as d
from src.services.curation.clustering.refresh_schedule import ScheduleBook
from src.services.curation.file_job import HEARTBEAT_STALE_S, FileJob


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_start_conflict_keeps_the_request_pending() -> None:
    async def go() -> str | None:
        async with _client(lambda _r: httpx.Response(409, json={'detail': 'busy'})) as c:
            return await d._start_auto_label_job(c, 'http://api', '/curation', 'alpha')

    assert asyncio.run(go()) is None


def test_pausing_cancels_only_the_paused_projects_own_job() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={'cancelled': True})

    book = ScheduleBook()
    for slug in ('alpha', 'beta', 'gamma'):
        book.get(slug).on_started(1250, f'job-{slug}', previous_trained=1000)
    book.get('gamma').job_id = None  # paused, nothing of ours in flight

    async def go() -> None:
        async with _client(handler) as c:
            await d._release_inactive(
                c,
                api='http://api',
                api_prefix='/curation',
                book=book,
                active={'beta'},
                paused={'alpha', 'gamma'},
            )

    asyncio.run(go())
    assert calls == ['/curation/projects/alpha/pipeline/auto_label/cancel']
    assert book.get('alpha').job_id is None
    assert book.get('alpha').last_trained_count == 1000
    assert book.get('beta').job_id == 'job-beta'


def test_a_deleted_project_is_forgotten_without_an_api_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f'unexpected call {request.url}')

    book = ScheduleBook()
    book.get('gone').on_started(1250, 'job-1', previous_trained=1000)

    async def go() -> None:
        async with _client(handler) as c:
            await d._release_inactive(
                c, api='http://api', api_prefix='/curation', book=book, active=set(), paused=set()
            )

    asyncio.run(go())
    assert book.slugs() == []


def test_a_dead_workers_job_is_repaired_to_interrupted_for_the_retry(tmp_path: Path) -> None:
    """The status route the daemon polls repairs a stale heartbeat, so the
    schedule sees ``interrupted`` and retries after its backoff."""
    job = FileJob(tmp_path)
    job.write({'job_id': 'j', 'status': 'running'})
    job.touch_heartbeat()
    old = time.time() - HEARTBEAT_STALE_S - 5
    os.utime(job.heartbeat_file, (old, old))
    state = job.repair_if_stale(frozenset({'running'}), error_prefix='auto_label')
    assert state['status'] == 'interrupted'
