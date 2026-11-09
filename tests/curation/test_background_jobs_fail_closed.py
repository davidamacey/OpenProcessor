"""A background job that crashed is logged at shutdown, and one project's failed ingest
dispatch does not stop the other projects' ticks."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from structlog.testing import capture_logs

from src.services.curation.background_jobs import stop_background_jobs
from src.services.curation.bulk_ingest import dispatcher


if TYPE_CHECKING:
    import pytest


def test_a_crashed_background_job_is_logged_at_shutdown() -> None:
    async def crash() -> None:
        raise RuntimeError('sweeper blew up')

    async def go() -> list[dict]:
        task = asyncio.create_task(crash())
        await asyncio.sleep(0)
        with capture_logs() as logs:
            await stop_background_jobs([task])
        return logs

    logs = asyncio.run(go())
    crashed = [entry for entry in logs if entry['event'] == 'background_job_crashed']
    assert len(crashed) == 1
    assert 'sweeper blew up' in crashed[0]['error']


def test_a_cancelled_background_job_is_not_a_crash() -> None:
    async def forever() -> None:
        await asyncio.sleep(3600)

    async def go() -> list[dict]:
        task = asyncio.create_task(forever())
        await asyncio.sleep(0)
        with capture_logs() as logs:
            await stop_background_jobs([task])
        return logs

    assert asyncio.run(go()) == []


def test_one_projects_failed_dispatch_does_not_stop_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.exc import OperationalError

    seen: list[str] = []
    monkeypatch.setattr(
        'src.services.projects.bootstrap.for_each_project', lambda: iter(['alpha', 'beta'])
    )

    class Session:
        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *_a: object) -> None:
            return None

    async def dispatch(_session: object, _project: object) -> list[object]:
        seen.append(f'call{len(seen)}')
        if len(seen) == 1:
            raise OperationalError('select', {}, Exception('db down'))
        return []

    monkeypatch.setattr(dispatcher, 'async_session_factory', lambda: Session)
    monkeypatch.setattr(dispatcher, 'dispatch_once', dispatch)
    monkeypatch.setattr(dispatcher, 'current_project_id', lambda: 'p')

    with capture_logs() as logs:
        asyncio.run(dispatcher.run_tick())

    assert seen == ['call0', 'call1']
    failed = [e for e in logs if e['event'] == 'ingest_dispatch_failed']
    assert [e['project'] for e in failed] == ['alpha']


def test_background_jobs_start_only_when_enabled_and_stop_cancels_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.curation.background_jobs import start_background_jobs

    monkeypatch.setenv('OP_INGEST_DISPATCH_S', '0')

    async def lifecycle() -> tuple[int, int, bool]:
        none_started = len(start_background_jobs())
        monkeypatch.setenv('OP_INGEST_DISPATCH_S', '60')
        jobs = start_background_jobs()
        await stop_background_jobs(jobs)
        return none_started, len(jobs), jobs[0].cancelled()

    assert asyncio.run(lifecycle()) == (0, 1, True)
