"""``region_fp_centroids``: the start queues one row on the jobs queue (409 while active) and the
handler builds the centroids, then pulls tight matches only when asked."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
from fixtures.job_rows import make_project, sql

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.clustering import region_fp_job
from src.services.jobs.service import start_job


KIND = 'region_fp_centroids'


async def _start(project: Any) -> Any:
    async with async_session_factory()() as session:
        return await start_job(
            session, KIND, project, params={'auto_fp_threshold': 0.0}, requested_by='test'
        )


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_start_queues_one_row_on_the_jobs_queue(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    result = await _start(project)
    [(status, queue, kind)] = sql(
        pg, 'SELECT status, queue, kind FROM jobs WHERE id = %s', result.job.id
    )
    assert (status, queue, kind) == ('queued', 'jobs', KIND)


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_second_start_while_one_is_active_is_job_busy(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    await _start(project)
    with pytest.raises(RequestRejectedError) as busy:
        await _start(project)
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')


class _Ctx:
    project_slug = 'p'

    def __init__(self, threshold: float) -> None:
        self.params = {'auto_fp_threshold': threshold}

    async def client(self, _name: str) -> str:
        return 'opensearch'

    async def progress(self, *_a: Any) -> None:
        return None

    def cancelled(self) -> bool:
        return False


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    async def build(client: Any) -> dict[str, Any]:
        seen.append('build')
        return {'status': 'success'}

    async def auto(client: Any, *, threshold: float) -> dict[str, Any]:
        seen.append(f'auto:{threshold}')
        return {'status': 'success', 'n_boxes_moved': 3}

    async def resolve(_slug: str) -> object:
        return object()

    monkeypatch.setattr(region_fp_job, 'build_region_fp_centroids', build)
    monkeypatch.setattr(region_fp_job, 'auto_assign_fp_from_centroids', auto)
    monkeypatch.setattr(region_fp_job, 'resolve_project', resolve)
    monkeypatch.setattr(region_fp_job, 'bind_project', lambda _record: contextlib.nullcontext())
    return seen


@pytest.mark.asyncio
async def test_threshold_zero_builds_only(calls: list[str]) -> None:
    result = await region_fp_job.region_fp_centroids(_Ctx(0.0))  # type: ignore[arg-type]
    assert calls == ['build']
    assert 'auto_fp' not in result


@pytest.mark.asyncio
async def test_a_threshold_pulls_matches_after_the_build(calls: list[str]) -> None:
    result = await region_fp_job.region_fp_centroids(_Ctx(0.2))  # type: ignore[arg-type]
    assert calls == ['build', 'auto:0.2']
    assert result['auto_fp']['n_boxes_moved'] == 3
