"""The reprocess job on the queue: start, busy, broker down, the task, cancel and resume (4A-1)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fixtures.job_rows import make_project, sql

from curation.jobs_a.ctx import FakeCtx
from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.services.curation import reprocess_job
from src.services.curation.reprocess_models import ReprocessScopeResult
from src.services.jobs.service import cancel_job


pytestmark = pytest.mark.db

INDEXES = SimpleNamespace(items_index='p__items', images_index='p__images')


@pytest.fixture(autouse=True)
def _bound_indexes(monkeypatch):
    monkeypatch.setattr(reprocess_job, 'get_curation_config', lambda: INDEXES)


async def start(project, ids, **kw):
    async with async_session_factory()() as session:
        return await reprocess_job.start(
            session,
            project,
            request={'k': 1},
            scopes=['detect'],
            image_ids=ids,
            **kw,
        )


@pytest.mark.asyncio
async def test_start_queues_one_row_with_its_targets_and_sends_one_message(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    job = await start(project, [f'img{i}' for i in range(7)])
    assert job.status == 'queued'
    assert producer.sent() == [str(job.id)]
    assert sql(pg, 'SELECT count(*) FROM job_items WHERE job_id = %s', job.id) == [(7,)]
    [(first, last)] = sql(
        pg,
        'SELECT (array_agg(item_id ORDER BY position))[1], '
        '(array_agg(item_id ORDER BY position))[7] FROM job_items WHERE job_id = %s',
        job.id,
    )
    assert (first, last) == ('img0', 'img6')


@pytest.mark.asyncio
async def test_a_second_start_while_one_is_active_is_409_job_busy(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    first = await start(project, ['a'])
    with pytest.raises(RequestRejectedError) as busy:
        await start(project, ['b'])
    assert busy.value.status_code == 409
    assert busy.value.detail['error'] == 'job_busy'
    assert busy.value.detail['active_job_id'] == str(first.id)
    assert sql(
        pg, "SELECT count(*) FROM jobs WHERE kind = 'reprocess' AND project_id = %s", project
    ) == [(1,)]


@pytest.mark.asyncio
async def test_broker_down_is_503_and_fails_the_row(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    producer.fail_send = True
    with pytest.raises(RequestRejectedError) as down:
        await start(project, ['a'])
    assert down.value.status_code == 503
    assert down.value.detail['error'] == 'queue_unavailable'
    assert sql(
        pg, "SELECT status FROM jobs WHERE kind = 'reprocess' AND project_id = %s", project
    ) == [('failed',)]


@pytest.mark.asyncio
async def test_cancel_while_queued_is_cancelled(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    job = await start(project, ['a'])
    async with async_session_factory()() as session:
        moved = await cancel_job(session, job.id, project, requested_by='test')
    assert moved.job.status == 'cancelled'


class Recorder:
    """Stands in for ``process_images``: records the slices, optionally fails some images."""

    def __init__(self, ctx=None, cancel_on_call=None, fail=()):
        self.slices: list[list[str]] = []
        self.ctx = ctx
        self.cancel_on_call = cancel_on_call
        self.fail = set(fail)

    async def __call__(self, _opensearch, _service, *, scopes, image_ids, **_kw):
        self.slices.append(list(image_ids))
        results = []
        for scope in scopes:
            res = ReprocessScopeResult(scope=scope, selected=len(image_ids))
            res.queued = len([i for i in image_ids if i not in self.fail])
            res.failed = len([i for i in image_ids if i in self.fail])
            res.add_count('merged', 1)
            results.append(res)
        if self.cancel_on_call == len(self.slices):
            self.ctx.cancel_now()
            return results, True
        return results, False


def ctx_for(job, project, **kw):
    return FakeCtx(project, job.id, job.params, **kw)


@pytest.mark.asyncio
async def test_the_run_goes_slice_by_slice_and_aggregates(
    pg, job_env, producer, wakes, monkeypatch
):
    project = make_project(pg)
    ids = [f'img{i:04d}' for i in range(600)]
    job = await start(project, ids)
    rec = Recorder(fail={'img0003'})
    monkeypatch.setattr(reprocess_job, 'process_images', rec)
    ctx = ctx_for(job, project)
    result, with_errors = await reprocess_job.run_reprocess(ctx, object(), object())
    assert [len(s) for s in rec.slices] == [256, 256, 88]
    assert [i for s in rec.slices for i in s] == ids  # in order, once each
    [detect] = result['results']
    assert (detect['selected'], detect['queued'], detect['failed']) == (600, 599, 1)
    assert detect['detail'] == {'merged': 3}
    assert with_errors is True
    assert ctx.progresses[-1][:3] == (600, 600, 'processing')
    assert result['perf_summary']['images'] == 600


@pytest.mark.asyncio
async def test_an_interrupted_job_resumes_from_its_checkpoint_without_repeats(
    pg, job_env, producer, wakes, monkeypatch
):
    project = make_project(pg)
    ids = [f'img{i:04d}' for i in range(600)]
    job = await start(project, ids)
    seen: list[str] = []

    first = Recorder()
    monkeypatch.setattr(reprocess_job, 'process_images', first)
    ctx = ctx_for(job, project, cancel_after_checkpoints=1)  # stops after the first checkpoint
    await reprocess_job.run_reprocess(ctx, object(), object())
    seen += [i for s in first.slices for i in s]
    assert ctx.checkpoints[-1]['done'] == 256

    second = Recorder()
    monkeypatch.setattr(reprocess_job, 'process_images', second)
    resumed = ctx_for(job, project, resume_from=ctx.checkpoints[-1])
    result, with_errors = await reprocess_job.run_reprocess(resumed, object(), object())
    seen += [i for s in second.slices for i in s]
    assert seen == ids  # the item count at the end equals the total, no repeats
    assert result['results'][0]['selected'] == 600
    assert result['results'][0]['queued'] == 600
    assert with_errors is False


@pytest.mark.asyncio
async def test_a_cancel_inside_a_slice_stops_without_counting_the_slice(
    pg, job_env, producer, wakes, monkeypatch
):
    project = make_project(pg)
    job = await start(project, [f'img{i}' for i in range(300)])
    ctx = ctx_for(job, project)
    monkeypatch.setattr(reprocess_job, 'process_images', Recorder(ctx, cancel_on_call=1))
    result, _ = await reprocess_job.run_reprocess(ctx, object(), object())
    assert result['cancelled'] is True
    assert ctx.checkpoints == []  # a partial slice is never a resume point


@pytest.mark.asyncio
async def test_a_job_planned_for_other_indexes_refuses_to_run(
    pg, job_env, producer, wakes, monkeypatch
):
    project = make_project(pg)
    job = await start(project, ['a'])
    monkeypatch.setattr(
        reprocess_job,
        'get_curation_config',
        lambda: SimpleNamespace(items_index='x', images_index='y'),
    )
    with pytest.raises(RuntimeError, match='other indexes'):
        await reprocess_job.run_reprocess(ctx_for(job, project), object(), object())


@pytest.mark.asyncio
async def test_the_registered_task_runs_the_coroutine_once_through_the_wrapper(
    pg, job_env, producer, wakes, queue_env, worker_app, monkeypatch
):
    from contextlib import asynccontextmanager

    from src.tasks import jobs_a_support

    project = make_project(pg, 'rp-task')
    job = await start(project, ['a', 'b', 'c'])
    rec = Recorder()
    monkeypatch.setattr(reprocess_job, 'process_images', rec)

    @asynccontextmanager
    async def no_bind(_ctx):
        yield

    class Ingest:
        async def service(self, _os):
            return object()

    async def fake_client():
        return SimpleNamespace(close=lambda: None)

    async def fake_ingest():
        return Ingest()

    monkeypatch.setattr(jobs_a_support, 'bound_project', no_bind)
    monkeypatch.setattr(jobs_a_support, 'opensearch_client', fake_client)
    monkeypatch.setattr(jobs_a_support, 'worker_ingest', fake_ingest)

    outcome = await asyncio.to_thread(
        worker_app.tasks['op.jobs_a.reprocess'].apply, args=[str(job.id)], task_id=str(job.id)
    )
    assert outcome.successful(), outcome.traceback
    [(status, result, done)] = sql(
        pg, "SELECT status, result, (progress->>'done')::int FROM jobs WHERE id = %s", job.id
    )
    assert (status, done) == ('completed', 3)
    assert result['results'][0]['queued'] == 3
    assert rec.slices == [['a', 'b', 'c']]
