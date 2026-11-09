"""The assist run on the queue: one row and one message, members and sample in the job, busy,
the wire state of a job row, and the worker's fail-closed check (4A-2)."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

import pytest
from fixtures.job_rows import make_project, sql

from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.schemas.jobs import JobView
from src.services.curation.assist import job as assist_job
from src.services.curation.assist.models import AssistRunRequest, ResolvedScope
from src.services.curation.assist.prepare import prepare_queued


REQUEST = AssistRunRequest.model_validate(
    {
        'target': {'kind': 'cluster', 'cluster_uid': '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'},
        'actions': ['classify'],
        'dry_run': False,
    }
)
RESOLVED = ResolvedScope(
    scope_count=4,
    scope_digest='sha256:x',
    sampled=2,
    sample_seed=7,
    vlm={'name': 'ua', 'revision': 1, 'model': 'm'},
    prompt_pack={'name': 'generic', 'revision': None},
)


def prepared() -> Any:
    scope = SimpleNamespace(sampled_ids=['c1', 'c2'], box_ids=None)
    return SimpleNamespace(request=REQUEST, scope=scope, resolved=lambda: RESOLVED)


async def start(project, members=('c1', 'c2', 'c3', 'c4')):
    async with async_session_factory()() as session:
        return await assist_job.start(
            session, project, prepared(), members=None if members is None else list(members)
        )


def view(status: str, **over: Any) -> JobView:
    now = dt.datetime(2026, 11, 8, tzinfo=dt.UTC)
    fields: dict[str, Any] = {
        'id': '01a1bd81-eaed-7c0b-a6e9-ef655dfa9809',
        'kind': 'assist_run',
        'project': 'p',
        'status': status,
        'scope': '',
        'queue': 'jobs',
        'attempt': 1,
        'retry_count': 0,
        'max_attempts': 3,
        'pause_reason': None,
        'pause_requested': False,
        'cancel_requested': False,
        'progress': {},
        'progress_updated_at': None,
        'result': None,
        'error_code': None,
        'error_message': None,
        'params': {
            'request': REQUEST.model_dump(mode='json'),
            'resolved': RESOLVED.model_dump(mode='json'),
        },
        'requested_by': 'api',
        'created_at': now,
        'queued_at': now,
        'started_at': None,
        'finished_at': None,
        'updated_at': now,
    }
    return JobView(**{**fields, **over})


@pytest.mark.db
@pytest.mark.asyncio
async def test_start_queues_one_row_with_its_members_and_sample(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    job = await start(project)
    assert job.status == 'queued'
    assert producer.sent() == [str(job.id)]
    assert sql(pg, 'SELECT count(*) FROM job_items WHERE job_id = %s', job.id) == [(4,)]
    assert job.params['sampled_ids'] == ['c1', 'c2']
    assert job.params['members_total'] == 4


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_run_with_no_cluster_scope_keeps_no_members(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    job = await start(project, members=None)
    assert job.params['members_total'] is None
    assert sql(pg, 'SELECT count(*) FROM job_items WHERE job_id = %s', job.id) == [(0,)]


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_second_start_while_one_is_active_is_409_job_busy(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    first = await start(project)
    with pytest.raises(RequestRejectedError) as busy:
        await start(project)
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')
    assert busy.value.detail['active_job_id'] == str(first.id)


def test_a_running_job_that_waits_for_the_region_worker_is_waiting_region() -> None:
    running = view('running', progress={'done': 1, 'total': 5, 'stage': 'waiting_region'})
    state = assist_job.state_from_view(running)
    assert (state.status, state.progress.processed, state.progress.total) == (
        'waiting_region',
        1,
        5,
    )
    assert assist_job.state_from_view(view('running', progress={'stage': 'asking'})).status == (
        'running'
    )


def test_a_cancelling_job_still_reads_running_and_a_finished_one_carries_its_result() -> None:
    assert assist_job.state_from_view(view('cancelling')).status == 'running'
    done = view('completed', result={'proposal': None}, error_message=None)
    state = assist_job.state_from_view(done)
    assert (state.status, state.progress.total) == ('completed', RESOLVED.sampled)
    assert state.result is not None


@pytest.mark.asyncio
async def test_the_worker_refuses_a_run_whose_pack_changed_since_it_was_planned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inputs(_opensearch, _request):
        vlm = SimpleNamespace(name='ua', revision=1, endpoint=None)
        pack = SimpleNamespace(name='newer-pack')
        return None, 2, pack, vlm

    monkeypatch.setattr('src.services.curation.assist.prepare._resolve_inputs', inputs)
    with pytest.raises(RuntimeError, match='changed since the run was planned'):
        await prepare_queued(object(), REQUEST, RESOLVED, sampled_ids=['c1'], box_ids=None)


@pytest.mark.asyncio
async def test_the_worker_accepts_a_run_whose_inputs_are_the_planned_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inputs(_opensearch, _request):
        vlm = SimpleNamespace(name='ua', revision=1, endpoint=None)
        pack = SimpleNamespace(name='generic')
        return None, None, pack, vlm

    monkeypatch.setattr('src.services.curation.assist.prepare._resolve_inputs', inputs)
    run = await prepare_queued(object(), REQUEST, RESOLVED, sampled_ids=['c1'], box_ids=None)
    assert run.scope.sampled_ids == ['c1']
    assert (run.scope.digest, run.scope.seed) == ('sha256:x', 7)
