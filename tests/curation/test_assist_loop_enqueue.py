# ruff: noqa: F811 - pytest fixtures are imported by name and used as test arguments
"""A loop is a queued job plus its run in the tables: start, the busy refusal, pause, resume,
cancel, budget edits and the control flags the backend decides."""

from __future__ import annotations

import sys
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.job_fixtures import job_env, kinds, producer, wakes  # noqa: F401
from fixtures.job_rows import make_project, sql

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import new_project_record
from src.core.exceptions import RequestRejectedError
from src.db.session import async_session_factory
from src.schemas.curation.assist_loop import AssistLoopBudgetEdit, AssistLoopRequest, LoopBudget
from src.services.curation.assist_loop import service
from src.services.curation.assist_loop.partition import RootSize


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


@pytest.fixture
def bound(pg, job_env):
    slug = f'lp-{uuid.uuid4().hex[:6]}'
    project = make_project(pg, slug)
    record = replace(new_project_record(slug, base_curation_config()), id=project)
    with bind_project(record):
        yield project


def _prepared(crops: int = 500):
    endpoint = SimpleNamespace(model_id='m')
    return SimpleNamespace(
        request=AssistLoopRequest(dry_run=False, budget=LoopBudget(max_vlm_crops=crops)),
        vlm=SimpleNamespace(name='vlm', revision=1, endpoint=endpoint),
        audit_vlm=None,
        pack=SimpleNamespace(name='pack'),
        pack_revision=1,
        images_per_call=4,
        roots=[RootSize(10001, 300, 300)],
    )


async def _start(crops: int = 500):
    async with async_session_factory()() as session:
        return await service.start_loop(session, _prepared(crops))


async def test_a_start_queues_one_job_its_run_budgets_and_first_iteration(
    pg, bound, producer, wakes
) -> None:
    state = await _start(500)

    assert state.status == 'queued'
    assert state.control.state == 'queued'
    assert (state.control.can_pause, state.control.can_resume) == (True, False)
    assert state.control.can_edit_budget is True
    assert state.budget.vlm.max_crops == 500
    ((job_id, kind, params),) = sql(
        pg, 'SELECT id, kind, params FROM jobs WHERE project_id = %s', bound
    )
    assert kind == 'assist_loop'
    assert list(params) == ['request_sha256'], 'jobs.params holds only the request reference'
    assert producer.sent() == [str(job_id)]
    run = sql(pg, 'SELECT id, job_id FROM assist_loop_runs WHERE project_id = %s', bound)
    assert run == [(uuid.UUID(state.loop_id), job_id)]
    limits = dict(
        sql(
            pg,
            "SELECT pool || '.' || unit, limit_value FROM assist_loop_budgets WHERE run_id = %s",
            uuid.UUID(state.loop_id),
        )
    )
    assert limits['vlm.crops'] == 500
    assert limits['vlm.prompt_tokens'] is None
    assert sql(
        pg,
        'SELECT ordinal, status FROM assist_loop_iterations WHERE run_id = %s',
        uuid.UUID(state.loop_id),
    ) == [(1, 'open')]


async def test_a_second_start_while_one_is_live_is_409_assist_loop_busy_naming_it(
    pg, bound, producer, wakes
) -> None:
    first = await _start()

    with pytest.raises(RequestRejectedError) as busy:
        await _start()

    assert busy.value.status_code == 409
    assert busy.value.detail['error'] == 'assist_loop_busy'
    assert busy.value.detail['loop_id'] == first.loop_id


async def test_pause_and_resume_a_queued_loop_and_the_flags_follow(
    pg, bound, producer, wakes
) -> None:
    started = await _start()

    async with async_session_factory()() as session:
        paused = await service.pause(session, started.loop_id)
    assert (paused.status, paused.pause_reason) == ('paused', 'user')
    assert (paused.control.can_resume, paused.control.can_pause) == (True, False)

    async with async_session_factory()() as session:
        resumed = await service.resume(session, started.loop_id)
    assert resumed.status == 'queued'
    assert resumed.pause_reason is None


async def test_a_resume_is_blocked_with_the_reason_while_the_project_is_paused(
    pg, bound, producer, wakes
) -> None:
    started = await _start()
    async with async_session_factory()() as session:
        await service.pause(session, started.loop_id)
    sql(pg, 'UPDATE projects SET paused_at = now() WHERE id = %s', bound)

    async with async_session_factory()() as session:
        state = await service.get_state(session, started.loop_id)
        assert state.control.resume_blocked_reason == 'project_paused'
        assert state.control.can_resume is False
        with pytest.raises(RequestRejectedError) as blocked:
            await service.resume(session, started.loop_id)
    assert blocked.value.status_code == 409
    assert blocked.value.detail['error'] == 'resume_blocked'
    assert blocked.value.detail['reason'] == 'project_paused'


async def test_cancel_is_terminal_and_a_cancelled_loop_cannot_resume_or_edit_its_budget(
    pg, bound, producer, wakes
) -> None:
    started = await _start()

    async with async_session_factory()() as session:
        cancelled = await service.cancel(session, started.loop_id)
    assert cancelled.status == 'cancelled'
    assert cancelled.control.can_cancel is False
    assert cancelled.control.can_rollback is True

    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as gone:
            await service.resume(session, started.loop_id)
    assert gone.value.detail['error'] == 'job_finished'
    edit = AssistLoopBudgetEdit(limits=[{'pool': 'vlm', 'unit': 'crops', 'limit_value': 900}])
    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as locked:
            await service.edit_budget(session, started.loop_id, edit)
    assert locked.value.detail['error'] == 'budget_not_editable'


async def test_a_budget_edit_raises_a_limit_and_refuses_one_below_what_is_spent(
    pg, bound, producer, wakes
) -> None:
    started = await _start(500)
    run = uuid.UUID(started.loop_id)
    sql(
        pg,
        "UPDATE assist_loop_budgets SET spent = 120 WHERE run_id = %s AND pool = 'vlm' AND unit = 'crops'",
        run,
    )

    raised = AssistLoopBudgetEdit(limits=[{'pool': 'vlm', 'unit': 'crops', 'limit_value': 900}])
    async with async_session_factory()() as session:
        state = await service.edit_budget(session, started.loop_id, raised)
    assert state.budget.vlm.max_crops == 900
    assert state.status == 'queued', 'an edit never resumes or starts the loop'

    low = AssistLoopBudgetEdit(limits=[{'pool': 'vlm', 'unit': 'crops', 'limit_value': 100}])
    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as below:
            await service.edit_budget(session, started.loop_id, low)
    assert below.value.status_code == 409
    assert below.value.detail['error'] == 'budget_below_spent'
    assert below.value.detail['floor'] == 120

    uncapped = AssistLoopBudgetEdit(limits=[{'pool': 'vlm', 'unit': 'crops', 'limit_value': None}])
    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as required:
            await service.edit_budget(session, started.loop_id, uncapped)
    assert required.value.status_code == 422


async def test_a_loop_id_of_another_project_or_garbage_is_unknown(
    pg, bound, producer, wakes
) -> None:
    for bad in (str(uuid.uuid4()), 'not-a-uuid'):
        async with async_session_factory()() as session:
            with pytest.raises(RequestRejectedError) as unknown:
                await service.get_state(session, bad)
        assert unknown.value.status_code == 404
