"""The `promote` job on PostgreSQL: one live promotion per Triton name, the job path, withdrawal.

The Triton handoff itself is faked (``execute_promote``); what runs for real is the repository
(rows, the partial unique index, the advisory lock), ``start_job`` (exclusivity, one message) and
the registered task through the job wrapper.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fixtures.job_rows import make_project, sql
from promote.conftest import make_plan

from src.core.exceptions import RequestRejectedError
from src.db.repos import models as repo
from src.db.session import async_session_factory, project_tx
from src.schemas.train_promote import PromoteResponse
from src.services.training import promote_job


pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def new_model(project, name: str, labels=('car', 'truck')) -> repo.NewModel:
    return repo.NewModel(
        project_id=project,
        name=name,
        origin='trained',
        family='yolo26',
        task='detect',
        labels=labels,
        artifact_storage_key=f'models/{name}/1/model.onnx',
        created_by='test',
    )


async def promote(project, name: str, version: int, labels=('car', 'truck')):
    async with async_session_factory()() as session, project_tx(session, project):
        return await repo.record_promotion(
            session,
            model=new_model(project, name, labels),
            triton_model_name=name,
            triton_version=version,
            class_remap=None,
            job_id=None,
            promoted_by='test',
        )


def live_rows(pg, name: str):
    return sql(
        pg,
        'SELECT triton_version, superseded_at IS NULL FROM promotions '
        'WHERE triton_model_name = %s ORDER BY triton_version',
        name,
    )


async def test_a_second_promote_of_a_name_supersedes_the_first(pg, job_env) -> None:
    project = make_project(pg)
    await promote(project, 'cars__det', 1)
    await promote(project, 'cars__det', 2)

    assert live_rows(pg, 'cars__det') == [(1, False), (2, True)]
    versions = sql(pg, "SELECT version FROM models WHERE name = 'cars__det' ORDER BY version")
    assert versions == [(1,), (2,)]


async def test_two_concurrent_promotes_of_one_name_leave_one_live_promotion(pg, job_env) -> None:
    project = make_project(pg)
    results = await asyncio.gather(
        promote(project, 'cars__race', 1), promote(project, 'cars__race', 2)
    )

    assert len(results) == 2
    live = [row for row in live_rows(pg, 'cars__race') if row[1]]
    assert len(live) == 1
    assert len(live_rows(pg, 'cars__race')) == 2


async def test_withdrawing_a_promotion_makes_the_previous_one_live_again(pg, job_env) -> None:
    project = make_project(pg)
    await promote(project, 'cars__wd', 1)
    _, second = await promote(project, 'cars__wd', 2)

    async with async_session_factory()() as session, project_tx(session, project):
        await repo.withdraw_promotion(session, second.id)

    assert live_rows(pg, 'cars__wd') == [(1, True)]
    assert sql(pg, "SELECT count(*) FROM models WHERE name = 'cars__wd'") == [(1,)]


async def test_the_plan_survives_the_job_parameters() -> None:
    plan, payload = make_plan()
    again, again_payload = promote_job.plan_from_params(promote_job.plan_to_params(plan, payload))

    assert again == plan
    assert again_payload == payload


async def test_start_queues_one_job_and_a_second_start_of_the_name_is_busy(
    pg, job_env, producer, wakes
) -> None:
    project = make_project(pg)
    plan, payload = make_plan()
    async with async_session_factory()() as session:
        started = await promote_job.start_promote(session, project, plan, payload, requested_by='t')
    assert started.created is True
    assert producer.sent() == [str(started.job.id)]
    [(kind, status, scope)] = sql(pg, 'SELECT kind, status, scope FROM jobs')
    assert (kind, status, scope) == ('promote', 'queued', 'cars__det')

    async with async_session_factory()() as session:
        with pytest.raises(RequestRejectedError) as busy:
            await promote_job.start_promote(session, project, plan, payload, requested_by='t')
    assert (busy.value.status_code, busy.value.detail['error']) == (409, 'job_busy')

    other, _ = make_plan('cars__other')
    async with async_session_factory()() as session:
        again = await promote_job.start_promote(session, project, other, payload, requested_by='t')
    assert again.created is True  # exclusivity is per Triton model name


@pytest.fixture
def worker_app(queue_env):
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


async def test_the_registered_task_records_the_promotion_and_completes_the_job(
    pg, job_env, producer, wakes, worker_app, monkeypatch
) -> None:
    project = make_project(pg, 'cars')
    plan, payload = make_plan('cars__task')
    async with async_session_factory()() as session:
        started = await promote_job.start_promote(session, project, plan, payload, requested_by='t')
    job_id = started.job.id

    async def fake_execute(the_plan, the_payload, recorder, on_phase=None):
        on_phase('exporting')
        recorded = await recorder.record(triton_name=the_plan.triton_name, triton_version=1)
        on_phase('loading')
        assert list(recorded.labels) == ['car', 'truck']
        return PromoteResponse(
            job_id=the_plan.job_id,
            triton_name=the_plan.triton_name,
            onnx_path='/m/1/model.onnx',
            config_path='/m/config.pbtxt',
            labels_path='/m/labels.txt',
            triton_loaded=True,
        )

    async def lookup(slug):
        return SimpleNamespace(slug=slug, id=project)

    monkeypatch.setattr('src.services.training.promote_run.execute_promote', fake_execute)
    monkeypatch.setattr(
        'src.services.projects.registry.get_project_registry',
        lambda: SimpleNamespace(lookup=lookup),
    )

    outcome = await asyncio.to_thread(
        worker_app.tasks['op.promote.run'].apply, args=[str(job_id)], task_id=str(job_id)
    )
    assert outcome.successful(), outcome.traceback

    [(status, result)] = sql(pg, 'SELECT status, result FROM jobs WHERE id = %s', job_id)
    assert status == 'completed'
    assert result['triton_name'] == 'cars__task'
    assert sql(
        pg,
        'SELECT m.labels, p.triton_version, p.job_id = %s FROM promotions p '
        "JOIN models m ON m.id = p.model_id WHERE p.triton_model_name = 'cars__task'",
        job_id,
    ) == [(['car', 'truck'], 1, True)]


async def test_a_failing_promote_fails_the_job_and_leaves_no_promotion(
    pg, job_env, producer, wakes, worker_app, monkeypatch
) -> None:
    project = make_project(pg, 'carsbad')
    plan, payload = make_plan('cars__bad')
    async with async_session_factory()() as session:
        started = await promote_job.start_promote(session, project, plan, payload, requested_by='t')
    job_id = started.job.id

    async def failing(the_plan, the_payload, recorder, on_phase=None):
        await recorder.record(triton_name=the_plan.triton_name, triton_version=1)
        await recorder.withdraw()
        raise RequestRejectedError(502, 'triton refused the load')

    async def lookup(slug):
        return SimpleNamespace(slug=slug, id=project)

    monkeypatch.setattr('src.services.training.promote_run.execute_promote', failing)
    monkeypatch.setattr(
        'src.services.projects.registry.get_project_registry',
        lambda: SimpleNamespace(lookup=lookup),
    )
    await asyncio.to_thread(
        worker_app.tasks['op.promote.run'].apply, args=[str(job_id)], task_id=str(job_id)
    )

    [(status, message)] = sql(pg, 'SELECT status, error_message FROM jobs WHERE id = %s', job_id)
    assert status == 'failed'
    assert 'triton refused the load' in message
    assert live_rows(pg, 'cars__bad') == []


async def test_cancel_while_queued_cancels_the_job(pg, job_env, producer, wakes) -> None:
    from src.services.jobs.service import cancel_job

    project = make_project(pg)
    plan, payload = make_plan()
    async with async_session_factory()() as session:
        started = await promote_job.start_promote(session, project, plan, payload, requested_by='t')
    async with async_session_factory()() as session:
        cancelled = await cancel_job(session, started.job.id, project, requested_by='t')
    assert cancelled.job.status == 'cancelled'


async def test_the_wire_status_follows_the_job_row(pg, job_env, producer, wakes) -> None:
    project = make_project(pg)
    plan, payload = make_plan()
    async with async_session_factory()() as session:
        started = await promote_job.start_promote(session, project, plan, payload, requested_by='t')
    async with async_session_factory()() as session:
        wire = await promote_job.read_job(session, project, started.job.id)
    assert wire is not None
    assert (wire.status, wire.job_id, wire.triton_name) == ('queued', 'run-1', 'cars__det')
    assert wire.poll_after_s == 3
    async with async_session_factory()() as session:
        latest = await promote_job.latest_for_run(session, project, 'run-1')
    assert latest is not None
    assert latest.promote_id == str(started.job.id)


async def test_another_projects_live_promotion_of_the_name_is_refused(pg, job_env) -> None:
    first, second = make_project(pg), make_project(pg)
    await promote(first, 'shared__name', 1)

    with pytest.raises(repo.LivePromotionConflictError):
        await promote(second, 'shared__name', 1)
    assert live_rows(pg, 'shared__name') == [(1, True)]
