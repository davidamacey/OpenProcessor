"""JobPerfSummary: validated when a throughput job returns it (GPU research S-8, S-11)."""

from __future__ import annotations

import pytest
from fixtures.job_rows import make_project, sql, start
from pydantic import ValidationError

from src.config.job_kinds.kind import JobKind
from src.db.engine import get_sync_engine
from src.schemas.jobs import JobPerfSummary
from src.tasks.runtime import hooks
from src.tasks.runtime.job_task import run_job


GOOD = {
    'images': 1000,
    'wall_s': 50.0,
    'images_per_s': 20.0,
    'decode_routes': {'gpu': 900, 'cpu_small': 100},
    'stage_seconds': {'decode': 20.0, 'embed': 25.0},
    'pe_calls_per_image': 1.0,
    'servers': ['triton-a'],
}


def test_a_valid_summary_validates() -> None:
    summary = JobPerfSummary.model_validate(GOOD)
    assert summary.model_dump() == GOOD


def test_an_unknown_decode_route_fails_validation() -> None:
    bad = {**GOOD, 'decode_routes': {'gpu': 1, 'telepathy': 2}}
    with pytest.raises(ValidationError, match='unknown decode route'):
        JobPerfSummary.model_validate(bad)


def test_a_negative_count_or_an_extra_field_fails_validation() -> None:
    with pytest.raises(ValidationError):
        JobPerfSummary.model_validate({**GOOD, 'images': -1})
    with pytest.raises(ValidationError):
        JobPerfSummary.model_validate({**GOOD, 'surprise': 1})


@pytest.fixture
def throughput_kind(kinds):
    return JobKind(
        name='reprocess',
        task_name='op.test.reprocess',
        queue='jobs',
        project_scope='project',
        soft_limit_s=3600,
        hard_limit_s=4500,
        max_queue_wait_s=3600,
        gpu_affinity='any_card',
    )


@pytest.mark.db
def test_a_valid_summary_round_trips_through_jobs_result(
    pg, job_env, wakes, throughput_kind
) -> None:
    job = start(get_sync_engine(), make_project(pg), 'reprocess')

    async def handler(ctx):
        return {'perf_summary': GOOD, 'reprocessed': 1000}

    run_job(throughput_kind, handler, {}, str(job.id))
    [(status, result)] = sql(pg, 'SELECT status, result FROM jobs WHERE id = %s', job.id)
    assert status == 'completed'
    assert JobPerfSummary.model_validate(result['perf_summary']).images == 1000


@pytest.mark.db
def test_a_reprocess_result_without_a_summary_fails_the_kinds_result_check(
    pg, job_env, wakes, throughput_kind
) -> None:
    job = start(get_sync_engine(), make_project(pg), 'reprocess')

    async def handler(ctx):
        return {'reprocessed': 1000}

    run_job(throughput_kind, handler, {}, str(job.id))
    assert sql(pg, 'SELECT status, error_code FROM jobs WHERE id = %s', job.id) == [
        ('failed', 'result_invalid')
    ]


@pytest.mark.db
def test_an_invalid_summary_fails_the_job(pg, job_env, wakes, throughput_kind) -> None:
    job = start(get_sync_engine(), make_project(pg), 'reprocess')

    async def handler(ctx):
        return {'perf_summary': {**GOOD, 'decode_routes': {'telepathy': 1}}}

    run_job(throughput_kind, handler, {}, str(job.id))
    [(status, code, message)] = sql(
        pg, 'SELECT status, error_code, error_message FROM jobs WHERE id = %s', job.id
    )
    assert (status, code) == ('failed', 'result_invalid')
    assert 'unknown decode route' in message


@pytest.mark.db
def test_other_kinds_need_no_summary(pg, job_env, wakes, kinds) -> None:
    job = start(get_sync_engine(), make_project(pg), 't_plain')

    async def handler(ctx):
        return {'anything': 1}

    run_job(kinds['t_plain'], handler, {}, str(job.id))
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job.id) == [('completed',)]


@pytest.mark.db
def test_an_any_card_kind_gets_the_routers_server_and_others_none(
    pg, job_env, wakes, throughput_kind, kinds
) -> None:
    engine = get_sync_engine()
    project = make_project(pg)
    seen: dict[str, str | None] = {}
    hooks.install_server_chooser(lambda kind: f'triton-for-{kind}')
    try:

        async def handler(ctx):
            seen[ctx.kind] = ctx.server
            return {'perf_summary': GOOD} if ctx.kind == 'reprocess' else {}

        run_job(throughput_kind, handler, {}, str(start(engine, project, 'reprocess').id))
        run_job(kinds['t_plain'], handler, {}, str(start(engine, project, 't_plain').id))
    finally:
        hooks.install_server_chooser(None)
    assert seen == {'reprocess': 'triton-for-reprocess', 't_plain': None}
