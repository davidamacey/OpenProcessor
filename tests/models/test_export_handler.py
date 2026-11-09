"""The ``model_export`` handler on a fake build: phases in order, a cancel between phases, the
failure paths, and every engine fact written for a build that matches its contract."""

from __future__ import annotations

import pytest
from fixtures.job_rows import deployment_session, sql

from src.db.engine import get_sync_engine
from src.db.repos import jobs as repo


pytestmark = pytest.mark.db


def stages(pg, job) -> list[str]:
    rows = sql(
        pg,
        "SELECT payload->'progress'->>'stage' FROM events WHERE job_id = %s "
        "AND type = 'job.progress' ORDER BY id",
        job.id,
    )
    ordered: list[str] = []
    for (stage,) in rows:
        if not ordered or ordered[-1] != stage:
            ordered.append(stage)
    return ordered


def test_a_fake_build_completes_in_the_phases_in_order(pg, export_env) -> None:
    job = export_env.start_job()
    export_env.run(job)
    status, code, _msg, result = export_env.job_row(pg, job)
    assert (status, code) == ('completed', None)
    assert stages(pg, job) == ['validating', 'exporting', 'loading']
    assert result['triton_loaded'] is True
    assert result['engines'] == [export_env.engine_of(job)]
    assert set(result['step_times']) >= {'triton_load'}
    assert export_env.loads == [(export_env.engine_of(job), True)]


def test_a_matching_build_writes_every_engine_column(pg, export_env) -> None:
    job = export_env.start_job()
    export_env.run(job)
    [row] = sql(
        pg,
        'SELECT status, precision, input_size, input_contract, output_contract, build_profile, '
        'build_settings, build_stamp, onnx_sha256, engine_sha256, device_mib_measured, max_batch, '
        'opt_batch, trt_version, gpu_arch, engine_storage_key, build_job_id, built_at '
        'FROM model_engines WHERE triton_model_name = %s',
        export_env.engine_of(job),
    )
    (status, precision, size, inp, out, profile, settings, stamp, onnx, sha, mib, maxb, opt, trt,
     arch, key, build_job, built_at) = row  # fmt: skip
    assert (status, precision, size, profile) == ('ready', 'fp16', 640, 'default')
    assert (inp['dtype'], inp['size'], inp['resize'], inp['fill']) == (
        'uint8',
        [640, 640],
        'letterbox',
        114,
    )
    assert out['kind'] == 'end2end_boxes'
    assert out['dtypes'] == ['int32', 'fp32', 'fp16', 'int32']
    assert settings['max_batch'] == 16
    assert settings['opt_batch'] == 8
    assert stamp.startswith('min1/opt8/max16 640x640 ')
    assert onnx is None  # the fake build leaves no ONNX
    assert len(sha) == 64
    assert (mib, maxb, opt, trt, arch) == (900, 16, 8, '11.1.0.106', 'sm_86')
    assert key == f'models/{export_env.engine_of(job)}/1/model.plan'
    assert build_job == job.id
    assert built_at is not None


def test_auto_load_off_loads_once_to_measure_and_does_not_keep_it(pg, export_env) -> None:
    job = export_env.start_job(auto_load=False)
    export_env.run(job)
    assert export_env.loads == [(export_env.engine_of(job), False)]
    [(mib,)] = sql(
        pg,
        'SELECT device_mib_measured FROM model_engines WHERE triton_model_name = %s',
        export_env.engine_of(job),
    )
    assert mib == 900
    assert export_env.job_row(pg, job)[3]['triton_loaded'] is False


def test_a_cancel_between_phases_ends_cancelled_and_removes_the_partial_engine(
    pg, export_env, monkeypatch
) -> None:
    import export_tasks

    real = export_tasks._run_build

    async def build_then_cancel(ctx, cmd, step_times):
        await real(ctx, cmd, step_times)
        with deployment_session(get_sync_engine()) as session:
            repo.cancel(session, ctx.job_id, 'test')
        await ctx.progress(
            1, 100, 'exporting', force=True
        )  # the write that brings the request back
        return True

    monkeypatch.setattr(export_tasks, '_run_build', build_then_cancel)
    job = export_env.start_job()
    export_env.run(job)
    assert export_env.job_row(pg, job)[0] == 'cancelled'
    assert not (export_env.models_dir / export_env.engine_of(job)).exists(), (
        'partial engine removed'
    )
    assert export_env.statuses(pg, job) == ['failed']
    assert export_env.loads == []


def test_a_cancel_keeps_the_directory_of_an_engine_that_was_already_ready(
    pg, export_env, monkeypatch
) -> None:
    import export_tasks

    first = export_env.start_job()
    export_env.run(first)
    assert export_env.job_row(pg, first)[0] == 'completed'
    real = export_tasks._run_build

    async def build_then_cancel(ctx, cmd, step_times):
        await real(ctx, cmd, step_times)
        with deployment_session(get_sync_engine()) as session:
            repo.cancel(session, ctx.job_id, 'test')
        await ctx.progress(1, 100, 'exporting', force=True)
        return True

    monkeypatch.setattr(export_tasks, '_run_build', build_then_cancel)
    second = export_env.start_job(name=first.params['triton_name'])
    export_env.run(second)
    assert export_env.job_row(pg, second)[0] == 'cancelled'
    assert (export_env.models_dir / export_env.engine_of(first)).exists()
    assert export_env.statuses(pg, first) == ['failed', 'ready']


def test_a_failing_build_fails_the_job_with_the_output_tail_and_the_engine_row(
    pg, export_env, monkeypatch, tmp_path
) -> None:
    import export_tasks

    boom = tmp_path / 'boom.py'
    boom.write_text('import sys\nprint("boom: no TensorRT")\nsys.exit(3)\n')
    monkeypatch.setattr(
        export_tasks, 'build_command', lambda _job: [__import__('sys').executable, str(boom)]
    )
    job = export_env.start_job()
    export_env.run(job)
    status, code, message, _ = export_env.job_row(pg, job)
    assert (status, code) == ('failed', 'handler_error')
    assert 'boom: no TensorRT' in message
    assert export_env.loads == []
    assert export_env.statuses(pg, job) == ['failed']


def test_a_job_without_a_head_kind_fails_before_any_build(pg, export_env) -> None:
    job = export_env.start_job()
    with deployment_session(get_sync_engine()) as session:
        session.connection().exec_driver_sql(
            "UPDATE jobs SET params = params - 'head_kind' WHERE id = %s", (job.id,)
        )
    export_env.run(job)
    status, _code, message, _ = export_env.job_row(pg, job)
    assert status == 'failed'
    assert 're-upload the model' in message
    assert export_env.statuses(pg, job) == []
