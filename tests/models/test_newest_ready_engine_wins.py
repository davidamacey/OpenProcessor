"""When a Triton model name has several ready engine rows the newest build is the one read.

The shipped engines are seeded as ready rows with no build time; a real export of the same name
must replace them in ``ready_engine_contracts`` without deleting them (``model_engines`` has no
``created_at``, so ``built_at`` orders the rows and the UUIDv7 id breaks a tie).
"""

from __future__ import annotations

import datetime as dt

import pytest
from fixtures.job_rows import deployment_session, sql

from src.db.engine import get_sync_engine
from src.db.repos import engine_builds
from src.db.repos.embedding_spaces import ready_engine_contracts


pytestmark = pytest.mark.db

NAME = 'car_trt_end2end'


def facts(arch: str, dtype: str) -> engine_builds.EngineFacts:
    return engine_builds.EngineFacts(
        input_size=640,
        input_contract={'dtype': dtype},
        output_contract={'kind': 'end2end_boxes'},
        build_profile='default',
        build_settings={},
        build_stamp='stamp',
        onnx_sha256=None,
        engine_sha256='0' * 64,
        device_mib_measured=None,
        max_batch=8,
        opt_batch=4,
        precision='fp16',
        trt_version='11.1',
        gpu_arch=arch,
        engine_storage_key='k',
    )


def ready_row(session, model_id, job_id, arch: str, dtype: str, name: str = NAME):
    row = engine_builds.begin_engine(
        session, model_id=model_id, triton_model_name=name, max_batch=8, build_job_id=job_id
    )
    engine_builds.finish_engine(session, row, facts(arch, dtype))
    return row


def read(name: str = NAME) -> dict:
    with deployment_session(get_sync_engine()) as session:
        found = ready_engine_contracts(session)
    return {name: found[name]} if name in found else {}


def test_the_newest_ready_row_wins_and_a_seeded_row_without_a_time_loses(pg, job_env) -> None:
    with deployment_session(get_sync_engine()) as session:
        model = engine_builds.add_model_version(
            session,
            name='car',
            labels=['a'],
            family='yolo',
            artifact_storage_key='k',
            created_by='t',
        )
        old = ready_row(session, model.id, None, 'sm_80', 'fp32')
        new = ready_row(session, model.id, None, 'sm_86', 'uint8')
    now = dt.datetime.now(dt.UTC)
    sql(pg, 'UPDATE model_engines SET built_at = %s WHERE id = %s', now - dt.timedelta(days=1), old)
    sql(pg, 'UPDATE model_engines SET built_at = %s WHERE id = %s', now, new)
    assert read() == {NAME: {'dtype': 'uint8'}}

    sql(pg, 'UPDATE model_engines SET built_at = %s WHERE id = %s', now - dt.timedelta(days=2), new)
    assert read() == {NAME: {'dtype': 'fp32'}}

    sql(pg, 'UPDATE model_engines SET built_at = NULL WHERE id = %s', old)
    assert read() == {NAME: {'dtype': 'uint8'}}, 'a row with no build time loses to one with a time'


def test_a_failed_or_building_row_is_never_read(pg, job_env) -> None:
    other = 'bus_trt_end2end'
    with deployment_session(get_sync_engine()) as session:
        model = engine_builds.add_model_version(
            session,
            name='car',
            labels=['a'],
            family='yolo',
            artifact_storage_key='k',
            created_by='t',
        )
        ready = ready_row(session, model.id, None, 'sm_80', 'fp32', other)
    with deployment_session(get_sync_engine()) as session:
        engine_builds.begin_engine(
            session, model_id=model.id, triton_model_name=other, max_batch=8, build_job_id=None
        )
    assert read(other) == {other: {'dtype': 'fp32'}}
    sql(pg, "UPDATE model_engines SET status = 'failed' WHERE id = %s", ready)
    assert read(other) == {}
