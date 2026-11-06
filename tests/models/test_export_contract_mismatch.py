"""5B-4: the export refuses a plan whose input differs from the declared contract (GPU S-13)."""

from __future__ import annotations

import pytest
from fixtures.export_env import plan_facts
from fixtures.job_rows import sql


pytestmark = pytest.mark.db


@pytest.mark.parametrize(
    ('dtype', 'shape', 'word'),
    [
        ('fp32', (-1, 3, 640, 640), 'fp32'),
        ('uint8', (-1, 3, 512, 512), '512'),
        ('uint8', (-1, 1, 640, 640), '[-1, 1, 640, 640]'),
    ],
)
def test_a_plan_that_differs_from_the_contract_ends_failed_contract_mismatch(
    pg, export_env, dtype, shape, word
) -> None:
    export_env.plan = plan_facts(dtype, shape)
    job = export_env.start_job()
    export_env.run(job)
    status, code, message, _ = export_env.job_row(pg, job)
    assert (status, code) == ('failed', 'handler_error')
    assert 'contract_mismatch' in message
    assert word in message
    assert export_env.statuses(pg, job) == ['failed']
    assert export_env.loads == []


def test_a_rebuild_with_a_new_profile_is_a_new_row_not_an_overwrite(
    pg, export_env, monkeypatch
) -> None:
    import trt_build_settings

    first = export_env.start_job()
    export_env.run(first)
    monkeypatch.setenv('OP_TRT_BUILD_TARGET', 'small')
    assert trt_build_settings.build_target() == 'small'
    second = export_env.start_job(name=first.params['triton_name'])
    export_env.run(second)
    rows = sql(
        pg,
        'SELECT build_profile, status FROM model_engines WHERE triton_model_name = %s '
        "AND status = 'ready' ORDER BY build_profile",
        export_env.engine_of(first),
    )
    assert rows == [('default', 'ready'), ('small', 'ready')]
