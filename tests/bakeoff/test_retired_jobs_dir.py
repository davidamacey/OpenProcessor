"""``OP_BAKEOFF_JOBS_DIR`` is retired: bake-offs are ``bakeoff`` jobs, not job files."""

from __future__ import annotations

import pytest

from src.config.gpu_arbiter import GpuArbiterConfig
from src.config.retired_env import RETIRED_ENV, reject_retired_env


def test_the_job_directory_variable_fails_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    assert RETIRED_ENV['OP_BAKEOFF_JOBS_DIR'].startswith('removed:')
    monkeypatch.setenv('OP_BAKEOFF_JOBS_DIR', '/var/lib/openprocessor/bakeoff_jobs')
    with pytest.raises(RuntimeError, match='OP_BAKEOFF_JOBS_DIR was removed'):
        reject_retired_env()


def test_the_arbiter_config_has_no_job_directory() -> None:
    assert not hasattr(GpuArbiterConfig(), 'bakeoff_jobs_dir')
