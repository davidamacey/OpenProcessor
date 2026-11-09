"""Training runs live in ``training_runs``; no per-project job directory is configured any more.

``autolabel_dir`` is the one directory left: it goes with the auto-label job family
(4F-cleanup), and this test drops its exemption then.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config.curation import CurationConfig
from src.config.curation_accessors import PROJECT_SCOPED_FIELDS
from src.config.projects import ProjectResources
from src.config.retired_env import RETIRED_ENV, reject_retired_env


EXEMPT_UNTIL_4F_CLEANUP = {'autolabel_dir'}


def _job_dir_fields(names: set[str]) -> set[str]:
    return {
        n for n in names if n.endswith(('jobs_dir', '_jobs', 'job_dir')) or n == 'autolabel_dir'
    }


def test_no_resource_or_config_field_names_a_job_directory() -> None:
    resources = {f.name for f in dataclasses.fields(ProjectResources)}
    config = {f.name for f in dataclasses.fields(CurationConfig)}
    leftovers = _job_dir_fields(resources | config | set(PROJECT_SCOPED_FIELDS))
    assert leftovers - EXEMPT_UNTIL_4F_CLEANUP == set()


def test_the_train_jobs_dir_variable_is_retired(monkeypatch: pytest.MonkeyPatch) -> None:
    assert RETIRED_ENV['OP_TRAIN_JOBS_DIR'].startswith('removed:')
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', '/jobs')
    with pytest.raises(RuntimeError, match='OP_TRAIN_JOBS_DIR was removed'):
        reject_retired_env()
