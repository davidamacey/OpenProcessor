"""P2: training runs are project-scoped end to end (projects_plan.md §5.3).

Covers: a run whose `export_dir` escapes `project_export_root` is refused by the trainer;
`mlflow_experiment` comes from the spec; the arbiter's `all_train_jobs_dirs()` includes a project
dir. (That the start path stamps the project fields is tested in `tests/training`.)
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.config.curation_accessors import base_curation_config
from src.config.projects import DEFAULT_SLUG, ProjectRecord, resources_for_new


pytestmark = pytest.mark.unbound

TRAINER_DIR = Path(__file__).resolve().parents[2] / 'docker' / 'trainer'
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


def test_trainer_refuses_export_outside_project(tmp_path) -> None:
    import job_spec

    project_root = tmp_path / 'exports' / 'alpha'
    other_export = tmp_path / 'exports' / 'beta' / 'current'
    other_export.mkdir(parents=True)
    project_root.mkdir(parents=True)

    with pytest.raises(job_spec.ExportOutsideProjectError) as excinfo:
        job_spec.parse_and_validate_spec(
            'x',
            {
                'dataset_export_dir': str(other_export),
                'project': 'alpha',
                'project_export_root': str(project_root),
            },
        )
    assert excinfo.value.code == 'export_outside_project'


def test_mlflow_experiment_read_from_spec(tmp_path) -> None:
    import job_spec

    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    spec = job_spec.parse_and_validate_spec(
        'x', {'dataset_export_dir': str(export_dir), 'mlflow_experiment': 'openprocessor-alpha'}
    )
    assert spec.mlflow_experiment == 'openprocessor-alpha'


def test_arbiter_sees_a_run_in_a_project_dir(tmp_path, monkeypatch) -> None:
    from src.services.projects.registry import ProjectRegistry, set_project_registry
    from src.services.training import arbiter_dirs

    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path / 'jobs'))
    alpha = _record('alpha')
    registry = ProjectRegistry()
    registry._by_slug = {'alpha': alpha}
    registry._revision = 0
    set_project_registry(registry)
    try:
        dirs = arbiter_dirs.all_train_jobs_dirs()
        assert dirs['alpha'] == alpha.resources.train_jobs_dir
        assert DEFAULT_SLUG in dirs
    finally:
        set_project_registry(None)
