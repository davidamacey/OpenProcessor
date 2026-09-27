"""P2: training jobs are project-scoped end to end (projects_plan.md §5.3).

Covers: job.json carries `project`/`project_export_root`/
`mlflow_experiment`; the trainer's glob finds jobs in the default dir and
a project dir, FIFO; a job whose `export_dir` escapes `project_export_root`
is refused; `mlflow_experiment` comes from the spec; the arbiter's
`all_train_jobs_dirs()` includes a project dir with a running job.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import DEFAULT_SLUG, ProjectRecord, resources_for_new
from src.services.training import jobs as train_jobs


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


def _spec(export_dir: Path) -> train_jobs.TrainJobSpec:
    export_dir.mkdir(parents=True, exist_ok=True)
    return train_jobs.TrainJobSpec(dataset_export_dir=str(export_dir))


def test_write_job_stamps_project_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path / 'jobs'))
    export_dir = tmp_path / 'exports' / 'alpha' / 'current'
    with bind_project(_record('alpha')):
        job_id = asyncio.run(train_jobs.write_job(_spec(export_dir)))
        spec_raw = asyncio.run(train_jobs.read_job_spec(job_id))
    assert spec_raw is not None
    assert spec_raw['project'] == 'alpha'
    assert spec_raw['mlflow_experiment'] == 'openprocessor-alpha'
    assert spec_raw['project_export_root']

    jobs_dir = tmp_path / 'jobs' / 'projects' / 'alpha'
    assert (jobs_dir / f'{job_id}.job.json').is_file()


def test_default_project_job_lands_in_the_default_dir(tmp_path, monkeypatch) -> None:
    from src.config.curation import base_curation_config
    from src.config.projects import new_project_record

    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path / 'jobs'))
    export_dir = tmp_path / 'exports' / 'current'
    with bind_project(new_project_record('default', base_curation_config())):
        job_id = asyncio.run(train_jobs.write_job(_spec(export_dir)))
    # P1R §6.1/D-A: project_jobs_dir() always nests /projects/<slug>,
    # `default` included -- no more env-only unnested special case.
    assert (tmp_path / 'jobs' / 'projects' / 'default' / f'{job_id}.job.json').is_file()


def test_trainer_glob_finds_default_and_project_jobs_fifo(tmp_path) -> None:
    import job_protocol

    list_pending_jobs = job_protocol.list_pending_jobs

    jobs_dir = tmp_path / 'jobs'
    (jobs_dir / 'projects' / 'alpha').mkdir(parents=True)
    (jobs_dir / 'projects' / 'beta').mkdir(parents=True)

    default_job = jobs_dir / 'a_default.job.json'
    default_job.write_text('{}', encoding='utf-8')
    time.sleep(0.01)
    alpha_job = jobs_dir / 'projects' / 'alpha' / 'b_alpha.job.json'
    alpha_job.write_text('{}', encoding='utf-8')
    time.sleep(0.01)
    beta_job = jobs_dir / 'projects' / 'beta' / 'c_beta.job.json'
    beta_job.write_text('{}', encoding='utf-8')

    found = list_pending_jobs(jobs_dir)
    assert found == [default_job, alpha_job, beta_job]


def test_trainer_refuses_export_outside_project(tmp_path) -> None:
    import job_protocol

    ExportOutsideProjectError = job_protocol.ExportOutsideProjectError
    parse_and_validate_job = job_protocol.parse_and_validate_job

    project_root = tmp_path / 'exports' / 'alpha'
    other_export = tmp_path / 'exports' / 'beta' / 'current'
    other_export.mkdir(parents=True)
    project_root.mkdir(parents=True)

    job_path = tmp_path / 'jobs' / 'x.job.json'
    job_path.parent.mkdir(parents=True)
    job_path.write_text(
        json.dumps(
            {
                'job_id': 'x',
                'dataset_export_dir': str(other_export),
                'project': 'alpha',
                'project_export_root': str(project_root),
            }
        ),
        encoding='utf-8',
    )

    with pytest.raises(ExportOutsideProjectError) as excinfo:
        parse_and_validate_job(job_path)
    assert excinfo.value.code == 'export_outside_project'


def test_reject_job_writes_error_code(tmp_path) -> None:
    import job_protocol

    ExportOutsideProjectError = job_protocol.ExportOutsideProjectError
    reject_job = job_protocol.reject_job

    job_path = tmp_path / 'x.job.json'
    job_path.write_text('{}', encoding='utf-8')
    reject_job(job_path, ExportOutsideProjectError('nope'))

    status = json.loads((tmp_path / 'x.status.json').read_text(encoding='utf-8'))
    assert status['error_code'] == 'export_outside_project'
    assert status['state'] == 'failed'


def test_mlflow_experiment_read_from_spec(tmp_path) -> None:
    import job_protocol

    parse_and_validate_job = job_protocol.parse_and_validate_job

    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    job_path = tmp_path / 'x.job.json'
    job_path.write_text(
        json.dumps(
            {
                'job_id': 'x',
                'dataset_export_dir': str(export_dir),
                'mlflow_experiment': 'openprocessor-alpha',
            }
        ),
        encoding='utf-8',
    )
    spec = parse_and_validate_job(job_path)
    assert spec.mlflow_experiment == 'openprocessor-alpha'


def test_arbiter_sees_a_run_in_a_project_dir(tmp_path, monkeypatch) -> None:
    from src.services.projects.registry import ProjectRegistry, set_project_registry
    from src.services.training import gpu_arbiter

    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path / 'jobs'))
    alpha = _record('alpha')
    registry = ProjectRegistry(lambda: None)
    registry._by_slug = {'alpha': alpha}
    registry._revision = 0
    set_project_registry(registry)
    try:
        dirs = gpu_arbiter.all_train_jobs_dirs()
        assert dirs['alpha'] == alpha.resources.train_jobs_dir
        assert DEFAULT_SLUG in dirs
    finally:
        set_project_registry(None)


def test_arbiter_keeps_the_gpu_claimed_for_another_projects_bakeoff(tmp_path) -> None:
    """A bake-off queued by a non-default project must keep the GPU services
    stopped: the reconcile loop's bakeoff_active() check has to see every
    project's bakeoff_jobs_dir, not only the arbiter config's single dir."""
    import dataclasses

    from src.services.projects.registry import ProjectRegistry, set_project_registry
    from src.services.training import gpu_arbiter

    alpha = _record('alpha')
    alpha = dataclasses.replace(
        alpha,
        resources=dataclasses.replace(alpha.resources, bakeoff_jobs_dir=tmp_path / 'alpha_bo'),
    )
    (tmp_path / 'alpha_bo').mkdir()
    registry = ProjectRegistry(lambda: None)
    registry._by_slug = {'alpha': alpha}
    registry._revision = 0
    set_project_registry(registry)
    try:
        assert gpu_arbiter.bakeoff_active() is False
        (tmp_path / 'alpha_bo' / 'bo1.job.json').write_text('{}', encoding='utf-8')
        assert gpu_arbiter.bakeoff_active() is True
    finally:
        set_project_registry(None)
