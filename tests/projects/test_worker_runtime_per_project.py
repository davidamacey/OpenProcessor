"""Detection worker: each item's own project's runtime/config, never a
sibling's (projects_plan.md §5.1.5/§5.1.8).

RED-FIRST evidence (pre-fix): before ``_ItemTask.project`` and the
consumer's ``set_bound_project(t.project)`` call existed, a worker
process bound exactly one project for its whole lifetime (``--project``/
``$OP_PROJECT``), so two items from different projects processed by the
same consumer resolved the SAME (whichever-was-bound-last) project's
config -- there was no per-item rebind at all. See the task report for
the captured failure text against the pre-fix code.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from scripts.curation.worker.fairness import read_liveness, write_liveness
from scripts.curation.worker.state import _ItemTask
from src.config.curation import base_curation_config
from src.config.project_context import current_project, set_bound_project
from src.config.projects import ProjectRecord, resources_for_new


pytestmark = pytest.mark.unbound


def _record(tmp_path: Any, slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    resources = resources_for_new(slug, base_curation_config())
    state_dir = tmp_path / slug
    state_dir.mkdir(parents=True, exist_ok=True)
    resources = resources.__class__(**{**resources.__dict__, 'project_state_dir': state_dir})
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',  # type: ignore[arg-type]
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


def test_each_item_resolves_its_own_project_config(tmp_path: Any) -> None:
    alpha = _record(tmp_path, 'alpha')
    beta = _record(tmp_path, 'beta')
    # Distinct "active profile" proxy per project: mlflow_experiment and
    # train_jobs_dir are both real per-project resource fields
    # (PROJECT_SCOPED_FIELDS) that differ by construction for any two
    # projects -- a stand-in for a per-project detection profile, which
    # does not exist as a resource field in this codebase (grepped;
    # only a process-wide OP_REGION_PROFILE env var does).
    assert alpha.resources.mlflow_experiment != beta.resources.mlflow_experiment
    assert alpha.resources.train_jobs_dir != beta.resources.train_jobs_dir

    task_alpha = _ItemTask(
        crop_id='a-1',
        image_path='/a.jpg',
        item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        region_status='pending_detection',
        class_name='',
        project=alpha,
    )
    task_beta = _ItemTask(
        crop_id='b-1',
        image_path='/b.jpg',
        item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        region_status='pending_detection',
        class_name='',
        project=beta,
    )

    # Simulate a single consumer processing task_alpha then task_beta,
    # exactly as runner.py's stage_a_consumer/stage_a_sam_consumer do at
    # the top of their per-item loop body.
    resolved_experiments = []
    resolved_train_dirs = []
    for t in (task_alpha, task_beta):
        set_bound_project(t.project)
        bound = current_project()
        assert bound.record.slug == t.project.slug
        resolved_experiments.append(bound.record.resources.mlflow_experiment)
        resolved_train_dirs.append(bound.record.resources.train_jobs_dir)

    assert resolved_experiments == [
        alpha.resources.mlflow_experiment,
        beta.resources.mlflow_experiment,
    ]
    assert resolved_train_dirs == [alpha.resources.train_jobs_dir, beta.resources.train_jobs_dir]
    # Processing beta must not have left alpha's project bound.
    assert current_project().record.slug == 'beta'


def test_liveness_lands_under_each_projects_own_location_never_merged(tmp_path: Any) -> None:
    alpha = _record(tmp_path, 'alpha')
    beta = _record(tmp_path, 'beta')

    write_liveness(alpha, inflight=3, applied=True, paused=False, host='workerhost')
    write_liveness(beta, inflight=7, applied=True, paused=False, host='workerhost')

    alpha_doc = read_liveness(alpha, host='workerhost')
    beta_doc = read_liveness(beta, host='workerhost')
    assert alpha_doc is not None
    assert beta_doc is not None
    assert alpha_doc['project'] == 'alpha'
    assert alpha_doc['inflight'] == 3
    assert beta_doc['project'] == 'beta'
    assert beta_doc['inflight'] == 7

    # Each doc must live under its OWN project's state dir -- not one
    # shared file, and not visible under the other project's dir.
    alpha_path = alpha.resources.project_state_dir / 'runtime_detection_worker_workerhost.json'
    beta_path = beta.resources.project_state_dir / 'runtime_detection_worker_workerhost.json'
    assert alpha_path.exists()
    assert beta_path.exists()
    assert alpha_path != beta_path
    assert alpha_path.read_text() != beta_path.read_text()
