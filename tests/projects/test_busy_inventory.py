"""P2: src.services.projects.busy.running_jobs (projects_plan.md §5.4).

Each source function is exercised with a fake and must report only its
own project's job(s) -- another project's file on disk (or the stub
sources' deliberate emptiness) must never leak into the inventory.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime

from src.config.curation import base_curation_config
from src.config.projects import ProjectRecord, resources_for_new
from src.services.projects.busy import (
    JobRef,
    _autolabel_jobs,
    _bakeoff_jobs,
    _detection_worker_inflight,
    _export_jobs,
    _train_jobs,
    running_jobs,
)


def _record(slug: str, tmp_path=None) -> ProjectRecord:
    """A project record whose ``train_jobs_dir``/``bakeoff_jobs_dir``
    are rebased under ``tmp_path`` when given -- ``resources_for_new``
    resolves ``bakeoff_jobs_dir`` from the (process-cached)
    ``base_curation_config().state_dir``, which a per-test
    ``monkeypatch.setenv`` can't retroactively change, so tests that
    need a real writable dir override it explicitly instead."""
    now = datetime.now(UTC).isoformat()
    resources = resources_for_new(slug, base_curation_config())
    if tmp_path is not None:
        resources = dataclasses.replace(
            resources,
            train_jobs_dir=tmp_path / 'jobs' / 'projects' / slug,
            bakeoff_jobs_dir=tmp_path / 'state' / 'projects' / slug / 'bakeoff_jobs',
        )
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


def test_train_jobs_only_non_terminal(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))
    record = _record('alpha', tmp_path)
    jobs_dir = record.resources.train_jobs_dir
    jobs_dir.mkdir(parents=True)
    (jobs_dir / 'a.status.json').write_text(json.dumps({'state': 'running'}), encoding='utf-8')
    (jobs_dir / 'b.status.json').write_text(json.dumps({'state': 'finished'}), encoding='utf-8')
    (jobs_dir / 'c.status.json').write_text(json.dumps({'state': 'lost'}), encoding='utf-8')

    found = _train_jobs(record)
    assert found == [JobRef(kind='train', job_id='a')]


def test_train_jobs_scoped_to_its_own_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))
    alpha = _record('alpha', tmp_path)
    beta = _record('beta', tmp_path)
    alpha.resources.train_jobs_dir.mkdir(parents=True)
    beta.resources.train_jobs_dir.mkdir(parents=True)
    (alpha.resources.train_jobs_dir / 'a.status.json').write_text(
        json.dumps({'state': 'running'}), encoding='utf-8'
    )
    (beta.resources.train_jobs_dir / 'b.status.json').write_text(
        json.dumps({'state': 'running'}), encoding='utf-8'
    )

    assert _train_jobs(alpha) == [JobRef(kind='train', job_id='a')]
    assert _train_jobs(beta) == [JobRef(kind='train', job_id='b')]


def test_bakeoff_jobs_pending_only(tmp_path) -> None:
    record = _record('alpha', tmp_path)
    jobs_dir = record.resources.bakeoff_jobs_dir
    jobs_dir.mkdir(parents=True)
    (jobs_dir / 'x.job.json').write_text('{}', encoding='utf-8')
    done_dir = jobs_dir / 'done'
    done_dir.mkdir()
    (done_dir / 'y.job.json').write_text('{}', encoding='utf-8')

    found = _bakeoff_jobs(record)
    assert found == [JobRef(kind='bakeoff', job_id='x')]


def test_autolabel_and_export_and_detection_stubs_report_nothing(tmp_path) -> None:
    """Documented gaps (owned by other waves/agents), not silently-wrong
    positives: each stub returns [] rather than guessing at a shape."""
    record = _record('alpha', tmp_path)
    assert _autolabel_jobs(record) == []
    assert _export_jobs(record) == []
    assert _detection_worker_inflight(record) == []


def test_running_jobs_aggregates_every_source_for_one_project(tmp_path) -> None:
    alpha = _record('alpha', tmp_path)
    beta = _record('beta', tmp_path)

    alpha.resources.train_jobs_dir.mkdir(parents=True)
    (alpha.resources.train_jobs_dir / 'a.status.json').write_text(
        json.dumps({'state': 'running'}), encoding='utf-8'
    )
    alpha.resources.bakeoff_jobs_dir.mkdir(parents=True)
    (alpha.resources.bakeoff_jobs_dir / 'bo.job.json').write_text('{}', encoding='utf-8')

    beta.resources.train_jobs_dir.mkdir(parents=True)
    (beta.resources.train_jobs_dir / 'z.status.json').write_text(
        json.dumps({'state': 'running'}), encoding='utf-8'
    )

    alpha_jobs = running_jobs(alpha)
    assert JobRef(kind='train', job_id='a') in alpha_jobs
    assert JobRef(kind='bakeoff', job_id='bo') in alpha_jobs
    assert JobRef(kind='train', job_id='z') not in alpha_jobs

    beta_jobs = running_jobs(beta)
    assert beta_jobs == [JobRef(kind='train', job_id='z')]
