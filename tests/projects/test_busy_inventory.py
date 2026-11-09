"""The detection worker's in-flight batch still counts as busy (it is a stream worker, not a job).

The jobs side of the inventory is a query on ``jobs`` rows: ``tests/projects/test_busy_from_jobs.py``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import UTC, datetime

from src.config.curation_accessors import base_curation_config
from src.config.projects import ProjectRecord, resources_for_new
from src.services.projects.busy import JobRef, _detection_worker_inflight


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
            project_state_dir=tmp_path / 'state' / 'projects' / slug,
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


def test_detection_worker_inflight_reads_the_real_liveness_file(tmp_path) -> None:
    """fairness.py's write_liveness drops one runtime_detection_worker_
    <host>.json per host under project_state_dir every cycle."""
    record = _record('alpha', tmp_path)
    state_dir = record.resources.project_state_dir
    state_dir.mkdir(parents=True)
    (state_dir / 'runtime_detection_worker_workerhost.json').write_text(
        json.dumps(
            {
                'inflight': 3,
                'applied': True,
                'paused': False,
                'updated_at': time.time(),
                'host': 'workerhost',
            }
        ),
        encoding='utf-8',
    )
    (state_dir / 'runtime_detection_worker_idlehost.json').write_text(
        json.dumps(
            {
                'inflight': 0,
                'applied': True,
                'paused': False,
                'updated_at': time.time(),
                'host': 'idlehost',
            }
        ),
        encoding='utf-8',
    )

    assert _detection_worker_inflight(record) == [
        JobRef(kind='detection_worker', job_id='workerhost')
    ]


def test_detection_worker_inflight_scoped_to_its_own_dir(tmp_path) -> None:
    alpha = _record('alpha', tmp_path)
    beta = _record('beta', tmp_path)
    alpha.resources.project_state_dir.mkdir(parents=True)
    beta.resources.project_state_dir.mkdir(parents=True)
    (alpha.resources.project_state_dir / 'runtime_detection_worker_h1.json').write_text(
        json.dumps({'inflight': 1, 'updated_at': time.time(), 'host': 'h1'}), encoding='utf-8'
    )
    (beta.resources.project_state_dir / 'runtime_detection_worker_h2.json').write_text(
        json.dumps({'inflight': 1, 'updated_at': time.time(), 'host': 'h2'}), encoding='utf-8'
    )

    assert _detection_worker_inflight(alpha) == [JobRef(kind='detection_worker', job_id='h1')]
    assert _detection_worker_inflight(beta) == [JobRef(kind='detection_worker', job_id='h2')]


def test_a_crashed_detection_workers_stale_liveness_file_is_not_busy(tmp_path) -> None:
    """A worker that died mid-batch leaves ``inflight > 0`` behind. Past
    the worker heartbeat window it must stop counting as busy, or the
    project could never be deleted or archived."""
    import time

    from src.services.curation.worker_liveness import DEFAULT_MAX_AGE_S

    record = _record('alpha', tmp_path)
    state_dir = record.resources.project_state_dir
    state_dir.mkdir(parents=True)
    (state_dir / 'runtime_detection_worker_dead.json').write_text(
        json.dumps(
            {'inflight': 4, 'host': 'dead', 'updated_at': time.time() - DEFAULT_MAX_AGE_S - 5}
        ),
        encoding='utf-8',
    )
    (state_dir / 'runtime_detection_worker_live.json').write_text(
        json.dumps({'inflight': 2, 'host': 'live', 'updated_at': time.time()}),
        encoding='utf-8',
    )

    assert _detection_worker_inflight(record) == [JobRef(kind='detection_worker', job_id='live')]
