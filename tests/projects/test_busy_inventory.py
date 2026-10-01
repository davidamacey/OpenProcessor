"""P2: src.services.projects.busy.running_jobs (projects_plan.md §5.4).

Each source function is exercised with a fake and must report only its
own project's job(s) -- another project's file on disk (or the stub
sources' deliberate emptiness) must never leak into the inventory.
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import UTC, datetime

import pytest

from src.config.curation import base_curation_config
from src.config.projects import ProjectRecord, resources_for_new
from src.services.projects.busy import (
    JobRef,
    _autolabel_jobs,
    _bakeoff_jobs,
    _dataset_import_jobs,
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
            autolabel_dir=tmp_path / 'jobs' / 'auto_label' / 'projects' / slug,
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


def test_train_job_label_and_started_at_come_from_the_real_sources(tmp_path, monkeypatch) -> None:
    """P3F pass-3 m-b: `label` is the submitter's own `mlflow_run_name`
    from the companion `<job_id>.job.json` spec (not the job id), and
    `started_at` is a real, non-null ISO timestamp -- the prior pass's
    tests only ever asserted `started_at is None`, so a `_iso_or_none`
    that always returned None would have stayed green."""
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))
    record = _record('alpha', tmp_path)
    jobs_dir = record.resources.train_jobs_dir
    jobs_dir.mkdir(parents=True)
    (jobs_dir / 'a.status.json').write_text(
        json.dumps({'state': 'running', 'started_at': '2026-01-15T08:30:00+00:00'}),
        encoding='utf-8',
    )
    (jobs_dir / 'a.job.json').write_text(
        json.dumps({'mlflow_run_name': 'yolo26-medium-nightly-3'}), encoding='utf-8'
    )

    found = _train_jobs(record)
    assert found == [
        JobRef(
            kind='train',
            job_id='a',
            started_at='2026-01-15T08:30:00+00:00',
            label='yolo26-medium-nightly-3',
        )
    ]


def test_train_job_label_falls_back_to_job_id_with_no_run_name(tmp_path, monkeypatch) -> None:
    """No `job.json`, or one with no `mlflow_run_name` set -> label is
    None, and `lifecycle.running_jobs` (tested separately) is the one
    that falls back to the job id -- documented, not silent."""
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))
    record = _record('alpha', tmp_path)
    jobs_dir = record.resources.train_jobs_dir
    jobs_dir.mkdir(parents=True)
    (jobs_dir / 'a.status.json').write_text(json.dumps({'state': 'running'}), encoding='utf-8')

    assert _train_jobs(record) == [JobRef(kind='train', job_id='a', started_at=None, label=None)]

    (jobs_dir / 'a.job.json').write_text(json.dumps({'mlflow_run_name': None}), encoding='utf-8')
    assert _train_jobs(record) == [JobRef(kind='train', job_id='a', started_at=None, label=None)]


def test_train_job_label_survives_a_non_object_job_json(tmp_path, monkeypatch) -> None:
    """P3F pass-4 nit n-f: a `job.json` that is valid JSON but not an
    object (e.g. a bare list) must not crash the busy preflight.
    `_train_job_label` used to call `.get(...)` unconditionally, raising
    AttributeError on anything that isn't a dict -- this failed
    `running_jobs`, and with it the delete/archive busy check, for a
    hand-edited or corrupted `job.json`."""
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))
    record = _record('alpha', tmp_path)
    jobs_dir = record.resources.train_jobs_dir
    jobs_dir.mkdir(parents=True)
    (jobs_dir / 'a.status.json').write_text(json.dumps({'state': 'running'}), encoding='utf-8')
    (jobs_dir / 'a.job.json').write_text(json.dumps(['list']), encoding='utf-8')

    assert _train_jobs(record) == [JobRef(kind='train', job_id='a', started_at=None, label=None)]


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


def test_autolabel_jobs_running_only(tmp_path) -> None:
    record = _record('alpha', tmp_path)
    state_dir = record.resources.autolabel_dir
    state_dir.mkdir(parents=True)
    (state_dir / 'state.json').write_text(
        json.dumps({'status': 'running', 'job_id': 'al-1'}), encoding='utf-8'
    )
    assert _autolabel_jobs(record) == [JobRef(kind='autolabel', job_id='al-1')]

    (state_dir / 'state.json').write_text(
        json.dumps({'status': 'completed', 'job_id': 'al-1'}), encoding='utf-8'
    )
    assert _autolabel_jobs(record) == []


def test_autolabel_job_started_at_from_real_epoch(tmp_path) -> None:
    """P3F pass-3 m-b: a real (non-zero) epoch `started_at` must come
    through as a non-null ISO timestamp -- the prior pass's tests never
    exercised this, only the "no heartbeat yet" `0.0` sentinel case."""
    record = _record('alpha', tmp_path)
    state_dir = record.resources.autolabel_dir
    state_dir.mkdir(parents=True)
    epoch = 1768000000.0  # 2026-01-10T02:26:40Z
    (state_dir / 'state.json').write_text(
        json.dumps({'status': 'running', 'job_id': 'al-1', 'started_at': epoch}),
        encoding='utf-8',
    )

    found = _autolabel_jobs(record)
    assert len(found) == 1
    assert found[0].started_at is not None
    assert datetime.fromtimestamp(epoch, UTC).isoformat() == found[0].started_at


def test_autolabel_jobs_scoped_to_its_own_dir(tmp_path) -> None:
    alpha = _record('alpha', tmp_path)
    beta = _record('beta', tmp_path)
    alpha.resources.autolabel_dir.mkdir(parents=True)
    beta.resources.autolabel_dir.mkdir(parents=True)
    (alpha.resources.autolabel_dir / 'state.json').write_text(
        json.dumps({'status': 'running', 'job_id': 'a1'}), encoding='utf-8'
    )
    (beta.resources.autolabel_dir / 'state.json').write_text(
        json.dumps({'status': 'idle'}), encoding='utf-8'
    )
    assert _autolabel_jobs(alpha) == [JobRef(kind='autolabel', job_id='a1')]
    assert _autolabel_jobs(beta) == []


def test_export_jobs_stub_reports_nothing(tmp_path) -> None:
    """Documented gap: /export/yolo runs synchronously inside the
    request -- there is no background job file to poll."""
    record = _record('alpha', tmp_path)
    assert _export_jobs(record) == []


IMPORT_A = 'imp_20260101T000000_aaaaaaaa'
IMPORT_B = 'imp_20260101T000000_bbbbbbbb'


def _write_import(record, import_id: str, state: dict, *, heartbeat: bool = True) -> None:
    from src.services.curation.dataset_import.limits import imports_base_dir

    directory = imports_base_dir() / 'projects' / record.slug / import_id
    directory.mkdir(parents=True)
    (directory / 'state.json').write_text(json.dumps({'import_id': import_id, **state}))
    if heartbeat:
        (directory / 'heartbeat').touch()
    assert directory.is_dir()


@pytest.fixture(autouse=True)
def _imports_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('OP_DATASET_IMPORTS_DIR', str(tmp_path / 'imports'))


def test_dataset_import_jobs_reports_nothing_with_no_state_dir(tmp_path) -> None:
    record = _record('alpha', tmp_path)
    assert _dataset_import_jobs(record) == []


@pytest.mark.parametrize('status', ['queued', 'running', 'paused_backpressure', 'undoing'])
def test_dataset_import_jobs_reports_a_live_import(tmp_path, status) -> None:
    record = _record('alpha', tmp_path)
    _write_import(record, IMPORT_A, {'status': status, 'started_at': '2026-01-01T00:00:00'})
    assert _dataset_import_jobs(record) == [
        JobRef(kind='dataset_import', job_id=IMPORT_A, started_at='2026-01-01T00:00:00')
    ]


@pytest.mark.parametrize(
    'status', ['completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted', 'undone']
)
def test_dataset_import_jobs_ignores_terminal_status(tmp_path, status) -> None:
    record = _record('alpha', tmp_path)
    _write_import(record, IMPORT_A, {'status': status})
    assert _dataset_import_jobs(record) == []


def test_dataset_import_jobs_ignores_a_dead_workers_import(tmp_path) -> None:
    """No heartbeat for longer than the stale window: the process died."""
    import os
    import time

    record = _record('alpha', tmp_path)
    _write_import(record, IMPORT_A, {'status': 'running'})
    from src.services.curation.dataset_import.limits import imports_base_dir

    heartbeat = imports_base_dir() / 'projects' / 'alpha' / IMPORT_A / 'heartbeat'
    old = time.time() - 3600
    os.utime(heartbeat, (old, old))
    assert _dataset_import_jobs(record) == []


def test_dataset_import_jobs_scoped_to_its_own_dir(tmp_path) -> None:
    alpha = _record('alpha', tmp_path)
    beta = _record('beta', tmp_path)
    _write_import(alpha, IMPORT_A, {'status': 'running'})
    _write_import(beta, IMPORT_B, {'status': 'running'})
    assert [j.job_id for j in _dataset_import_jobs(alpha)] == [IMPORT_A]
    assert [j.job_id for j in _dataset_import_jobs(beta)] == [IMPORT_B]


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


# The per-project job states of the four in-process curation job runners
# (projects_plan.md §5.4: item-scores, probe, selection, plus the UMAP viz
# job). Each writes ``<env base>/projects/<slug>/state.json`` + a
# ``heartbeat`` file.
_STATE_JOBS = [
    ('probe', 'OP_PROBE_JOBS_DIR'),
    ('scores', 'OP_SCORES_STATE_DIR'),
    ('select', 'OP_SELECT_JOBS_DIR'),
    ('viz', 'OP_VIZ_JOBS_DIR'),
]


def _write_job_state(base, slug: str, status: str, *, heartbeat_age_s: float | None) -> None:
    import os
    import time

    job_dir = base / 'projects' / slug
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'state.json').write_text(
        json.dumps({'job_id': f'{slug}-job-1', 'status': status}), encoding='utf-8'
    )
    heartbeat = job_dir / 'heartbeat'
    if heartbeat_age_s is None:
        heartbeat.unlink(missing_ok=True)
        return
    heartbeat.touch()
    then = time.time() - heartbeat_age_s
    os.utime(heartbeat, (then, then))


@pytest.mark.parametrize(('kind', 'env_var'), _STATE_JOBS)
def test_a_running_curation_job_makes_only_its_own_project_busy(
    tmp_path, monkeypatch, kind: str, env_var: str
) -> None:
    base = tmp_path / kind
    monkeypatch.setenv(env_var, str(base))
    alpha = _record('alpha', tmp_path)
    beta = _record('beta', tmp_path)
    _write_job_state(base, 'alpha', 'running', heartbeat_age_s=1.0)
    _write_job_state(base, 'beta', 'completed', heartbeat_age_s=1.0)

    assert JobRef(kind=kind, job_id='alpha-job-1') in running_jobs(alpha)
    assert [j for j in running_jobs(beta) if j.kind == kind] == []


@pytest.mark.parametrize(('kind', 'env_var'), _STATE_JOBS)
def test_a_just_started_curation_job_with_no_heartbeat_yet_is_busy(
    tmp_path, monkeypatch, kind: str, env_var: str
) -> None:
    base = tmp_path / kind
    monkeypatch.setenv(env_var, str(base))
    alpha = _record('alpha', tmp_path)
    _write_job_state(base, 'alpha', 'running', heartbeat_age_s=None)

    assert JobRef(kind=kind, job_id='alpha-job-1') in running_jobs(alpha)


@pytest.mark.parametrize(('kind', 'env_var'), _STATE_JOBS)
def test_a_curation_job_with_a_stale_heartbeat_is_not_busy(
    tmp_path, monkeypatch, kind: str, env_var: str
) -> None:
    """A ``running`` state whose heartbeat is past the job module's own
    staleness window is an orphan of a dead process, not a live job."""
    base = tmp_path / kind
    monkeypatch.setenv(env_var, str(base))
    alpha = _record('alpha', tmp_path)
    _write_job_state(base, 'alpha', 'running', heartbeat_age_s=3600.0)

    assert [j for j in running_jobs(alpha) if j.kind == kind] == []


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
