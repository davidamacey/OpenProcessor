"""Tests for startup-time orphaned-job reconciliation (persistence-hardening
Gap 1, ``docs/design/curation_design_rationale.md``).

Each of ``item_scores.job``, ``selection.job`` and ``embedding_viz`` shares
the same state.json/heartbeat/cancel.flag file-backed job-runner shape and
delegates its ``reconcile_orphaned_jobs()`` to the shared
:func:`src.services.curation.job_reconcile.reconcile_stale_running` helper.
``autolabel.job`` has a materially different shape (a separate worker
container + a pending-trigger file) and is tested on its own below.

Every test redirects the module's state-dir env var to a per-test
``tmp_path`` (same convention ``test_scores_router.py`` /
``test_select_router.py`` / ``test_embedding_viz.py`` use) so nothing here
touches a real ``/jobs`` mount.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pytest


# =============================================================================
# Shared helper — direct unit tests
# =============================================================================


def test_reconcile_stale_running_no_state_file_is_a_noop(tmp_path) -> None:
    from src.services.curation.job_reconcile import reconcile_stale_running

    assert (
        reconcile_stale_running(
            tmp_path / 'state.json', tmp_path / 'heartbeat', stale_s=30.0, error_prefix='x'
        )
        is False
    )


def test_reconcile_stale_running_ignores_non_running_status(tmp_path) -> None:
    from src.services.curation.job_reconcile import reconcile_stale_running

    state_file = tmp_path / 'state.json'
    state_file.write_text(json.dumps({'status': 'completed'}))
    assert (
        reconcile_stale_running(state_file, tmp_path / 'heartbeat', stale_s=30.0, error_prefix='x')
        is False
    )
    assert json.loads(state_file.read_text())['status'] == 'completed'


def test_reconcile_stale_running_marks_orphaned_run_interrupted(tmp_path) -> None:
    from src.services.curation.job_reconcile import reconcile_stale_running

    state_file = tmp_path / 'state.json'
    heartbeat_file = tmp_path / 'heartbeat'
    state_file.write_text(json.dumps({'job_id': 'abc', 'status': 'running'}))
    heartbeat_file.touch()
    # Backdate the heartbeat well past staleness.
    old = time.time() - 120
    os.utime(heartbeat_file, (old, old))

    changed = reconcile_stale_running(
        state_file, heartbeat_file, stale_s=30.0, error_prefix='widget job'
    )
    assert changed is True
    raw = json.loads(state_file.read_text())
    assert raw['status'] == 'interrupted'
    assert 'widget job' in raw['error']
    assert raw['finished_at']


def test_reconcile_stale_running_missing_heartbeat_is_orphaned(tmp_path) -> None:
    """No heartbeat file at all + status='running' -- at startup nothing
    legitimate could be running yet, so this must also be reconciled
    (unlike a module's own live _is_busy() check, which treats a missing
    heartbeat as 'hasn't ticked yet, still busy')."""
    from src.services.curation.job_reconcile import reconcile_stale_running

    state_file = tmp_path / 'state.json'
    state_file.write_text(json.dumps({'job_id': 'abc', 'status': 'running'}))

    changed = reconcile_stale_running(
        state_file, tmp_path / 'heartbeat', stale_s=30.0, error_prefix='widget job'
    )
    assert changed is True
    assert json.loads(state_file.read_text())['status'] == 'interrupted'


def test_reconcile_stale_running_fresh_heartbeat_is_left_alone(tmp_path) -> None:
    """A heartbeat younger than stale_s means a genuinely live run --
    must not be touched."""
    from src.services.curation.job_reconcile import reconcile_stale_running

    state_file = tmp_path / 'state.json'
    heartbeat_file = tmp_path / 'heartbeat'
    state_file.write_text(json.dumps({'job_id': 'abc', 'status': 'running'}))
    heartbeat_file.touch()

    changed = reconcile_stale_running(
        state_file, heartbeat_file, stale_s=30.0, error_prefix='widget job'
    )
    assert changed is False
    assert json.loads(state_file.read_text())['status'] == 'running'


# =============================================================================
# item_scores.job.reconcile_orphaned_jobs
# =============================================================================


def test_item_scores_job_reconciles_orphaned_running_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_SCORES_STATE_DIR', str(tmp_path / 'scores'))
    from src.services.curation.item_scores import job

    (tmp_path / 'scores').mkdir()
    state = {'job_id': 'j1', 'status': 'running', 'scorers': ['uniqueness']}
    (tmp_path / 'scores' / 'state.json').write_text(json.dumps(state))
    heartbeat = tmp_path / 'scores' / 'heartbeat'
    heartbeat.touch()
    old = time.time() - 120
    os.utime(heartbeat, (old, old))

    assert job.reconcile_orphaned_jobs() is True
    on_disk = json.loads((tmp_path / 'scores' / 'state.json').read_text())
    assert on_disk['status'] == 'interrupted'


def test_item_scores_job_leaves_fresh_running_state_alone(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_SCORES_STATE_DIR', str(tmp_path / 'scores'))
    from src.services.curation.item_scores import job

    (tmp_path / 'scores').mkdir()
    (tmp_path / 'scores' / 'state.json').write_text(
        json.dumps({'job_id': 'j1', 'status': 'running'})
    )
    (tmp_path / 'scores' / 'heartbeat').touch()

    assert job.reconcile_orphaned_jobs() is False
    on_disk = json.loads((tmp_path / 'scores' / 'state.json').read_text())
    assert on_disk['status'] == 'running'


# =============================================================================
# selection.job.reconcile_orphaned_jobs
# =============================================================================


def test_selection_job_reconciles_orphaned_running_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_SELECT_JOBS_DIR', str(tmp_path / 'select'))
    from src.services.curation.selection import job

    (tmp_path / 'select').mkdir()
    (tmp_path / 'select' / 'state.json').write_text(
        json.dumps({'job_id': 'j2', 'status': 'running', 'k': 100})
    )
    heartbeat = tmp_path / 'select' / 'heartbeat'
    heartbeat.touch()
    old = time.time() - 120
    os.utime(heartbeat, (old, old))

    assert job.reconcile_orphaned_jobs() is True
    on_disk = json.loads((tmp_path / 'select' / 'state.json').read_text())
    assert on_disk['status'] == 'interrupted'


def test_selection_job_leaves_fresh_running_state_alone(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_SELECT_JOBS_DIR', str(tmp_path / 'select'))
    from src.services.curation.selection import job

    (tmp_path / 'select').mkdir()
    (tmp_path / 'select' / 'state.json').write_text(
        json.dumps({'job_id': 'j2', 'status': 'running'})
    )
    (tmp_path / 'select' / 'heartbeat').touch()

    assert job.reconcile_orphaned_jobs() is False
    on_disk = json.loads((tmp_path / 'select' / 'state.json').read_text())
    assert on_disk['status'] == 'running'


# =============================================================================
# probe_job.reconcile_orphaned_jobs
# =============================================================================


def test_probe_job_reconciles_orphaned_running_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_PROBE_JOBS_DIR', str(tmp_path / 'probe'))
    from src.services.curation import probe_job

    (tmp_path / 'probe').mkdir()
    (tmp_path / 'probe' / 'state.json').write_text(
        json.dumps({'job_id': 'j6', 'status': 'running', 'train_job_id': 't1'})
    )
    heartbeat = tmp_path / 'probe' / 'heartbeat'
    heartbeat.touch()
    old = time.time() - 120
    os.utime(heartbeat, (old, old))

    assert probe_job.reconcile_orphaned_jobs() is True
    on_disk = json.loads((tmp_path / 'probe' / 'state.json').read_text())
    assert on_disk['status'] == 'interrupted'


def test_probe_job_leaves_fresh_running_state_alone(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_PROBE_JOBS_DIR', str(tmp_path / 'probe'))
    from src.services.curation import probe_job

    (tmp_path / 'probe').mkdir()
    (tmp_path / 'probe' / 'state.json').write_text(
        json.dumps({'job_id': 'j6', 'status': 'running'})
    )
    (tmp_path / 'probe' / 'heartbeat').touch()

    assert probe_job.reconcile_orphaned_jobs() is False
    on_disk = json.loads((tmp_path / 'probe' / 'state.json').read_text())
    assert on_disk['status'] == 'running'


# =============================================================================
# embedding_viz.reconcile_orphaned_jobs
# =============================================================================


def test_embedding_viz_reconciles_orphaned_running_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_JOBS_DIR', str(tmp_path / 'viz'))
    from src.services.curation import embedding_viz

    (tmp_path / 'viz').mkdir()
    (tmp_path / 'viz' / 'state.json').write_text(
        json.dumps({'job_id': 'j3', 'status': 'running', 'scope': 'residual'})
    )
    heartbeat = tmp_path / 'viz' / 'heartbeat'
    heartbeat.touch()
    old = time.time() - 120
    os.utime(heartbeat, (old, old))

    assert embedding_viz.reconcile_orphaned_jobs() is True
    on_disk = json.loads((tmp_path / 'viz' / 'state.json').read_text())
    assert on_disk['status'] == 'interrupted'


def test_embedding_viz_leaves_fresh_running_state_alone(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_VIZ_JOBS_DIR', str(tmp_path / 'viz'))
    from src.services.curation import embedding_viz

    (tmp_path / 'viz').mkdir()
    (tmp_path / 'viz' / 'state.json').write_text(json.dumps({'job_id': 'j3', 'status': 'running'}))
    (tmp_path / 'viz' / 'heartbeat').touch()

    assert embedding_viz.reconcile_orphaned_jobs() is False
    on_disk = json.loads((tmp_path / 'viz' / 'state.json').read_text())
    assert on_disk['status'] == 'running'


# =============================================================================
# autolabel.job.reconcile_orphaned_jobs — different shape (separate worker
# container, pending-trigger file)
# =============================================================================


def test_autolabel_job_reconciles_orphaned_running_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(tmp_path / 'auto_label'))
    import importlib

    from src.services.curation.autolabel import job

    importlib.reload(job)

    job._STATE_DIR.mkdir(parents=True, exist_ok=True)
    job._STATE_FILE.write_text(json.dumps({'job_id': 'j4', 'status': 'running', 'stage': 'vlm'}))
    job._HEARTBEAT_FILE.touch()
    old = time.time() - 120
    os.utime(job._HEARTBEAT_FILE, (old, old))

    assert job.reconcile_orphaned_jobs() is True
    on_disk = json.loads(job._STATE_FILE.read_text())
    assert on_disk['status'] == 'interrupted'


def test_autolabel_job_leaves_fresh_running_state_alone(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(tmp_path / 'auto_label'))
    import importlib

    from src.services.curation.autolabel import job

    importlib.reload(job)

    job._STATE_DIR.mkdir(parents=True, exist_ok=True)
    job._STATE_FILE.write_text(json.dumps({'job_id': 'j4', 'status': 'running'}))
    job._HEARTBEAT_FILE.touch()

    assert job.reconcile_orphaned_jobs() is False
    assert json.loads(job._STATE_FILE.read_text())['status'] == 'running'


def test_autolabel_job_ignores_pending_unclaimed_trigger(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued trigger the worker hasn't picked up yet is not an
    orphaned run -- the worker container has its own independent
    lifecycle and may simply not have gotten to it."""
    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(tmp_path / 'auto_label'))
    import importlib

    from src.services.curation.autolabel import job

    importlib.reload(job)

    job._STATE_DIR.mkdir(parents=True, exist_ok=True)
    job._STATE_FILE.write_text(json.dumps({'job_id': 'j5', 'status': 'queued'}))
    job._TRIGGER_FILE.write_text(json.dumps({'job_id': 'j5'}))

    assert job.reconcile_orphaned_jobs() is False
    assert json.loads(job._STATE_FILE.read_text())['status'] == 'queued'
