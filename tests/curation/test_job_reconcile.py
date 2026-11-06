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
