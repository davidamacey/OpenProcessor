"""Unit tests for the cross-process singleton-start lock
(src.services.curation.job_lock), shared by item_scores.job and
probe_job's start_job/start_probe_job.
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from src.services.curation.job_lock import exclusive_start_lock


def test_lock_is_acquired_when_uncontended(tmp_path: Path) -> None:
    with exclusive_start_lock(tmp_path / 'start.lock') as acquired:
        assert acquired is True


def test_lock_is_released_after_the_with_block(tmp_path: Path) -> None:
    lock_file = tmp_path / 'start.lock'
    with exclusive_start_lock(lock_file):
        pass
    with exclusive_start_lock(lock_file) as acquired:
        assert acquired is True


def test_nested_lock_attempt_in_the_same_process_fails(tmp_path: Path) -> None:
    """flock is per-open-file-description, not per-process, so even two
    acquisitions from the very same process (two separate `os.open` calls
    on the same path) must not both succeed -- this is what actually
    protects two coroutines racing `start_job()` within one worker."""
    lock_file = tmp_path / 'start.lock'
    with exclusive_start_lock(lock_file) as outer:
        assert outer is True
        with exclusive_start_lock(lock_file) as inner:
            assert inner is False


def _child_acquire_and_report(lock_file: str, hold_s: float, result_path: str) -> None:
    with exclusive_start_lock(Path(lock_file)) as acquired:
        Path(result_path).write_text('1' if acquired else '0')
        if acquired:
            time.sleep(hold_s)


def test_lock_is_exclusive_across_real_processes(tmp_path: Path) -> None:
    """The actual bug this lock closes: two OS processes (not just two
    coroutines in one process) racing a job start. Spawns a real child
    process that holds the lock briefly; the parent, attempting to
    acquire the same lock file while the child holds it, must see
    ``acquired is False``."""
    lock_file = tmp_path / 'start.lock'
    child_result = tmp_path / 'child_result.txt'
    # 'fork' (Linux default), not 'spawn' -- spawn re-imports the child
    # target by qualified name from a fresh interpreter, which fails to
    # resolve a function defined in a pytest-collected test module.
    ctx = multiprocessing.get_context('fork')
    proc = ctx.Process(
        target=_child_acquire_and_report,
        args=(str(lock_file), 1.0, str(child_result)),
    )
    proc.start()
    # Give the child a generous head start to actually acquire the lock
    # before the parent tries -- flaky-averse, not tuned to the wire.
    deadline = time.time() + 5.0
    while not child_result.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert child_result.exists(), 'child never reported whether it acquired the lock'
    assert child_result.read_text() == '1'

    with exclusive_start_lock(lock_file) as parent_acquired:
        assert parent_acquired is False

    proc.join(timeout=10)
    assert proc.exitcode == 0
