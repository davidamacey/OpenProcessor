"""The 1,000-job live check's pytest-timeout must cover its own fixture and call bounds."""

from __future__ import annotations

from tests.live import test_job_load as load


def _timeout_marker() -> int:
    marks = [m for m in load.pytestmark if m.name == 'timeout']
    assert marks, 'the load test has no timeout marker of its own'
    return int(marks[0].args[0])


def test_the_item_timeout_covers_scale_up_run_logs_and_scale_down() -> None:
    needed = (
        2 * load.SCALE_S
        + load.PROJECT_SETUP_S
        + max(120.0, load.COUNT * load.BOUND_S_PER_JOB)
        + load.LOGS_S
    )
    assert _timeout_marker() >= needed
