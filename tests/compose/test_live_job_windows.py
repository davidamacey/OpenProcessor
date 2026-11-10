"""The live harness proves one-at-a-time singleton jobs from the API's job timestamps.

``jobs.created_at`` and ``jobs.finished_at`` are PostgreSQL ``now()``, the *transaction start*. A
second start that arrives while the first is finishing begins its INSERT transaction before the
first one's finishing UPDATE does, waits on ``uq_jobs_one_active`` until that UPDATE commits, and
so carries a ``created_at`` earlier than the first job's ``finished_at`` although the two never
overlapped (observed: created 42.950727, first finished 42.950782). ``started_at`` is taken by the claim
transaction, which only begins after the INSERT committed, so it orders after the first finish.
"""

from __future__ import annotations

from tests.live._jobs import job_windows_disjoint


# Rows from the failing live run (tests/live/test_one_vs_n.py, config 1/1/1): the second job's
# created_at is 55 microseconds before the first job's finished_at.
FIRST = {
    'created_at': '2026-11-10T15:41:42.429522+00:00',
    'started_at': '2026-11-10T15:41:42.472302+00:00',
    'finished_at': '2026-11-10T15:41:42.950782+00:00',
}
SECOND = {
    'created_at': '2026-11-10T15:41:42.950727+00:00',
    'started_at': '2026-11-10T15:41:42.990149+00:00',
    'finished_at': '2026-11-10T15:41:43.470415+00:00',
}
OVERLAP = {**SECOND, 'started_at': '2026-11-10T15:41:42.700000+00:00'}


def test_a_serial_pair_whose_insert_began_before_the_first_finish_is_disjoint() -> None:
    assert job_windows_disjoint([FIRST, SECOND])


def test_a_second_job_that_started_running_before_the_first_finished_is_not_disjoint() -> None:
    assert not job_windows_disjoint([FIRST, OVERLAP])
