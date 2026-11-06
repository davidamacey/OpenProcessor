"""The job state table is internally consistent."""

from __future__ import annotations

from src.domain import job_states as js


def test_every_transition_names_known_statuses() -> None:
    known = set(js.STATUSES)
    assert all(a in known and b in known for a, b in js.TRANSITIONS)


def test_a_terminal_status_has_no_way_out() -> None:
    assert not [t for t in js.TRANSITIONS if t[0] in js.TERMINAL]


def test_terminal_and_active_do_not_overlap_and_interrupted_is_neither() -> None:
    assert not js.TERMINAL & js.ACTIVE
    assert js.INTERRUPTED not in js.TERMINAL | js.ACTIVE
    assert set(js.STATUSES) == js.TERMINAL | js.ACTIVE | {js.INTERRUPTED}


def test_statuses_are_unique() -> None:
    assert len(set(js.STATUSES)) == len(js.STATUSES)
