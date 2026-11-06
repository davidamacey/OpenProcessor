"""jobs-beat is a singleton: one replica, no fixed container_name (D-54)."""

from __future__ import annotations

from compose._helpers import _services


def test_beat_has_exactly_one_replica_and_no_fixed_container_name() -> None:
    beat = _services()['jobs-beat']
    assert beat['deploy']['replicas'] == 1
    assert 'container_name' not in beat


def test_beat_runs_celery_beat_with_a_tmpfs_schedule() -> None:
    beat = _services()['jobs-beat']
    assert 'celery -A src.tasks.app beat' in beat['command']
    assert '-s /run/beat/celerybeat-schedule' in beat['command']
    assert any(str(t).startswith('/run/beat') for t in beat['tmpfs'])
