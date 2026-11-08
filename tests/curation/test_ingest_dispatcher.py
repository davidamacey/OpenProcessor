"""The dispatcher's decisions (pure, injected clock). The pass over real sources is
``ingest_db/test_ingest_dispatch_once.py``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.services.curation.bulk_ingest.dispatcher import Action, backoff_s, decide
from src.services.curation.bulk_ingest.sources import IngestSource, ScheduleTrigger, WatchTrigger


NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC).timestamp()
IDLE = {'pending': 0, 'done': 0, 'failed': 0, 'dead': 0}
PENDING = {**IDLE, 'pending': 4}


def _source(**over: Any) -> IngestSource:
    base: dict[str, Any] = {
        'source_id': 'src_testsource',
        'name': 't',
        'backend': 's3',
        'bucket': 'b',
    }
    return IngestSource(**{**base, **over})


def test_a_source_without_triggers_is_never_acted_on() -> None:
    assert decide(_source(), {}, PENDING, None, now=NOW) is None


def test_a_watcher_polls_on_its_interval_and_not_before() -> None:
    src = _source(watch=WatchTrigger(True, 60))
    assert decide(src, {}, IDLE, None, now=NOW) == Action('poll', src.source_id)
    assert decide(src, {'last_poll_started_at': NOW - 30}, IDLE, None, now=NOW) is None
    assert decide(src, {'last_poll_started_at': NOW - 61}, IDLE, None, now=NOW) == Action(
        'poll', src.source_id
    )


def test_the_schedule_fires_once_it_is_due_and_a_down_period_makes_one_fire() -> None:
    src = _source(schedule=ScheduleTrigger(True, '0 * * * *'))
    just_fired = datetime.fromtimestamp(NOW, UTC).isoformat()
    assert decide(src, {'last_fire_at': just_fired}, IDLE, None, now=NOW) is None
    long_ago = datetime.fromtimestamp(NOW - 5 * 86400, UTC).isoformat()
    assert decide(src, {'last_fire_at': long_ago}, IDLE, None, now=NOW) == Action(
        'run_scan', src.source_id
    )


def test_pending_rows_are_drained_only_for_a_source_with_a_trigger() -> None:
    hook = _source(webhook_enabled=True)
    assert decide(hook, {}, PENDING, None, now=NOW) == Action('drain', hook.source_id)
    assert decide(hook, {}, IDLE, None, now=NOW) is None
    assert (
        decide(_source(), {}, PENDING, None, now=NOW) is None
    )  # a one-shot source waits for a person


def test_a_cancelled_run_is_not_restarted_and_a_failed_one_waits() -> None:
    hook = _source(webhook_enabled=True)
    cancelled = {'status': 'cancelled', 'finished_at': None}
    assert decide(hook, {}, PENDING, cancelled, now=NOW) is None
    finished = datetime.fromtimestamp(NOW - 60, UTC).isoformat()
    assert decide(hook, {}, PENDING, {'status': 'failed', 'finished_at': finished}, now=NOW) is None
    long_ago = datetime.fromtimestamp(NOW - 3600, UTC).isoformat()
    assert decide(
        hook, {}, PENDING, {'status': 'failed', 'finished_at': long_ago}, now=NOW
    ) == Action('drain', hook.source_id)


def test_nothing_happens_during_backoff() -> None:
    src = _source(watch=WatchTrigger(True, 10), webhook_enabled=True)
    state = {'backoff_until': NOW + 30, 'last_poll_started_at': NOW - 999}
    assert decide(src, state, PENDING, None, now=NOW) is None
    assert decide(src, state, PENDING, None, now=NOW + 31) is not None


def test_the_backoff_doubles_from_ten_seconds_to_ten_minutes() -> None:
    assert [backoff_s(n) for n in (1, 2, 3, 4, 7, 20)] == [10.0, 20.0, 40.0, 80.0, 600.0, 600.0]
