"""The dispatcher's decisions (pure, injected clock), the tick over real source files,
and the poll pass's failure and backoff handling."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from src.services.curation.bulk_ingest import dispatcher
from src.services.curation.bulk_ingest.dispatcher import Action, decide
from src.services.curation.bulk_ingest.object_queue import QueueEntry
from src.services.curation.bulk_ingest.object_store import ObjectFetchError
from src.services.curation.bulk_ingest.sources import (
    IngestSource,
    ScheduleTrigger,
    WatchTrigger,
    save_source,
    source_queue,
    source_state,
)


if TYPE_CHECKING:
    from pathlib import Path

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


@pytest.fixture(autouse=True)
def bulk_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('OP_BULK_INGEST_DIR', str(tmp_path / 'bulk'))
    return tmp_path


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


def test_the_tick_acts_per_source_once_and_one_run_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for sid in ('src_aaaaaa', 'src_bbbbbb', 'src_cccccc'):
        save_source(_source(source_id=sid, webhook_enabled=True))
        source_queue(sid).enqueue([QueueEntry('k.jpg', 'e')])
    save_source(_source(source_id='src_dddddd', watch=WatchTrigger(True, 60)))
    acted: list[Action] = []
    monkeypatch.setattr(dispatcher, '_execute', lambda action, _src: acted.append(action))
    monkeypatch.setattr(dispatcher, 'active_run', lambda: None)

    taken = dispatcher.run_project_tick(now=NOW)
    # The watcher's scan needs no run slot; of the three drains only the first gets the slot.
    assert [a.kind for a in taken].count('poll') == 1
    assert [a.kind for a in taken].count('drain') == 1
    assert taken == acted


def test_while_a_run_is_live_only_scans_proceed(monkeypatch: pytest.MonkeyPatch) -> None:
    save_source(_source(source_id='src_aaaaaa', webhook_enabled=True))
    source_queue('src_aaaaaa').enqueue([QueueEntry('k.jpg', 'e')])
    save_source(_source(source_id='src_dddddd', watch=WatchTrigger(True, 60)))
    acted: list[Action] = []
    monkeypatch.setattr(dispatcher, '_execute', lambda action, _src: acted.append(action))
    monkeypatch.setattr(dispatcher, 'active_run', lambda: object())
    dispatcher.run_project_tick(now=NOW)
    assert [a.kind for a in acted] == ['poll']


def test_a_schedule_counts_from_the_moment_it_was_first_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_source(_source(schedule=ScheduleTrigger(True, '*/5 * * * *')))
    monkeypatch.setattr(dispatcher, '_execute', lambda *_: None)
    monkeypatch.setattr(dispatcher, 'active_run', lambda: None)
    assert dispatcher.run_project_tick(now=NOW) == []  # not fired on sight
    stamped = source_state('src_testsource').read()['last_fire_at']
    assert datetime.fromisoformat(stamped).timestamp() == NOW
    assert [a.kind for a in dispatcher.run_project_tick(now=NOW + 301)] == ['run_scan']


def test_a_failed_poll_records_the_error_and_backs_off_and_a_good_one_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = _source(watch=WatchTrigger(True, 10))
    save_source(src)
    started: list[bool] = []

    def failing(*_a: Any, **_k: Any) -> None:
        raise ObjectFetchError('listing unavailable')

    monkeypatch.setattr(dispatcher, 'scan_source', failing)
    asyncio.run(dispatcher._poll(src, start_run=lambda _s, scan: started.append(scan)))
    state = source_state(src.source_id).read()
    assert 'listing unavailable' in state['last_poll_error']
    first_backoff = state['backoff_until']
    assert first_backoff > 0 and state['failures'] == 1
    asyncio.run(dispatcher._poll(src, start_run=lambda _s, scan: started.append(scan)))
    assert source_state(src.source_id).read()['backoff_until'] > first_backoff - 1
    assert source_state(src.source_id).read()['failures'] == 2
    assert started == []  # no run is started from a failed scan

    source_queue(src.source_id).enqueue([QueueEntry('new.jpg', 'e')])
    monkeypatch.setattr(dispatcher, 'scan_source', lambda *_a, **_k: None)
    asyncio.run(dispatcher._poll(src, start_run=lambda _s, scan: started.append(scan)))
    state = source_state(src.source_id).read()
    assert (state['failures'], state['last_poll_error'], state['backoff_until']) == (0, None, None)
    assert started == [False]  # pending rows: a drain, without another scan
