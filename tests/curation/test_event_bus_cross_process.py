"""S-3: file-backed cross-process event bus (src/services/curation/event_hub.py).

Simulates "two uvicorn workers" as two independent ``EventHub`` instances
sharing one on-disk log file — the same relationship that holds between
real uvicorn worker *processes*, minus actually forking. Each instance
gets its own ``_EventLog`` pointed at the same path (constructed
directly, bypassing ``CurationConfig.state_dir`` resolution) so this
exercises the exact write/tail code path real processes use, without a
multiprocess test's flakiness.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import TYPE_CHECKING

import pytest

from src.services.curation.event_hub import EventHub, _EventLog


if TYPE_CHECKING:
    from pathlib import Path


def _make_hub(log_path: Path, max_bytes: int | None = None) -> EventHub:
    """Build an EventHub pointed at ``log_path`` without touching the
    real (likely unwritable in test envs) CurationConfig.state_dir."""
    hub = EventHub.__new__(EventHub)
    hub._subscribers = set()
    hub._lock = asyncio.Lock()
    hub._published = 0
    hub._dropped = 0
    hub._bus = 'file'
    hub._log = _EventLog(path=log_path, max_bytes=max_bytes)
    hub._tail_task = None
    return hub


@pytest.mark.asyncio
async def test_second_process_sees_first_processs_publish(tmp_path: Path) -> None:
    """Two EventHub instances (standing in for two uvicorn workers) share
    one log file. A subscriber on B must receive an event published on A."""
    log_path = tmp_path / 'events.jsonl'
    hub_a = _make_hub(log_path)
    hub_b = _make_hub(log_path)
    try:
        await hub_a.start_tail()
        await hub_b.start_tail()
        sub_b = await hub_b.subscribe()

        hub_a.publish({'type': 'crop.created', 'topic': 'crop', 'crop_id': 'x1'})

        got = await asyncio.wait_for(sub_b.queue.get(), timeout=2.0)
        assert got['crop_id'] == 'x1'
    finally:
        await hub_a.stop_tail()
        await hub_b.stop_tail()


@pytest.mark.asyncio
async def test_publisher_process_does_not_receive_its_own_event_twice(tmp_path: Path) -> None:
    """publish() must not dispatch directly on the file bus — only the
    tailer delivers, exactly once per process."""
    log_path = tmp_path / 'events.jsonl'
    hub_a = _make_hub(log_path)
    try:
        await hub_a.start_tail()
        sub_a = await hub_a.subscribe()

        hub_a.publish({'type': 'crop.created', 'topic': 'crop', 'crop_id': 'x2'})

        first = await asyncio.wait_for(sub_a.queue.get(), timeout=2.0)
        assert first['crop_id'] == 'x2'
        # No second delivery of the same event.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(sub_a.queue.get(), timeout=0.5)
    finally:
        await hub_a.stop_tail()


@pytest.mark.asyncio
async def test_rotation_delivers_every_event_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a tiny max_bytes, 50 published events force several rotations.
    Every event must still arrive, in order, at a subscriber tailing from
    the start.

    The single `.1` backup slot can only lose data if a *second* rotation
    clobbers it before any tailer has drained the first — a real
    deployment's 8 MiB default and a 0.25s poll make that astronomically
    unlikely, but this test's 1024-byte max_bytes forces rotations far
    faster than production. Tighten the poll interval to match so the
    test proves "every tailer keeps up with rotation," not "the poll
    interval happens to be fast enough for this file size."
    """
    import src.services.curation.event_hub as event_hub_mod

    monkeypatch.setattr(event_hub_mod, '_TAIL_POLL_S', 0.001)
    log_path = tmp_path / 'events.jsonl'
    hub_a = _make_hub(log_path, max_bytes=1024)
    hub_b = _make_hub(log_path, max_bytes=1024)
    try:
        await hub_a.start_tail()
        await hub_b.start_tail()
        sub_b = await hub_b.subscribe()

        # Yield to the event loop between publishes (as real callers would —
        # each publish() comes from a separate request/task, never a tight
        # zero-await loop) so the tailer's poll gets a chance to drain each
        # generation before the next rotation clobbers the single `.1` slot.
        n = 50
        for i in range(n):
            hub_a.publish({'type': 'crop.created', 'topic': 'crop', 'crop_id': f'c{i}'})
            await asyncio.sleep(0.01)

        received: list[str] = []

        async def _collect() -> None:
            while len(received) < n:
                ev = await sub_b.queue.get()
                received.append(ev['crop_id'])

        await asyncio.wait_for(_collect(), timeout=5.0)
        assert received == [f'c{i}' for i in range(n)]
    finally:
        await hub_a.stop_tail()
        await hub_b.stop_tail()


def test_line_over_max_size_is_dropped_not_written(tmp_path: Path) -> None:
    log = _EventLog(path=tmp_path / 'events.jsonl')
    huge_event = {'type': 'crop.created', 'topic': 'crop', 'crop_id': 'x' * 8192}
    log._append_sync(huge_event)
    assert log.path.read_text() == ''


def test_stats_report_file_bus_and_log_path(tmp_path: Path) -> None:
    hub = _make_hub(tmp_path / 'events.jsonl')
    stats = hub.stats()
    assert stats['bus'] == 'file'
    assert stats['log_path'] == str(tmp_path / 'events.jsonl')
    assert stats['subscribers'] == 0
    assert stats['events_published'] == 0


def test_unwritable_state_dir_falls_back_to_process_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only parent directory can't hold the events/ subdir —
    EventHub must degrade to in-process delivery rather than crash."""
    from src.config.curation import CurationConfig

    ro_dir = tmp_path / 'ro'
    ro_dir.mkdir(mode=0o555)
    cfg = CurationConfig(state_dir=ro_dir)
    monkeypatch.setattr('src.services.curation.event_hub.get_curation_config', lambda: cfg)
    try:
        hub = EventHub()
        assert hub._bus == 'process'
        assert hub.stats()['bus'] == 'process'
        assert hub.stats()['log_path'] is None
    finally:
        ro_dir.chmod(0o755)


@pytest.fixture
def _reload_bulk_writer_after_env_restored():
    """Reload ``bulk_writer`` once more after ``monkeypatch`` reverts the
    env it patched, so the module-level ``_EVENT_API_URL`` constant
    doesn't leak a test value into later tests importing this module.

    Fixture teardown order is LIFO: this fixture must be requested
    *before* ``monkeypatch`` in the test signature so its teardown runs
    *after* monkeypatch's (env already restored by then).
    """
    yield
    from scripts.curation.worker import bulk_writer

    importlib.reload(bulk_writer)


@pytest.mark.usefixtures('_reload_bulk_writer_after_env_restored')
def test_bulk_writer_event_api_url_falls_back_to_api_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OP_EVENT_API_URL is the explicit override; with only OP_API_BASE_URL
    set (the trainer/campaign convention for "where yolo-api lives"), the
    detection worker's publish path should still resolve a URL instead of
    silently going dark."""
    monkeypatch.delenv('OP_EVENT_API_URL', raising=False)
    monkeypatch.setenv('OP_API_BASE_URL', 'http://yolo-api:8000/')
    monkeypatch.delenv('OP_API', raising=False)

    from scripts.curation.worker import bulk_writer

    importlib.reload(bulk_writer)
    assert bulk_writer._EVENT_API_URL == 'http://yolo-api:8000'


@pytest.mark.usefixtures('_reload_bulk_writer_after_env_restored')
def test_bulk_writer_event_api_url_prefers_explicit_over_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_EVENT_API_URL', 'http://explicit:8000')
    monkeypatch.setenv('OP_API_BASE_URL', 'http://fallback:8000')

    from scripts.curation.worker import bulk_writer

    importlib.reload(bulk_writer)
    assert bulk_writer._EVENT_API_URL == 'http://explicit:8000'
