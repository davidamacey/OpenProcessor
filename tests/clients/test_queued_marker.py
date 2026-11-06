"""The ``op:queued:<id>`` hint: set with a ttl, cleared, and Unknown when the broker is down."""

from __future__ import annotations

import pytest
import redis

from src.clients import task_queue
from src.clients.task_queue import QueuedState


class _FakeRedis:
    def __init__(self, store: dict[str, tuple[str, int | None]], *, down: bool = False) -> None:
        self.store = store
        self.down = down

    def _check(self) -> None:
        if self.down:
            raise redis.ConnectionError('down')

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self._check()
        self.store[key] = (value, ex)
        return True

    def delete(self, key: str) -> int:
        self._check()
        return int(self.store.pop(key, None) is not None)

    def exists(self, key: str) -> int:
        self._check()
        return int(key in self.store)

    def close(self) -> None: ...


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[str, int | None]]:
    data: dict[str, tuple[str, int | None]] = {}
    monkeypatch.setattr(task_queue, '_client', lambda: _FakeRedis(data))
    return data


def test_mark_sets_the_key_with_the_ttl_and_state_reads_it(store: dict) -> None:
    task_queue.mark_queued('a', ttl_s=86400)
    assert store == {'op:queued:a': ('1', 86400)}
    assert task_queue.queued_state('a') is QueuedState.PRESENT
    assert task_queue.queued_state('b') is QueuedState.ABSENT


def test_clear_removes_only_that_job(store: dict) -> None:
    task_queue.mark_queued('a', ttl_s=60)
    task_queue.mark_queued('b', ttl_s=60)
    task_queue.clear_queued('a')
    assert task_queue.queued_state('a') is QueuedState.ABSENT
    assert task_queue.queued_state('b') is QueuedState.PRESENT


def test_a_connection_error_is_unknown_never_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task_queue, '_client', lambda: _FakeRedis({}, down=True))
    assert task_queue.queued_state('a') is QueuedState.UNKNOWN
    with pytest.raises(task_queue.QueueUnavailable):
        task_queue.mark_queued('a', ttl_s=60)
    with pytest.raises(task_queue.QueueUnavailable):
        task_queue.clear_queued('a')
