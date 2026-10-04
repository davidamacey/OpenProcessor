"""Startup repair of pre-0.4.0 projects runs once per project, throttled and
retried, and also registers the active profile's region class."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation.job_lock import exclusive_start_lock
from src.services.projects import legacy_repair


_REAL_ENSURE = legacy_repair._ensure_active_region_class


class _BusyError(Exception):
    status_code = 429


class _Client:
    def __init__(self, legacy: int = 5, failures: int = 0) -> None:
        self.legacy = legacy
        self.failures = failures
        self.updates = 0

    async def count(self, **_kw: Any) -> dict[str, Any]:
        return {'count': self.legacy}

    async def update_by_query(self, **_kw: Any) -> dict[str, Any]:
        if self.failures:
            self.failures -= 1
            raise _BusyError('rejected_execution_exception')
        self.updates += 1
        done, self.legacy = self.legacy, 0
        return {'updated': done}


@pytest.fixture(autouse=True)
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def _sleep(s: float) -> None:
        waits.append(s)

    monkeypatch.setattr(legacy_repair.asyncio, 'sleep', _sleep)

    async def _region(_client: Any) -> int:
        return 3

    monkeypatch.setattr(legacy_repair, '_ensure_active_region_class', _region)
    return waits


@pytest.mark.asyncio
async def test_a_second_process_skips_while_the_first_holds_the_lock(tmp_path) -> None:
    lock = tmp_path / 'legacy_repair.lock'
    client = _Client()
    with exclusive_start_lock(lock) as mine:
        assert mine
        await legacy_repair.repair_bound_project(client, 'p', lock, 'items')
    assert client.updates == 0
    await legacy_repair.repair_bound_project(client, 'p', lock, 'items')
    assert client.updates == 1


@pytest.mark.asyncio
async def test_rerun_with_nothing_legacy_starts_no_task(tmp_path) -> None:
    client = _Client(legacy=0)
    await legacy_repair.repair_bound_project(client, 'p', tmp_path / 'l', 'items')
    assert client.updates == 0


@pytest.mark.asyncio
async def test_429_is_retried_with_backoff_not_dropped(tmp_path, slept: list[float]) -> None:
    client = _Client(failures=2)
    await legacy_repair.repair_bound_project(client, 'p', tmp_path / 'l', 'items')
    assert client.updates == 1
    assert slept == [2.0, 4.0]


@pytest.mark.asyncio
async def test_a_persistent_failure_is_logged_not_raised(tmp_path) -> None:
    client = _Client(failures=99)
    await legacy_repair.repair_bound_project(client, 'p', tmp_path / 'l', 'items')
    assert client.updates == 0
    assert client.legacy == 5


@pytest.mark.asyncio
async def test_a_non_429_error_is_not_retried(tmp_path, slept: list[float]) -> None:
    class _Broken(_Client):
        async def update_by_query(self, **_kw: Any) -> dict[str, Any]:
            raise ValueError('bad request')

    await legacy_repair.repair_bound_project(_Broken(), 'p', tmp_path / 'l', 'items')
    assert slept == []


@pytest.mark.asyncio
async def test_region_class_failure_does_not_stop_the_backfill(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(_client: Any) -> int:
        raise RuntimeError('registry unreadable')

    monkeypatch.setattr(legacy_repair, '_ensure_active_region_class', _boom)
    client = _Client()
    await legacy_repair.repair_bound_project(client, 'p', tmp_path / 'l', 'items')
    assert client.updates == 1


@pytest.mark.asyncio
async def test_active_region_class_is_registered_after_the_snapshot_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The profile is read from the config snapshot, which is empty until refreshed."""
    order: list[str] = []

    class _Store:
        async def refresh(self, _client: Any) -> None:
            order.append('refresh')

    monkeypatch.setattr('src.services.config_store.get_config_store', lambda: _Store())

    def _ensure() -> int:
        order.append('ensure')
        return 7

    monkeypatch.setattr('src.services.curation.region_class.ensure_region_class', _ensure)
    assert await _REAL_ENSURE(object()) == 7
    assert order == ['refresh', 'ensure']


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def info(self, event: str, **_kw: Any) -> None:
        self.calls.append(('info', event))

    def warning(self, event: str, **_kw: Any) -> None:
        self.calls.append(('warning', event))

    def debug(self, event: str, **_kw: Any) -> None:
        self.calls.append(('debug', event))


@pytest.mark.asyncio
async def test_a_repair_that_changed_nothing_is_not_logged_at_info(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _Recorder()
    monkeypatch.setattr(legacy_repair, 'logger', rec)
    await legacy_repair.repair_bound_project(_Client(legacy=0), 'p', tmp_path / 'l', 'items')
    assert ('info', 'legacy_project_repair') not in rec.calls


@pytest.mark.asyncio
async def test_a_repair_that_backfilled_rows_is_logged_at_info(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _Recorder()
    monkeypatch.setattr(legacy_repair, 'logger', rec)
    await legacy_repair.repair_bound_project(_Client(legacy=5), 'p', tmp_path / 'l', 'items')
    assert ('info', 'legacy_project_repair') in rec.calls
