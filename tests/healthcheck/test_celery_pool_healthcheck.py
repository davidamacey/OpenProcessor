"""The pool healthcheck, fed the replies of Celery's ``inspect().stats()``.

A wedged prefork pool still answers ``inspect ping``; only ``inspect stats`` shows the pool.
The respawn storm is the case a non-empty-pids check calls healthy.
"""

from __future__ import annotations

import importlib.util
import json
import socket
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[2] / 'scripts' / 'healthcheck' / 'celery_pool_healthcheck.py'
)
_spec = importlib.util.spec_from_file_location('celery_pool_healthcheck', _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

NODE = 'jobs@host1'


def _stats(processes: list[int] | None, total: dict | None = None, impl: str = 'prefork') -> str:
    pool: dict = {'implementation': f'celery.concurrency.{impl}:TaskPool'}
    if processes is not None:
        pool['processes'] = processes
    return json.dumps({NODE: {'pool': pool, 'total': total or {}}})


@pytest.fixture(autouse=True)
def _isolated_samples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(check.tempfile, 'gettempdir', lambda: str(tmp_path))


class _FakeApp:
    def __init__(self, reply: object, calls: list[dict]) -> None:
        self._reply = reply
        self._calls = calls
        self.control = self

    def connection_for_read(self) -> _FakeApp:
        return self

    def __enter__(self) -> _FakeApp:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def ensure_connection(self, **kwargs: object) -> None:
        self._calls.append({'ensure_connection': kwargs})

    def inspect(self, **kwargs: object) -> _FakeApp:
        self._calls.append(kwargs)
        return self

    def stats(self) -> object:
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    def close(self) -> None:
        return None


def _reply(monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0) -> list[dict]:
    """Serve ``stdout`` (the JSON of ``inspect stats``) through the in-process inspect call."""
    calls: list[dict] = []
    reply = json.loads(stdout) if stdout and returncode == 0 else None
    return _serve(monkeypatch, reply, calls)


def _serve(monkeypatch: pytest.MonkeyPatch, reply: object, calls: list[dict]) -> list[dict]:
    from src.tasks.runtime import gpu_app

    def build(include: object) -> _FakeApp:
        calls.append({'include': list(include)})  # type: ignore[call-overload]
        return _FakeApp(reply, calls)

    monkeypatch.setattr(gpu_app, 'build_app', build)
    return calls


def test_a_steady_pool_is_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    _reply(monkeypatch, _stats([11, 12], {'op.infra.noop': 3}))
    assert check.check_pool_has_live_processes(NODE)[0] is True
    assert check.check_pool_has_live_processes(NODE)[0] is True


def test_a_pool_with_no_children_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    _reply(monkeypatch, _stats([]))
    healthy, reason = check.check_pool_has_live_processes(NODE)
    assert healthy is False
    assert 'no live child processes' in reason


def test_a_fully_replaced_pool_with_a_frozen_counter_is_a_respawn_storm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reply(monkeypatch, _stats([1, 2], {}))
    assert check.check_pool_has_live_processes(NODE)[0] is True
    _reply(monkeypatch, _stats([3, 4], {}))
    healthy, reason = check.check_pool_has_live_processes(NODE)
    assert healthy is False
    assert 'respawn storm' in reason


def test_recycling_that_accepted_tasks_is_not_a_storm(monkeypatch: pytest.MonkeyPatch) -> None:
    _reply(monkeypatch, _stats([1, 2], {'t': 5}))
    check.check_pool_has_live_processes(NODE)
    _reply(monkeypatch, _stats([3, 4], {'t': 9}))
    assert check.check_pool_has_live_processes(NODE)[0] is True


def test_a_stale_sample_is_not_judged(monkeypatch: pytest.MonkeyPatch) -> None:
    _reply(monkeypatch, _stats([1, 2], {}))
    check.check_pool_has_live_processes(NODE)
    monkeypatch.setattr(check.time, 'time', lambda: 10**10)
    _reply(monkeypatch, _stats([3, 4], {}))
    assert check.check_pool_has_live_processes(NODE)[0] is True


def test_a_non_prefork_pool_is_healthy_when_it_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    _reply(monkeypatch, _stats(None, impl='threads'))
    assert check.check_pool_has_live_processes(NODE)[0] is True


@pytest.mark.parametrize(
    ('stdout', 'returncode'),
    [('', 0), ('{}', 0), ('{"a": {"pool": "x"}}', 0), ('{}', 1)],
)
def test_an_unreadable_or_missing_reply_fails_closed(
    monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int
) -> None:
    _reply(monkeypatch, stdout, returncode)
    assert check.check_pool_has_live_processes(NODE)[0] is False


def test_a_broker_failure_is_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, ConnectionError('broker down'), [])
    healthy, reason = check.check_pool_has_live_processes(NODE)
    assert healthy is False
    assert 'broker down' in reason


def test_the_probe_asks_only_this_node_and_loads_no_task_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _reply(monkeypatch, _stats([1]))
    assert check.main(['x', 'jobs']) == 0
    assert {'include': []} in calls
    asked = next(c for c in calls if 'destination' in c)
    assert asked['destination'] == [f'jobs@{socket.gethostname()}']


def test_the_broker_connect_is_one_bounded_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _reply(monkeypatch, _stats([1]))
    check.check_pool_has_live_processes(NODE)
    attempt = next(c for c in calls if 'ensure_connection' in c)['ensure_connection']
    assert attempt == {'max_retries': 1, 'timeout': check.CONNECT_TIMEOUT_S}


def test_the_whole_probe_has_a_hard_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    alarms: list[int] = []
    monkeypatch.setattr(check.signal, 'alarm', alarms.append)
    _reply(monkeypatch, _stats([1]))
    check.main(['x', 'jobs'])
    assert alarms[0] == check.PROBE_DEADLINE_S < 15


def test_the_probe_starts_no_subprocess() -> None:
    assert not hasattr(check, 'subprocess')


def test_a_full_node_name_is_kept() -> None:
    assert check.node_name('periodic@box') == 'periodic@box'


def test_main_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert check.main(['x']) == 2
