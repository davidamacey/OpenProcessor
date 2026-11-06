"""The pool healthcheck, fed the JSON shapes ``celery inspect stats`` returns.

A wedged prefork pool still answers ``inspect ping``; only ``inspect stats`` shows the pool.
The respawn storm is the case a non-empty-pids check calls healthy.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

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


def _reply(monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(stdout=stdout, stderr='', returncode=returncode)

    monkeypatch.setattr(check.subprocess, 'run', fake_run)
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
    [('', 0), ('{}', 0), ('not json', 0), ('{"a": {"pool": "x"}}', 0), ('{}', 1)],
)
def test_an_unreadable_or_missing_reply_fails_closed(
    monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int
) -> None:
    _reply(monkeypatch, stdout, returncode)
    assert check.check_pool_has_live_processes(NODE)[0] is False


def test_a_timeout_is_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: object, **__: object) -> None:
        raise subprocess.TimeoutExpired('celery', 8)

    monkeypatch.setattr(check.subprocess, 'run', boom)
    assert check.check_pool_has_live_processes(NODE)[0] is False


def test_the_stats_request_names_only_this_node(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _reply(monkeypatch, _stats([1]))
    assert check.main(['x', 'jobs']) == 0
    argv = calls[0]
    assert argv[argv.index('-d') + 1] == f'jobs@{socket.gethostname()}'
    assert argv[argv.index('-A') + 1] == 'src.tasks.app'


def test_a_full_node_name_is_kept() -> None:
    assert check.node_name('periodic@box') == 'periodic@box'


def test_main_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert check.main(['x']) == 2
    _reply(monkeypatch, _stats([]))
    assert check.main(['x', NODE]) == 1
