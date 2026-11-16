"""The live suite's heal and guard logic, offline (tests/live/_heal.py)."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from tests.live import _heal


def _row(state: str = 'running', health: str = 'healthy', code: int = 0) -> dict[str, Any]:
    return {'State': state, 'Health': health, 'ExitCode': code}


def _healthy() -> dict[str, dict[str, Any]]:
    return {s: _row() for s in _heal.STACK_SERVICES}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _timeout(*_: Any, **__: Any) -> None:
    raise subprocess.TimeoutExpired('docker compose', 120)


def test_every_heal_step_runs_when_the_first_one_times_out() -> None:
    calls: list[tuple[str, ...]] = []

    def run(*args: str, **_: Any) -> None:
        calls.append(args)
        if args[0] == 'unpause' and args[1] == 'opensearch':
            _timeout()

    clock = Clock()
    with pytest.raises(_heal.HealError) as err:
        _heal.heal_stack(
            unpause=('opensearch', 'jobs-worker'),
            run=run,
            states=_healthy,
            clock=clock,
            sleep=clock.sleep,
        )
    verbs = [c[0] for c in calls]
    assert verbs == ['unpause', 'unpause', 'up']
    assert 'unpause opensearch: TimeoutExpired' in str(err.value)


def test_the_error_names_every_failed_step() -> None:
    def run(*_: str, **__: Any) -> None:
        _timeout()

    def states() -> dict[str, dict[str, Any]]:
        raise subprocess.TimeoutExpired('docker compose ps', 120)

    with pytest.raises(_heal.HealError) as err:
        _heal.heal_stack(unpause=('fake-vlm',), run=run, states=states)
    text = str(err.value)
    assert 'unpause fake-vlm' in text
    assert 'up api jobs-worker' in text
    assert 'ps: TimeoutExpired' in text


def test_the_guard_exits_with_code_4_and_the_service_table(monkeypatch: pytest.MonkeyPatch) -> None:
    # The offline suite itself runs under xdist; the guard's serial path is the one under test.
    monkeypatch.delenv('PYTEST_XDIST_WORKER', raising=False)
    down = _healthy()
    down['jobs-worker'] = _row('exited', '', 137)
    clock = Clock()
    with pytest.raises(pytest.exit.Exception) as exit_info:
        _heal.require_healthy_stack(
            run=lambda *a, **k: None,
            states=lambda: down,
            budget_s=30,
            clock=clock,
            sleep=clock.sleep,
            where='after test_x',
        )
    assert exit_info.value.returncode == 4
    assert 'jobs-worker: State=exited Health=- ExitCode=137' in str(exit_info.value)
    assert 'after test_x' in str(exit_info.value)


def test_in_an_xdist_worker_the_guard_fails_the_test_instead_of_ending_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw0')
    down = _healthy()
    down['jobs-worker'] = _row('exited', '', 137)
    clock = Clock()
    with pytest.raises(pytest.fail.Exception) as failed:
        _heal.require_healthy_stack(
            run=lambda *a, **k: None,
            states=lambda: down,
            budget_s=30,
            clock=clock,
            sleep=clock.sleep,
            where='after test_x',
        )
    assert 'jobs-worker: State=exited Health=- ExitCode=137' in str(failed.value)


def test_the_guard_is_silent_when_the_stack_is_healthy() -> None:
    def run(*_: str, **__: Any) -> None:
        raise AssertionError('a healthy stack must not be touched')

    _heal.require_healthy_stack(run=run, states=_healthy)


def test_the_guard_heals_once_and_passes_when_the_stack_comes_back() -> None:
    state = {'up': False}

    def run(*args: str, **_: Any) -> None:
        if args[0] == 'up':
            state['up'] = True

    def states() -> dict[str, dict[str, Any]]:
        return (
            _healthy()
            if state['up']
            else {s: _row('exited', '', 137) for s in _heal.STACK_SERVICES}
        )

    clock = Clock()
    _heal.require_healthy_stack(run=run, states=states, clock=clock, sleep=clock.sleep)
