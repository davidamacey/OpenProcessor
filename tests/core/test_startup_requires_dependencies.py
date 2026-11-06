"""The API refuses to start when the database, queue, OpenSearch or Triton is unreachable.

The probes are injected (the conftest binds an all-ready gate for every other test), and the
clock is faked, so a 60 s wait costs nothing.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from src.app import lifespan as lifespan_module, startup_gate
from src.config import get_settings
from src.db.schema_check import (
    Clock,
    DependencyUnavailableError,
    ProbeFailed,
    SchemaNotCurrentError,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.slept += seconds

    def monotonic(self) -> float:
        return self.now

    def clock(self) -> Clock:
        return Clock(sleep=self.sleep, monotonic=self.monotonic)


def _up() -> None:
    return None


def _down() -> None:
    raise ProbeFailed('ConnectError')


def _probes(down: str | None) -> dict:
    return {n: (_down if n == down else _up) for n in ('database', 'queue', 'opensearch', 'triton')}


@pytest.mark.parametrize('down', ['database', 'queue', 'opensearch', 'triton'])
def test_each_unreachable_dependency_refuses_the_start_naming_it(
    monkeypatch: pytest.MonkeyPatch, down: str
) -> None:
    monkeypatch.setattr(startup_gate, 'startup_probes', lambda _s: _probes(down))
    fake = FakeClock()
    with pytest.raises(
        DependencyUnavailableError, match=f'dependency_unavailable: {down}'
    ) as caught:
        asyncio.run(startup_gate.require_control_plane(get_settings(), fake.clock()))
    assert caught.value.names == [down]
    assert 'ConnectError' in str(caught.value)
    assert fake.slept >= 60  # the whole budget was spent before refusing


def test_several_unreachable_dependencies_are_all_named(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = {'database': _down, 'queue': _down, 'opensearch': _up, 'triton': _up}
    monkeypatch.setattr(startup_gate, 'startup_probes', lambda _s: probes)
    with pytest.raises(DependencyUnavailableError, match='database, queue'):
        asyncio.run(startup_gate.require_control_plane(get_settings(), FakeClock().clock()))


def test_a_dependency_that_comes_up_during_the_wait_lets_the_start_proceed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {'n': 0}

    def flaky() -> None:
        calls['n'] += 1
        if calls['n'] < 4:
            raise ProbeFailed('starting')

    monkeypatch.setattr(
        startup_gate, 'startup_probes', lambda _s: {**_probes(None), 'queue': flaky}
    )
    seen: list[str] = []
    monkeypatch.setattr(startup_gate, 'check_control_plane', lambda: seen.append('checked'))
    asyncio.run(startup_gate.require_control_plane(get_settings(), FakeClock().clock()))
    assert calls['n'] == 4
    assert seen == ['checked']


def test_the_schema_is_checked_only_after_every_dependency_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(startup_gate, 'startup_probes', lambda _s: _probes('database'))
    monkeypatch.setattr(
        startup_gate, 'check_control_plane', lambda: pytest.fail('checked with the database down')
    )
    with pytest.raises(DependencyUnavailableError):
        asyncio.run(startup_gate.require_control_plane(get_settings(), FakeClock().clock()))


def test_a_stale_schema_refuses_the_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(startup_gate, 'startup_probes', lambda _s: _probes(None))

    def stale() -> None:
        raise SchemaNotCurrentError('0000_old', '0001_control_plane')

    monkeypatch.setattr(startup_gate, 'check_control_plane', stale)
    with pytest.raises(SchemaNotCurrentError, match=r'0000_old.*0001_control_plane'):
        asyncio.run(startup_gate.require_control_plane(get_settings(), FakeClock().clock()))


def test_the_lifespan_refuses_before_any_other_startup_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def refuse(_settings: object, _clock: object = None) -> None:
        raise DependencyUnavailableError(['database'], 'refused')

    def started(*_a: object, **_k: object) -> None:
        pytest.fail('a startup step ran although a dependency is down')

    monkeypatch.setattr(lifespan_module, '_preflight', lambda: None)
    monkeypatch.setattr(startup_gate, 'require_control_plane', refuse)
    monkeypatch.setattr(lifespan_module, '_start_pools', started)

    async def run() -> None:
        async with lifespan_module.lifespan(FastAPI()):
            pytest.fail('the application started')

    with pytest.raises(DependencyUnavailableError, match='dependency_unavailable: database'):
        asyncio.run(run())


def test_the_queue_probe_names_the_missing_client_instead_of_passing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: a probe that cannot run is not a probe that passed."""
    import importlib

    def no_module(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, 'import_module', no_module)
    with pytest.raises(ModuleNotFoundError):
        startup_gate._queue_probe()  # the probe is private to the gate
