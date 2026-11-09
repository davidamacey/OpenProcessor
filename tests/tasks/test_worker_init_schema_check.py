"""A worker or beat whose database is unreachable or at another head exits instead of running.

Celery's signal dispatch logs an ``Exception`` raised by a receiver and carries on, so the
handlers raise ``SystemExit`` (a ``BaseException``): https://docs.celeryq.dev/en/stable/userguide/signals.html
"""

from __future__ import annotations

import inspect

import pytest

from src.db import schema_check
from src.db.schema_check import DependencyUnavailableError, SchemaNotCurrentError
from src.tasks import app as tasks_app


def _gpu_handlers() -> list:
    # tests/tasks/conftest.py disarms the GPU inventory the app build arms
    from export import export_tasks
    from scripts.curation.bakeoff import bakeoff_tasks

    return [export_tasks._on_worker_init, bakeoff_tasks._on_worker_init]


HANDLERS = [tasks_app._on_worker_init, tasks_app._on_beat_init, *_gpu_handlers()]


@pytest.fixture(autouse=True)
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    for module in (tasks_app, *(inspect.getmodule(h) for h in HANDLERS[2:])):
        monkeypatch.setattr(module, 'reject_retired_env', lambda: None)
    monkeypatch.setattr(
        schema_check, 'wait_for_dependencies', lambda _probes, **_: calls.append('wait')
    )
    monkeypatch.setattr(schema_check, 'check_control_plane', lambda: calls.append('check'))
    return calls


@pytest.mark.parametrize('handler', HANDLERS)
def test_a_current_head_starts(handler, calls: list[str]) -> None:
    handler()
    assert calls == ['wait', 'check']


@pytest.mark.parametrize('handler', HANDLERS)
def test_a_stale_head_exits(handler, monkeypatch: pytest.MonkeyPatch) -> None:
    def stale() -> None:
        raise SchemaNotCurrentError('0001', '0002')

    monkeypatch.setattr(schema_check, 'check_control_plane', stale)
    with pytest.raises(SystemExit) as exit_info:
        handler()
    assert exit_info.value.code != 0
    assert 'schema_not_current' in str(exit_info.value.code)


@pytest.mark.parametrize('handler', HANDLERS)
def test_an_unreachable_database_exits(handler, monkeypatch: pytest.MonkeyPatch) -> None:
    def unreachable(probes, **_: object) -> None:
        raise DependencyUnavailableError(['database'], 'database: OperationalError')

    monkeypatch.setattr(schema_check, 'wait_for_dependencies', unreachable)
    with pytest.raises(SystemExit) as exit_info:
        handler()
    assert 'dependency_unavailable: database' in str(exit_info.value.code)


@pytest.mark.parametrize('handler', HANDLERS)
def test_every_entry_point_uses_the_one_shared_check(
    handler, calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One helper holds the wait-then-check-then-exit sequence; no entry point carries a copy."""
    monkeypatch.setattr(schema_check, 'require_current_schema', lambda: calls.append('shared'))
    handler()
    assert calls == ['shared']
