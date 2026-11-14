"""The test PostgreSQL container is removed when it fails to start.

``DockerContainer.__enter__`` stops the container when ``start`` raises
(testcontainers-python, core/container.py); ``_ContainerHolder`` calls ``start`` directly, so a
container that was created and then failed its readiness wait stayed on the host until the
reaper found it.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from fixtures.postgres import (
    URL_ENV,
    DatabaseUnavailable,
    _ContainerHolder,
    _holder,
    _selects_db,
    pg_server,
)


class _FailingContainer:
    instances: list[_FailingContainer]

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.stopped = 0
        type(self).instances.append(self)

    def start(self) -> None:
        raise RuntimeError('readiness wait timed out')

    def stop(self) -> None:
        self.stopped += 1


def test_a_container_that_fails_to_start_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    _FailingContainer.instances = []
    module = types.ModuleType('testcontainers.community.postgres')
    module.PostgresContainer = _FailingContainer  # type: ignore[attr-defined]  # a stand-in module
    monkeypatch.setitem(sys.modules, 'testcontainers.community.postgres', module)

    holder = _ContainerHolder()
    holder.start()

    assert holder.server is None
    assert holder.error is not None
    assert 'readiness wait timed out' in holder.error
    assert [c.stopped for c in _FailingContainer.instances] == [1]


@pytest.mark.parametrize(
    ('markexpr', 'selected'),
    [
        ('', True),
        ('db', True),
        ('not live and not families', True),  # a db test carries neither mark: it runs
        ('db or live', True),
        ('not db', False),
        ('live', False),
        ('not live and not db', False),
    ],
)
def test_the_db_tier_is_selected_by_evaluating_the_mark_expression(
    markexpr: str, selected: bool
) -> None:
    assert _selects_db(markexpr) is selected


def test_a_worker_without_the_controllers_server_does_not_start_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(URL_ENV, raising=False)
    started: list[int] = []
    monkeypatch.setattr(_holder, 'start', lambda: started.append(1))
    request = types.SimpleNamespace(config=types.SimpleNamespace(workerinput={}))

    with pytest.raises(DatabaseUnavailable, match='controller'):
        pg_server.__wrapped__(request)

    assert started == []


def test_the_expressions_the_repo_uses_still_evaluate_through_pytests_matcher() -> None:
    # `_selects_db` relies on `_pytest.mark.expression.Expression`, which pytest does not
    # document as public (read from pytest 9.2.0). If an upgrade moves it, this test fails first.
    used = {
        'db': True,
        'not live and not families': True,
        'not db': False,
        '': True,
        'db and not live': True,
    }
    assert {expr: _selects_db(expr) for expr in used} == used
