"""The test PostgreSQL container is removed when it fails to start.

``DockerContainer.__enter__`` stops the container when ``start`` raises
(testcontainers-python, core/container.py); ``_ContainerHolder`` calls ``start`` directly, so a
container that was created and then failed its readiness wait stayed on the host until the
reaper found it.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

from fixtures.postgres import _ContainerHolder


if TYPE_CHECKING:
    import pytest


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
