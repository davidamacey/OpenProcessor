"""Capture published events without a database or Valkey.

``captured_events`` replaces ``publish`` and ``apublish`` and records each call in the shape the
browser sees (payload plus ``type`` and ``project``), so a unit test can assert on what a route
or service announced. Project resolution still runs, so an unbound caller still fails closed.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def captured_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    from src.services.events import publish as publish_module

    seen: list[dict[str, Any]] = []

    def record(event_type: str, payload: Any, *, project: Any = None, **_ignored: Any) -> None:
        # The private helpers keep the stamp and the fail-closed binding rule the real call has.
        target = publish_module._resolve(project)
        body = publish_module._stamp(payload)
        seen.append(
            {**body, 'type': event_type, 'project': None if target is None else str(target)}
        )

    async def arecord(
        event_type: str, payload: Any, *, project: Any = None, **_ignored: Any
    ) -> None:
        record(event_type, payload, project=project)

    async def arecord_many(events: Any, *, project: Any = None) -> None:
        for event_type, payload in events:
            record(event_type, payload, project=project)

    monkeypatch.setattr(publish_module, 'publish', record)
    monkeypatch.setattr(publish_module, 'apublish_many', arecord_many)
    monkeypatch.setattr(publish_module, 'apublish', arecord)
    return seen


@pytest.fixture(autouse=True)
def _no_event_writes_outside_db_tests(request: pytest.FixtureRequest) -> None:
    """Events are database rows now: only a ``db`` test may really write one."""
    if request.node.get_closest_marker('db') is None:
        request.getfixturevalue('captured_events')
