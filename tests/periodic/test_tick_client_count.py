"""A tick builds its guarded OpenSearch client once, not once per project.

Each guarded client owns a ``ProjectRegistry`` that loads every project on its first request, so
the number of ``make_script_opensearch`` calls per tick is the number of registry loads.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fixtures.fake_valkey import FakeValkeySets

from src.services.periodic import image_counts_drain
from src.services.projects import guard


if TYPE_CHECKING:
    import pytest


class _Client:
    async def close(self) -> None:
        return None


def test_a_tick_over_five_projects_builds_one_guarded_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[int] = []

    def factory(*_: Any, **__: Any) -> _Client:
        built.append(1)
        return _Client()

    async def each(_lock: int, body: Any, *, task: str) -> None:
        for slug in 'abcde':
            await body(SimpleNamespace(slug=slug))

    async def drain(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(guard, 'make_script_opensearch', factory)
    monkeypatch.setattr(image_counts_drain, 'valkey_client', FakeValkeySets)
    monkeypatch.setattr(image_counts_drain, 'drain', drain)
    monkeypatch.setattr(image_counts_drain, 'for_each_project', each)
    monkeypatch.setattr(image_counts_drain, 'read_tick_projects', lambda: [1] * 5)
    asyncio.run(image_counts_drain.tick())
    assert len(built) == 1
