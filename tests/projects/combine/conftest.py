from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from families_b.support import queue_env, worker_app
from fixtures.job_fixtures import FakeProducer, job_env, kinds, producer, wakes

from src.services.projects import lifecycle

from .world import World


__all__ = [
    'FakeProducer',
    'job_env',
    'kinds',
    'producer',
    'queue_env',
    'wakes',
    'worker_app',
    'world',
]


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    async def _no_capacity(_client: Any) -> list[dict[str, str]]:
        return []

    monkeypatch.setattr(lifecycle, '_capacity_error_or_warning', _no_capacity)
    return World(tmp_path, monkeypatch)
