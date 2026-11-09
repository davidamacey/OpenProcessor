from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from families_b.support import worker_app
from fixtures.job_fixtures import FakeProducer, job_env, kinds, producer, wakes

from .world import World


__all__ = [
    'FakeProducer',
    'job_env',
    'kinds',
    'producer',
    'wakes',
    'worker_app',
    'world',
]


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    return World(tmp_path, monkeypatch)
