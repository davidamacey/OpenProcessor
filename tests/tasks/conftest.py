"""Shared fixtures for the Celery app tests."""

from __future__ import annotations

import pytest
from fixtures.job_fixtures import (
    FakeProducer,
    FakeQueueHooks,
    job_env,
    kinds,
    producer,
    queue_hooks,
    wakes,
)


@pytest.fixture(autouse=True)
def _no_gpu_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that builds a GPU app must not leave the inventory armed for the next one."""
    from src.tasks.runtime import gpu_app

    monkeypatch.setattr(gpu_app, '_gpu_worker_device', None)


__all__ = [
    'FakeProducer',
    'FakeQueueHooks',
    'job_env',
    'kinds',
    'producer',
    'queue_hooks',
    'wakes',
]
