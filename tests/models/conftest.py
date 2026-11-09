"""Fixtures of the model-export tests: the job fixtures, an export handler on fakes."""

from __future__ import annotations

import pytest
from fixtures.export_env import ExportEnv, export_env, plan_facts
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
    """Importing the export app arms the GPU inventory; a test must not leave it armed."""
    from src.tasks.runtime import gpu_app

    monkeypatch.setattr(gpu_app, '_gpu_worker_device', None)


__all__ = [
    'ExportEnv',
    'FakeProducer',
    'FakeQueueHooks',
    'export_env',
    'job_env',
    'kinds',
    'plan_facts',
    'producer',
    'queue_hooks',
    'wakes',
]
