"""The job-test fixtures of ``tests/fixtures/job_fixtures.py`` for the lane-4A family tests."""

from __future__ import annotations

from typing import Any

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


@pytest.fixture
def worker_app(queue_env: None) -> Any:
    from src.tasks.app import app

    app.loader.import_default_modules()
    return app


__all__ = [
    'FakeProducer',
    'FakeQueueHooks',
    'job_env',
    'kinds',
    'producer',
    'queue_hooks',
    'wakes',
    'worker_app',
]
