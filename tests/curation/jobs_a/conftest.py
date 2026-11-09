"""The job-test fixtures of ``tests/fixtures/job_fixtures.py`` for the lane-4A family tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def queue_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A valid queue URL and password file, so a lazily loaded config can resolve."""
    password = tmp_path / 'queue_password'
    password.write_text('test-password\n')
    monkeypatch.setenv('OP_QUEUE_URL', 'redis://valkey:6379/0')
    monkeypatch.setenv('OP_QUEUE_PASSWORD_FILE', str(password))


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
    'queue_env',
    'queue_hooks',
    'wakes',
    'worker_app',
]
