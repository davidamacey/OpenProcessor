"""Fixtures of the beat tick tests: the job-runtime fakes (broker, wake-ups) and a queue config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fixtures.job_fixtures import job_env, kinds, producer, queue_hooks, wakes


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def queue_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A valid queue URL and password file, so the Celery app's configuration can load."""
    password = tmp_path / 'queue_password'
    password.write_text('test-password\n')
    monkeypatch.setenv('OP_QUEUE_URL', 'redis://valkey:6379/0')
    monkeypatch.setenv('OP_QUEUE_PASSWORD_FILE', str(password))


__all__ = ['job_env', 'kinds', 'producer', 'queue_env', 'queue_hooks', 'wakes']
