"""Shared fixtures for the Celery app tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def queue_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A valid queue URL and password file, so a lazily loaded config can resolve."""
    password = tmp_path / 'queue_password'
    password.write_text('test-password\n')
    monkeypatch.setenv('OP_QUEUE_URL', 'redis://valkey:6379/0')
    monkeypatch.setenv('OP_QUEUE_PASSWORD_FILE', str(password))
