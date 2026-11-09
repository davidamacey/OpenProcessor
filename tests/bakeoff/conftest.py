"""The bake-off tests share the fixtures of ``tests/fixtures/job_fixtures.py``."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_gpu_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the consumer builds a GPU app; it must not leave the inventory armed."""
    from src.tasks.runtime import gpu_app

    monkeypatch.setattr(gpu_app, '_gpu_worker_device', None)
