"""Fixtures for the training-job service tests: a GPU allowlist."""

from __future__ import annotations

import pytest

from src.config import GpuArbiterConfig


@pytest.fixture
def restricted_gpu_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure an allowlist of {0, 2} -- this deployment's
    two A6000s -- for tests that exercise the *restrictive* path. The
    generic default (no fixture) is permissive: any GPU id validates."""
    import src.config.gpu_arbiter as gpu_arbiter_config_module

    monkeypatch.setattr(
        gpu_arbiter_config_module,
        '_default_gpu_arbiter_config',
        GpuArbiterConfig(allowed_gpu_ids=frozenset({0, 2})),
    )
