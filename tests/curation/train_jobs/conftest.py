"""Fixtures for the training-job service tests: a redirected jobs volume and a GPU allowlist."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.config import GpuArbiterConfig


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect OP_TRAIN_JOBS_DIR at the env-var level and return the actual resolved dir for
    the bound (``default``, tests/conftest.py) project.

    ``default`` is an ordinary project (no special-casing): ``project_jobs_dir()`` always nests
    ``/projects/<slug>``, so callers must not assume ``OP_TRAIN_JOBS_DIR`` itself is where files
    land. Every test reads and writes the resolved subdir.
    """
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path))
    resolved = tmp_path / 'projects' / 'default'
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


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
