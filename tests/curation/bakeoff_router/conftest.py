"""Fixtures for the bake-off router tests: a filesystem-backed environment and a client over it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _project_paths import default_train_jobs_dir
from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """Point every bake-off / training dir at tmp_path and stub the GPU arbiter."""
    from src.routers.curation import bakeoff
    from src.services.curation import bakeoff_resolve, eval_datasets
    from src.services.training import gpu_arbiter_docker

    dirs = {
        'exports': tmp_path / 'exports',
        'external': tmp_path / 'bakeoff_eval',
        'train_jobs_root': tmp_path / 'train_jobs',
        'runs': tmp_path / 'runs',
        'jobs': tmp_path / 'bakeoff_jobs',
        'out': tmp_path / 'bakeoff_out',
    }
    for key in ('exports', 'external', 'train_jobs_root', 'runs'):
        dirs[key].mkdir()
    # default's train_jobs_dir nests under the OP_TRAIN_JOBS_DIR root.
    dirs['train_jobs'] = default_train_jobs_dir(dirs['train_jobs_root'])
    monkeypatch.setattr(eval_datasets, 'export_root', lambda: dirs['exports'])
    monkeypatch.setattr(eval_datasets, 'external_root', lambda: dirs['external'])
    monkeypatch.setattr(bakeoff_resolve, 'RUNS_HOST_ROOT', dirs['runs'])
    # bakeoff.JOBS_DIR/OUT_DIR are now _jobs_dir()/_out_dir() -- resolved
    # per-project at call time (projects_plan.md §5.3) instead of module-
    # level constants -- so tests patch the functions themselves.
    monkeypatch.setattr(bakeoff, '_jobs_dir', lambda: dirs['jobs'])
    monkeypatch.setattr(bakeoff, '_out_dir', lambda: dirs['out'])
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(dirs['train_jobs_root']))
    for key in [k for k in __import__('os').environ if k.startswith('OP_BAKEOFF_PROFILE')]:
        monkeypatch.delenv(key)
    eval_datasets.clear_cache()

    class _Action:
        action = 'noop'

    async def _stop(**_kw: Any) -> _Action:
        return _Action()

    monkeypatch.setattr(gpu_arbiter_docker, 'stop_gpu_services', _stop)
    return dirs


@pytest.fixture
def client() -> TestClient:
    from src.routers.curation import router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    return TestClient(app)
