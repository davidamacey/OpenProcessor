"""Fixtures for the bake-off router tests: a filesystem-backed environment and a client over it."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from _project_paths import default_train_jobs_dir
from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from pathlib import Path


class _Started(dict[str, dict[str, Any]]):
    """Job spec (the job parameters) by bake-off job id, plus the ids of cancelled jobs."""

    def __init__(self) -> None:
        super().__init__()
        self.cancelled: list[uuid.UUID] = []
        self.ids: dict[str, uuid.UUID] = {}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Point every bake-off / training dir at tmp_path and stub the GPU arbiter and the job service.

    ``env['started']`` records the ``bakeoff`` jobs the route started (their parameters are the
    job spec v2); no database is behind the route in these tests.
    """
    from src.routers.curation import bakeoff
    from src.services.curation import bakeoff_jobs, bakeoff_resolve, eval_datasets
    from src.services.training import gpu_arbiter_docker

    dirs = {
        'exports': tmp_path / 'exports',
        'external': tmp_path / 'bakeoff_eval',
        'train_jobs_root': tmp_path / 'train_jobs',
        'runs': tmp_path / 'runs',
        'out': tmp_path / 'bakeoff_out',
    }
    for key in ('exports', 'external', 'train_jobs_root', 'runs'):
        dirs[key].mkdir()
    # default's train_jobs_dir nests under the OP_TRAIN_JOBS_DIR root.
    dirs['train_jobs'] = default_train_jobs_dir(dirs['train_jobs_root'])
    monkeypatch.setattr(eval_datasets, 'export_root', lambda: dirs['exports'])
    monkeypatch.setattr(eval_datasets, 'external_root', lambda: dirs['external'])
    monkeypatch.setattr(bakeoff_resolve, 'RUNS_HOST_ROOT', dirs['runs'])
    # The results dir is resolved per project at call time, so tests patch the function.
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

    started = _Started()

    async def _start_job(_session: Any, kind: str, project: Any, **kwargs: Any) -> Any:
        assert kind == 'bakeoff'
        spec = kwargs['params']
        job_id = uuid.uuid4()
        started[spec['job_id']] = spec
        started.ids[spec['job_id']] = job_id
        return SimpleNamespace(created=True, job=SimpleNamespace(id=job_id))

    async def _cancel_job(_session: Any, job_id: uuid.UUID, project: Any, **_kw: Any) -> None:
        started.cancelled.append(job_id)

    monkeypatch.setattr(bakeoff_jobs, 'start_job', _start_job)
    monkeypatch.setattr(bakeoff, 'cancel_job', _cancel_job)
    monkeypatch.setattr(bakeoff, 'current_project_id', uuid.uuid4)
    dirs['started'] = started  # type: ignore[assignment]
    return dirs


@pytest.fixture
def client() -> TestClient:
    from src.routers.curation import router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    # The registry-reading routes take the OpenSearch dependency; the in-memory registry
    # needs no cluster behind it.
    from src.routers.curation import _raw_opensearch_dep

    app.dependency_overrides[_raw_opensearch_dep] = lambda: None
    from src.core.db_session import get_db

    app.dependency_overrides[get_db] = lambda: None
    # The handler create_app registers: a backing-service failure is a 503.
    from src.core.error_handlers import upstream_unavailable_response
    from src.utils.upstream_errors import UpstreamError

    app.exception_handler(UpstreamError)(upstream_unavailable_response)
    return TestClient(app)
