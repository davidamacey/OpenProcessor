"""`POST /curation/probe/run`, `GET /curation/probe/status`,
`POST /curation/probe/cancel`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest  # noqa: TC002
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client() -> TestClient:
    from src.core.db_session import get_db
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: AsyncMock()  # noqa: PLW0108  # FastAPI parses a class here as a dependency with parameters

    async def no_database():  # the refusals under test come before any job row
        yield None

    app.dependency_overrides[get_db] = no_database
    return TestClient(app).__enter__()


def _finished_status(checkpoint_path: str) -> SimpleNamespace:
    return SimpleNamespace(state='finished', checkpoint_path=checkpoint_path)


def test_probe_run_404_style_409_for_unknown_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.services.training.jobs.read_status', AsyncMock(return_value=None))
    r = _client().post('/curation/projects/default/probe/run', json={'job_id': 'nope'})
    assert r.status_code == 409, r.text


def test_probe_run_409_when_run_not_finished(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        AsyncMock(return_value=SimpleNamespace(state='running', checkpoint_path=None)),
    )
    r = _client().post('/curation/projects/default/probe/run', json={'job_id': 'run-1'})
    assert r.status_code == 409, r.text
    assert 'not finished' in r.json()['detail']


def test_probe_run_409_when_no_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        AsyncMock(return_value=SimpleNamespace(state='finished', checkpoint_path=None)),
    )
    r = _client().post('/curation/projects/default/probe/run', json={'job_id': 'run-1'})
    assert r.status_code == 409, r.text
    assert 'checkpoint_path' in r.json()['detail']


def test_probe_run_409_when_checkpoint_missing_on_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        AsyncMock(return_value=_finished_status('/no/such/file.onnx')),
    )
    r = _client().post('/curation/projects/default/probe/run', json={'job_id': 'run-1'})
    assert r.status_code == 409, r.text
    assert 'missing on disk' in r.json()['detail']
