"""`POST /curation/probe/run`, `GET /curation/probe/status`,
`POST /curation/probe/cancel`."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation import probe_job


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch: pytest.MonkeyPatch):
    # Real filesystem state (not module memory) as of the multi-worker
    # fix -- point it at a per-test tmp_path rather than the real /jobs
    # mount.
    monkeypatch.setenv('OP_PROBE_JOBS_DIR', str(tmp_path / 'probe_jobs'))
    probe_job._reset_for_tests()
    yield
    probe_job._reset_for_tests()


def _client() -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: AsyncMock()
    # Entered as a context manager (not just constructed) so all requests
    # in a test share one persistent event loop/portal -- required for an
    # asyncio.create_task started by one request to still be running (and
    # cancellable) when a later request in the same test polls/cancels it.
    return TestClient(app).__enter__()


def _finished_status(checkpoint_path: str) -> SimpleNamespace:
    return SimpleNamespace(state='finished', checkpoint_path=checkpoint_path)


def test_probe_run_404_style_409_for_unknown_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.services.training.jobs.read_status', AsyncMock(return_value=None))
    r = _client().post('/curation/probe/run', json={'job_id': 'nope'})
    assert r.status_code == 409, r.text


def test_probe_run_409_when_run_not_finished(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        AsyncMock(return_value=SimpleNamespace(state='running', checkpoint_path=None)),
    )
    r = _client().post('/curation/probe/run', json={'job_id': 'run-1'})
    assert r.status_code == 409, r.text
    assert 'not finished' in r.json()['detail']


def test_probe_run_409_when_no_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        AsyncMock(return_value=SimpleNamespace(state='finished', checkpoint_path=None)),
    )
    r = _client().post('/curation/probe/run', json={'job_id': 'run-1'})
    assert r.status_code == 409, r.text
    assert 'checkpoint_path' in r.json()['detail']


def test_probe_run_409_when_checkpoint_missing_on_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        AsyncMock(return_value=_finished_status('/no/such/file.onnx')),
    )
    r = _client().post('/curation/probe/run', json={'job_id': 'run-1'})
    assert r.status_code == 409, r.text
    assert 'missing on disk' in r.json()['detail']


def test_probe_run_starts_and_status_reflects_it(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    weights = tmp_path / 'best.onnx'
    weights.write_bytes(b'fake')
    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        AsyncMock(return_value=_finished_status(str(weights))),
    )

    async def _fake_run_probe_inference(model_path, opensearch, **kwargs):
        await asyncio.sleep(0.05)
        return 7

    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    client = _client()
    r = client.post('/curation/probe/run', json={'job_id': 'run-1'})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['status'] == 'running'
    assert body['train_job_id'] == 'run-1'
    assert body['model_path'] == str(weights)

    r2 = client.get('/curation/probe/status')
    assert r2.status_code == 200, r2.text
    assert r2.json()['status'] in ('running', 'completed')


def test_probe_run_default_architecture_is_yolo26(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """G-22: yolo26 is the only trained family, so the default (and an
    explicit request for it) must not 422/fail -- a stale yolo11 default
    would silently mismatch every promoted model."""
    from src.routers.curation.probe import ProbeRunRequest

    assert ProbeRunRequest(job_id='run-1').architecture == 'yolo26'

    weights = tmp_path / 'best.onnx'
    weights.write_bytes(b'fake')
    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        AsyncMock(return_value=_finished_status(str(weights))),
    )
    seen_architectures: list[str] = []

    async def _fake_run_probe_inference(model_path, opensearch, **kwargs):
        seen_architectures.append(kwargs['architecture'])
        await asyncio.sleep(0.05)
        return 7

    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    client = _client()
    r = client.post('/curation/probe/run', json={'job_id': 'run-1', 'architecture': 'yolo26'})
    assert r.status_code == 200, r.text
    for _ in range(50):
        if client.get('/curation/probe/status').json()['status'] == 'completed':
            break
    assert seen_architectures == ['yolo26']


def test_probe_run_409_when_already_running(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    weights = tmp_path / 'best.onnx'
    weights.write_bytes(b'fake')
    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        AsyncMock(return_value=_finished_status(str(weights))),
    )
    started = asyncio.Event()

    async def _fake_run_probe_inference(model_path, opensearch, **kwargs):
        started.set()
        await asyncio.sleep(5)
        return 1

    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    client = _client()
    r1 = client.post('/curation/probe/run', json={'job_id': 'run-1'})
    assert r1.status_code == 200, r1.text
    r2 = client.post('/curation/probe/run', json={'job_id': 'run-2'})
    assert r2.status_code == 409, r2.text


def test_probe_cancel_stops_the_active_job(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    weights = tmp_path / 'best.onnx'
    weights.write_bytes(b'fake')
    monkeypatch.setattr(
        'src.services.training.jobs.read_status',
        AsyncMock(return_value=_finished_status(str(weights))),
    )

    async def _fake_run_probe_inference(model_path, opensearch, **kwargs):
        await asyncio.sleep(5)
        return 1

    monkeypatch.setattr(
        'src.services.curation.probe_predictions.run_probe_inference',
        _fake_run_probe_inference,
    )
    client = _client()
    client.post('/curation/probe/run', json={'job_id': 'run-1'})
    r = client.post('/curation/probe/cancel')
    assert r.status_code == 200, r.text
    assert r.json()['cancelled'] is True


def test_probe_cancel_with_nothing_running() -> None:
    r = _client().post('/curation/probe/cancel')
    assert r.status_code == 200, r.text
    assert r.json()['cancelled'] is False
