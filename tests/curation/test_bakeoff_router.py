"""Smoke tests for src/routers/curation/bakeoff.py.

The reference implementation's bake-off router has no dedicated test
file on the reference branch, so this is a new,
minimal smoke suite rather than a port — it exercises the read-only,
filesystem-driven endpoints (no OpenSearch dependency) with tmp_path
standing in for the shared jobs/output directories.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def app_client() -> TestClient:
    from src.routers.curation import router as kb_router

    app = FastAPI()
    app.include_router(kb_router)
    return TestClient(app)


def test_bakeoff_router_is_registered() -> None:
    from src.main import app

    assert any(r.path == '/curation/bakeoff/runs' for r in app.routes)
    assert any(r.path == '/curation/bakeoff/eval_datasets' for r in app.routes)


def test_bakeoff_runs_empty_when_out_dir_missing(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'OUT_DIR', tmp_path / 'does-not-exist')
    r = app_client.get('/curation/bakeoff/runs')
    assert r.status_code == 200
    assert r.json() == {'runs': []}


def test_bakeoff_runs_lists_finished_jobs_newest_first(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    out_dir = tmp_path / 'out'
    for job_id, started_at in (
        ('job-older', '2026-01-01T00:00:00Z'),
        ('job-newer', '2026-02-01T00:00:00Z'),
    ):
        job_dir = out_dir / job_id
        job_dir.mkdir(parents=True)
        (job_dir / 'status.json').write_text(
            json.dumps({'state': 'finished', 'models': [], 'started_at': started_at})
        )
    monkeypatch.setattr(bakeoff, 'OUT_DIR', out_dir)

    r = app_client.get('/curation/bakeoff/runs')
    assert r.status_code == 200
    job_ids = [row['job_id'] for row in r.json()['runs']]
    assert job_ids == ['job-newer', 'job-older']


def test_bakeoff_status_404_when_missing(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'OUT_DIR', tmp_path / 'out')
    r = app_client.get('/curation/bakeoff/status/no-such-job')
    assert r.status_code == 404


def test_bakeoff_results_404_when_missing(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'OUT_DIR', tmp_path / 'out')
    r = app_client.get('/curation/bakeoff/results/no-such-job')
    assert r.status_code == 404


def test_bakeoff_run_requires_models_or_quantize(app_client: TestClient) -> None:
    r = app_client.post('/curation/bakeoff/run', json={'dataset': '/some/path'})
    assert r.status_code == 400


def test_bakeoff_run_requires_dataset(app_client: TestClient) -> None:
    r = app_client.post(
        '/curation/bakeoff/run',
        json={'models': [{'backend': 'ultralytics', 'name': 'm1'}]},
    )
    assert r.status_code == 400


def test_bakeoff_eval_datasets_empty_when_roots_missing(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'EVAL_DATASET_ROOTS', [('curated', tmp_path / 'nope')])
    r = app_client.get('/curation/bakeoff/eval_datasets')
    assert r.status_code == 200
    assert r.json() == {'datasets': [], 'count': 0}


def test_bakeoff_baseline_models_empty_when_registry_missing(
    app_client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    from src.routers.curation import bakeoff

    monkeypatch.setattr(bakeoff, 'BASELINES_PATH', tmp_path / 'no-such-baselines.json')
    r = app_client.get('/curation/bakeoff/baseline_models')
    assert r.status_code == 200
    assert r.json() == {'baselines': [], 'count': 0}
