"""Tests for POST /curation/scores/compute + status/cancel/coverage
(curation-strategy plan §3.3/§9).

Job lifecycle: start -> 409 on double-start -> status -> cancel. The actual
scoring coroutine (``crop_scores.job.run_scoring_job``) is monkeypatched to
block on an ``asyncio.Event`` the test controls — a real background task
scheduled via ``asyncio.create_task`` would otherwise race the synchronous
TestClient calls (it could complete before the test's next HTTP call,
making a "double-start returns 409" assertion flaky). This is deliberate
test-side control, not a production behavior change — ``job.py``'s own
docstring documents the monkeypatch seam.

Every test redirects ``OP_SCORES_STATE_DIR`` to a per-test ``tmp_path`` so
nothing touches the real ``/jobs/scores`` volume (mirrors
``test_train_jobs.py``'s ``OP_TRAIN_JOBS_DIR`` convention).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as kb_router

    monkeypatch.setenv('OP_SCORES_STATE_DIR', str(tmp_path / 'scores'))
    monkeypatch.setenv('OP_SCORES_ENABLED', '1')
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))

    fake_os = AsyncMock()
    fake_os.count = AsyncMock(return_value={'count': 0})

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os

    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


def _block_run_scoring_job(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Monkeypatch run_scoring_job to hang on an Event the test controls,
    keeping the job 'running' deterministically across multiple sync
    TestClient calls. Returns the Event (never set — the job never
    actually completes during these tests, matching the lifecycle we want
    to observe: running -> cancel)."""
    from src.services.curation.item_scores import job as scores_job

    hang_forever = asyncio.Event()

    async def _fake_run_scoring_job(job_id, opensearch, scorer_names) -> None:
        scores_job._touch_heartbeat()
        await hang_forever.wait()

    monkeypatch.setattr(scores_job, 'run_scoring_job', _fake_run_scoring_job)
    return hang_forever


def test_start_then_double_start_409(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_run_scoring_job(monkeypatch)

    r1 = app_client.post('/curation/scores/compute', json={'scorers': ['uniqueness']})
    assert r1.status_code == 200, r1.text
    assert r1.json()['status'] == 'running'

    r2 = app_client.post('/curation/scores/compute', json={'scorers': ['uniqueness']})
    assert r2.status_code == 409


def test_status_reflects_running_job(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_run_scoring_job(monkeypatch)
    app_client.post('/curation/scores/compute', json={})

    r = app_client.get('/curation/scores/status')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'running'
    assert set(body['scorers']) == {'mistakenness', 'near_dup', 'uniqueness'}


def test_cancel_running_job(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation.item_scores import job as scores_job

    _block_run_scoring_job(monkeypatch)
    app_client.post('/curation/scores/compute', json={'scorers': ['near_dup']})

    r = app_client.post('/curation/scores/cancel')
    assert r.status_code == 200
    assert r.json()['cancelled'] is True
    assert scores_job.is_cancelled()


def test_cancel_with_no_job_running(app_client: TestClient) -> None:
    r = app_client.post('/curation/scores/cancel')
    assert r.status_code == 200
    assert r.json()['cancelled'] is False


def test_status_idle_with_no_job(app_client: TestClient) -> None:
    r = app_client.get('/curation/scores/status')
    assert r.status_code == 200
    assert r.json()['status'] == 'idle'


def test_unknown_scorer_400(app_client: TestClient) -> None:
    r = app_client.post('/curation/scores/compute', json={'scorers': ['not_a_real_scorer']})
    assert r.status_code == 400
    assert 'not_a_real_scorer' in r.json()['detail']


def test_disabled_when_flag_off_400(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_SCORES_ENABLED', raising=False)
    r = app_client.post('/curation/scores/compute', json={'scorers': ['uniqueness']})
    assert r.status_code == 400
    assert 'disabled' in r.json()['detail']


def test_coverage_reports_per_field_counts(app_client: TestClient) -> None:
    async def _count(index, body):
        query = body['query']
        if 'exists' not in query:
            return {'count': 100}  # total-crops count (match_all)
        field = query['exists']['field']
        return {'count': 7} if field == 'uniqueness_score' else {'count': 3}

    app_client.fake_os.count = AsyncMock(side_effect=_count)  # type: ignore[attr-defined]

    r = app_client.get('/curation/scores/coverage')
    assert r.status_code == 200
    coverage = r.json()['coverage']
    assert set(coverage) == {'uniqueness', 'near_dup', 'mistakenness'}
    assert coverage['uniqueness']['n_scored'] == 7
    assert coverage['uniqueness']['field'] == 'uniqueness_score'
    assert coverage['uniqueness']['total'] == 100
    assert coverage['near_dup']['n_scored'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
